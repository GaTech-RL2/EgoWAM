import os
import time
import hydra
import torch
import numpy as np
import imageio
from PIL import Image
from einops import rearrange
from termcolor import cprint
from pathlib import Path
from typing import Optional, Callable
from omegaconf import OmegaConf
import psutil
from tqdm import tqdm

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import sys
sys.path.append(str(Path(__file__).parent.parent))
from dataset.traj_dset import TrajDataset, TrajSlicerDataset, TrajSubset, TrajMerge


def monitor_memory():
    """Monitor memory usage of the current process."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)  # Convert bytes to MB


class EgoVerseDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "/coc/flash7/mlin365/data/egoverse",
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        valid_ratio: float = 0.1,
        include_dex_keypoints: bool = False,
        include_head_pose: bool = False,
        mode: str = "train",
        seed: int = 0,
    ):
        self.hf_data_path = data_path
        self.transform = transform
        self.normalize_action = normalize_action
        self.valid_ratio = valid_ratio
        self.mode = mode
        self.seed = seed
        
        self.include_dex_keypoints = include_dex_keypoints
        self.include_head_pose = include_head_pose
        
        self.data_name = self.hf_data_path.split('/')[-1]
        self.repo_id = "bli678/egowm"
        
        # TODO: set the data_root_path as argument
        data_root_path = "data/egoverse/processed_wm_v2/put_cup_on_saucer"
        os.makedirs(data_root_path, exist_ok=True)
        self.data_path = os.path.join(data_root_path, self.data_name)
        
        self._load_dataset()
        
        if os.path.exists(os.path.join(self.data_path, f"{self.mode}.pth")):
            cprint(f"Using local dataset path: {self.data_path}", "yellow")
            # load pre-saved data tensors
            data_file = os.path.join(self.data_path, f"{self.mode}.pth")
            data_dict = torch.load(data_file)
            self.states = data_dict["states"]
            self.actions = data_dict["actions"]
            self.images = data_dict["images"]
            self.proprios = data_dict["proprios"]
            
            self.hand_keypoints = None
            if self.include_dex_keypoints and "hand_keypoints" in data_dict:
                self.hand_keypoints = data_dict["hand_keypoints"]
        
        else:
            # states:   [N, T, state_dim]
            # actions:  [N, T, action_dim]
            # images:   [N, T, C, H, W]
            # proprios: [N, T, proprio_dim]
            # hand_keypoints: [N, T, 126] if include_dex_keypoints else None
            self.states, self.actions, self.images, self.proprios, self.hand_keypoints = self._load_data()
        
        # normalization
        # TODO: normalized based on meta
        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(
                self.actions, self.seq_lengths
            )
            self.state_mean, self.state_std = self.get_data_mean_std(
                self.states, self.seq_lengths
            )
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(
                self.proprios, self.seq_lengths
            )
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
        
        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
            
        
    def _load_dataset(self):
        # Load dataset
        time_start = time.time()
        dataset_meta = LeRobotDatasetMetadata(
            repo_id=self.repo_id, root=self.hf_data_path, local_files_only=True
        )
        
        # # random split
        # dataset_meta._update_splits(valid_ratio=self.valid_ratio, seed=self.seed)
        # dataset_splits = dataset_meta.info["splits"]
        # if self.mode == "train":
        #     indices = dataset_splits["train"]
        # elif self.mode == "valid":
        #     indices = dataset_splits["valid"]
        # else:
        #     raise ValueError(f"Unknown mode {self.mode}")
        
        total_episodes = dataset_meta.total_episodes
        num_valid = int(self.valid_ratio * total_episodes)
        num_train = total_episodes - num_valid
        if self.mode == "train":
            indices = list(range(0, num_train))
        elif self.mode == "valid":
            indices = list(range(num_train, total_episodes))
        
        dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.hf_data_path,
            local_files_only=True,
            episodes=indices,
        )
        self.dataset = dataset
        time_end = time.time()
        
        cprint(f"Loaded {self.data_name} dataset in {time_end - time_start:.2f} seconds.", "cyan")
        print(f"Number of episodes: {dataset.num_episodes}, Number of frames: {dataset.num_frames}")
        dataset_keys = list(dataset_meta.features.keys())
        
        # Set valid keys for state, action, pixel
        self.state_keys = [key for key in dataset_keys if "observations.state" in key]
        self.pixel_keys = [key for key in dataset_keys if "observations.images" in key]
        self.action_keys = [key for key in dataset_keys if "actions" in key]

        if self.include_dex_keypoints: # for human data only for now
            self.valid_state_keys = self.state_keys[:2]
            self.valid_action_keys = self.action_keys[:2] 
        else:
            self.valid_state_keys = [k for k in self.state_keys if "ee_pose" in k]
            self.valid_action_keys = [k for k in self.action_keys if "cartesian" in k]
        self.valid_pixel_keys = self.pixel_keys[:1] # TODO: only use agent view for now
        assert len(self.valid_pixel_keys) == 1, "Only support one pixel key for now."

        # Set state_dim and action_dim
        first_demo = dataset[0]
        state = [first_demo[k] for k in self.valid_state_keys]
        state = np.concatenate(state, axis=-1)              # (state_dim,)
        
        action = [first_demo[k] for k in self.valid_action_keys]
        action = np.concatenate(action, axis=1)             # (chunk_length, action_dim)
        
        head_pose = [first_demo[k] for k in self.state_keys if "head_pose" in k]
        if len(head_pose) != 0 and self.include_head_pose:
            head_pose = np.concatenate(head_pose, axis=-1)  # (head_pose_dim, )
            self.proprio_dim = head_pose.shape[-1]
        else:
            self.proprio_dim = int(1)

        self.state_dim = state.shape[-1]
        self.action_dim = action.shape[-1]
        cprint(f"State dim: {self.state_dim}, Action dim: {self.action_dim}, Proprioception dim: {self.proprio_dim}", "cyan")

        # Get sequence lengths
        self.seq_lengths = [int(dataset.meta.episodes[eid]['length']) for eid in dataset.episodes]
        self.seq_lengths = torch.tensor(self.seq_lengths, dtype=torch.long)
    
    
    def _load_data(self):
        ds = self.dataset.hf_dataset.with_format("torch")

        if self.include_head_pose:
            hp_keys = [k for k in self.state_keys if "head_pose" in k]
        else:
            hp_keys = []

        # Get hand_keypoints key if needed
        hand_keypoints_key = None
        if self.include_dex_keypoints:
            hand_keypoints_keys = [k for k in self.state_keys if "hand_keypoints" in k]
            if len(hand_keypoints_keys) > 0:
                hand_keypoints_key = hand_keypoints_keys[0]

        states_list, actions_list, images_list, propros_list = [], [], [], []
        hand_keypoints_list = [] if self.include_dex_keypoints else None

        for ep_idx in tqdm(range(self.dataset.num_episodes), desc="Extracting data from episodes"):
            start = self.dataset.episode_data_index["from"][ep_idx].item()
            end   = self.dataset.episode_data_index["to"][ep_idx].item()
            
            batch = ds[start:end]
            T = batch[self.valid_state_keys[0]].shape[0]

            # ---------- states ----------
            state_cols = [batch[k] for k in self.valid_state_keys]
            states = torch.cat(state_cols, dim=-1)                  # (T, state_dim)

            # ---------- actions ----------
            act_cols = [batch[k] for k in self.valid_action_keys]
            actions_interp = torch.cat(act_cols, dim=-1)            # (T, L, action_dim)
            actions = actions_interp[:, 1] - actions_interp[:, 0]   # (T, action_dim)

            # ---------- images ----------
            img_key = self.valid_pixel_keys[0]
            imgs = batch[img_key]                                   # (T, C, H, W) uint8

            # ---------- proprio ----------
            if self.include_head_pose:
                hp_cols = [batch[k] for k in hp_keys]               
                proprio = torch.cat(hp_cols, dim=-1)                # (T, head_pose_dim)
            else:
                proprio = torch.zeros((T, 1), dtype=torch.float32)

            # ---------- hand_keypoints ----------
            if self.include_dex_keypoints and hand_keypoints_key is not None:
                hand_keypoints = batch[hand_keypoints_key]           # (T, 126) - 21 keypoints * 2 hands * 3 coords
                hand_keypoints = hand_keypoints.float()              # Ensure float type
                hand_keypoints_list.append(hand_keypoints)

            states_list.append(states)        # (T, state_dim)
            actions_list.append(actions)      # (T, action_dim)
            images_list.append(imgs)          # (T, C, H, W)
            propros_list.append(proprio)      # (T, p_dim)
 
        states_tensor = torch.stack(states_list, dim=0)     # (N, T, state_dim)
        actions_tensor = torch.stack(actions_list, dim=0)   # (N, T, action_dim)
        images_tensor = torch.stack(images_list, dim=0)     # (N, T, C, H, W)
        propros_tensor = torch.stack(propros_list, dim=0)   # (N, T, p_dim)
        
        # Stack hand_keypoints if available
        if self.include_dex_keypoints and hand_keypoints_list:
            hand_keypoints_tensor = torch.stack(hand_keypoints_list, dim=0)  # (N, T, 126)
        else:
            hand_keypoints_tensor = None
        
        # save the data tensors for faster loading next time
        os.makedirs(self.data_path, exist_ok=True)
        save_data_path = os.path.join(self.data_path, f"{self.mode}.pth")
        save_dict = {
            "states": states_tensor,
            "actions": actions_tensor,
            "images": images_tensor,
            "proprios": propros_tensor
        }
        if hand_keypoints_tensor is not None:
            save_dict["hand_keypoints"] = hand_keypoints_tensor
        torch.save(save_dict, save_data_path)
        print(f"Saved processed data tensors to {save_data_path}")

        return states_tensor, actions_tensor, images_tensor, propros_tensor, hand_keypoints_tensor
    
    
    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0) + 1e-6
        return data_mean, data_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]
    
    def get_frames(self, idx, frames):
        state = self.states[idx, frames]       # [T, state_dim]
        act = self.actions[idx, frames]        # [T, action_dim]
        proprio = self.proprios[idx, frames]   # [T, proprio_dim]
        
        img = self.images[idx, frames] / 255.0  # [T, C, H, W]
        if self.transform:
            img = self.transform(img)
            
        obs = {}
        obs["proprio"] = proprio
        obs['visual'] = img

        infos = {}
        if self.include_dex_keypoints and self.hand_keypoints is not None:
            hand_keypoints = self.hand_keypoints[idx, frames]  # [T, 126]
            infos["hand_keypoints"] = hand_keypoints

        return obs, act, state, infos

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
    
    def __len__(self):
        return len(self.seq_lengths)


def load_egoverse_slice_train_val(
    transform,
    data_path='/data/egoverse_datasets',
    normalize_action=False,
    valid_ratio=0.1,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    include_dex_keypoints=False,
    include_head_pose=False,
    filter=None,
    mode="train",
): 
    processed_pathes = sorted(os.listdir(data_path))
    if filter is not None:
        processed_pathes = [p for p in processed_pathes if filter in p]
    cprint(f"Found {len(processed_pathes)} processed datasets.", "yellow", attrs=["bold"])
    
    train_dsets, val_dsets = [], []
    time_start = time.time()
    for p in processed_pathes:
        cprint(f"Loading dataset from {p}", "yellow")
        data_path_p = os.path.join(data_path, p)
        if mode == "train":
            train_dset_i = EgoVerseDataset(
                data_path=data_path_p,
                transform=transform,
                normalize_action=normalize_action,
                valid_ratio=valid_ratio,
                include_dex_keypoints=include_dex_keypoints,
                include_head_pose=include_head_pose,
                mode = "train",
            )
            train_dsets.append(train_dset_i)
        
        val_dset_i = EgoVerseDataset(
            data_path=data_path_p,
            transform=transform,
            normalize_action=normalize_action,
            valid_ratio=valid_ratio,
            include_dex_keypoints=include_dex_keypoints,
            include_head_pose=include_head_pose,
            mode = "valid",
        )
        val_dsets.append(val_dset_i)
        cprint(f"Memory after loading dataset: {monitor_memory():.2f} MB", "cyan")

    time_end = time.time()
    cprint(f"Loaded all datasets in {time_end - time_start:.2f} seconds.", "yellow", attrs=["bold"])
    cprint(f"Memory after loading dataset: {monitor_memory():.2f} MB", "cyan")
    
    # Merge all datasets
    val_dset = val_dsets[0]
    if len(val_dsets) > 1:
        for i in range(1, len(val_dsets)):
            val_dset = TrajMerge(val_dset, val_dsets[i])
    
    if mode == "train":
        train_dset = train_dsets[0]
        if len(train_dsets) > 1:
            for i in range(1, len(train_dsets)):
                train_dset = TrajMerge(train_dset, train_dsets[i])
    else:
        train_dset = []

    # Construct Sliced Dataset
    # obs, act, state
    # state: [num_frames, state_dim]
    # act: [num_frames, action_dim * frameskip]
    # obs: {key: [num_frames, C, H, W]}, proprio: [num_frames, proprio_dim]
    num_frames = num_hist + num_pred
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)
    if mode == "train":
        train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    else:
        train_slices = []
    cprint(f"Train slices: {len(train_slices)}, Val slices: {len(val_slices)}", "yellow")
    cprint(f"Memory after loading dataset: {monitor_memory():.2f} MB", "cyan")
    
    # Return datasets
    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    
    return datasets, traj_dset


################################
# Main Test
################################
@hydra.main(config_path="./", config_name="egoverse")
def test_egoverse_dataset(cfg: OmegaConf):
    num_hist = 3
    num_pred = 1
    frameskip = 5
    datasets, traj_dsets = hydra.utils.call(
        cfg.dataset,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
    )
    
    train_dataset = datasets['train']
    train_traj_dset = traj_dsets['train']
    data = train_dataset[0]
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    # data = next(iter(dataloader))
    
    cprint("--------Finished Everything--------", "green", attrs=["bold"])


if __name__ == "__main__":
    test_egoverse_dataset()   