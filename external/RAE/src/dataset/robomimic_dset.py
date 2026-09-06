import os
import hydra
import h5py
import torch
import numpy as np
from einops import rearrange
from termcolor import cprint
from pathlib import Path
from typing import Optional, Callable
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt

# import sys
# sys.path.append(str(Path(__file__).parent.parent))
from dataset.traj_dset import TrajDataset, TrajSlicerDataset, TrajSubset, TrajMerge
from envs.robomimic.robomimic_utils import *


class RoboMimicDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/robomimic_datasets",
        task_name: str = "square",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        done_mode: int = 1,
        shape_rewards: bool = True,
        img_size: int = 64,
        max_traj_length: int = 200,
    ):
        self.data_path = data_path
        self.transform = transform
        self.normalize_action = normalize_action
        
        self.task_name = task_name
        self.done_mode = done_mode
        self.shape_rewards = shape_rewards
        self.img_size = img_size
        self.max_traj_length = max_traj_length
        self.n_rollout = n_rollout
        
        self.state_dim = None
        self.action_dim = None
        self.pixel_keys = None
        self.state_keys = None
        
        # states: [N, T, state_dim]
        # actions: [N, T, action_dim]
        # obs_imgs_dict: {key: [N, T, H, W, C]}
        self.states, self.actions, self.obs_imgs_dict, self.seq_lengths = self._load_data()
        
        # dummy proprio
        self.proprios = torch.zeros(
            (self.states.shape[0], self.states.shape[1], 1)
        )  
        self.proprio_dim = self.proprios.shape[-1]
        
        # normalization
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
        
        
    def _load_data(self):
        # Load the RoboMimic h5py files
        if str(self.data_path).endswith('.hdf5'):
            dataset_path = str(self.data_path)
            shape_meta = create_shape_meta(self.img_size, include_state=True)
        else:
            dataset_path, _, shape_meta = get_dataset_path_and_meta_info(
                env_id=self.task_name,
                shaped=self.shape_rewards,
                image_size=self.img_size,
                done_mode=self.done_mode,
                datadir=str(self.data_path),
            )
        cprint(f"...Loading RoboMimic dataset from {dataset_path}...", "green")
        
        f = h5py.File(dataset_path, "r")
        demos = list(f["data"].keys())
        demos = sorted(demos, key=lambda x: int(x.split('_')[1]))
        if self.n_rollout is not None:
            demos = demos[: self.n_rollout]
        
        obs_keys = shape_meta["obs"].keys()
        pixel_keys = sorted([key for key in obs_keys if "image" in key])
        state_keys = sorted([key for key in obs_keys if "image" not in key])
        self.pixel_keys = pixel_keys
        self.state_keys = state_keys

        # Set state_dim and action_dim
        first_demo = f["data"][demos[0]]
        action_dim = first_demo["actions"].shape[1]
        if 'state' in first_demo['obs'].keys():
            state_dim = first_demo['obs']['state'].shape[1]
        else:
            state_dim = 0
            for key in state_keys:
                state_dim += np.prod(first_demo["obs"][key].shape[1:])
        self.state_dim = state_dim
        self.action_dim = action_dim
        cprint(f"State dim: {state_dim}, Action dim: {action_dim}", "yellow")
        
        # Fill the dataset
        seq_length_list = []
        states_list = []
        actions_list = []
        obs_imgs_dict = {key: [] for key in pixel_keys}
        for i, demo in enumerate(tqdm(demos, desc="Loading demos")):
            states, actions, obs_imgs, seq_length, _ = add_traj_to_cache(
                demo, f, pixel_keys, state_keys
            )
            states_list.append(states)
            actions_list.append(actions)
            seq_length_list.append(seq_length)
            for key in pixel_keys:
                obs_imgs_dict[key].append(obs_imgs[key])
                   
        seq_lengths = torch.tensor(seq_length_list, dtype=torch.long)
        max_seq_length = torch.max(seq_lengths).item()
        assert max_seq_length <= self.max_traj_length, f"Max sequence length {max_seq_length} exceeds limit {self.max_traj_length}"
        
        # Padding states and actions
        states_tensor = torch.zeros((len(demos), self.max_traj_length, self.state_dim), dtype=torch.float32)
        actions_tensor = torch.zeros((len(demos), self.max_traj_length, self.action_dim), dtype=torch.float32)
        for i in range(len(demos)):
            T = seq_lengths[i]
            states_tensor[i, :T, :] = torch.tensor(states_list[i], dtype=torch.float32)
            actions_tensor[i, :T, :] = torch.tensor(actions_list[i], dtype=torch.float32)
        
        # Padding obs images
        obs_imgs_tensor_dict = {}
        for key in pixel_keys:
            obs_imgs_tensor = torch.zeros((len(demos), self.max_traj_length, *obs_imgs_dict[key][0].shape[1:]), dtype=torch.float32)
            for i in range(len(demos)):
                T = seq_lengths[i]
                obs_imgs_tensor[i, :T, :] = torch.tensor(obs_imgs_dict[key][i], dtype=torch.float32)
            obs_imgs_tensor_dict[key] = obs_imgs_tensor
        
        return states_tensor, actions_tensor, obs_imgs_tensor_dict, seq_lengths
    
    
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
        state = self.states[idx, frames]
        act = self.actions[idx, frames]
        proprio = self.proprios[idx, frames]
        
        obs = {}
        obs["proprio"] = proprio
        
        #TODO: support multiple pixel keys
        # for key in self.pixel_keys:
        #     img = self.obs_imgs_dict[key][idx, frames]          # (T, H, W, C)
        #     img = rearrange(img, "T H W C -> T C H W") / 255.0  # (T, C, H, W)
        #     if self.transform:
        #         img = self.transform(img)
        #     obs[key] = img
        
        # Now: take agent view only as visual input
        img = self.obs_imgs_dict[self.pixel_keys[0]][idx, frames]   # (T, H, W, C)
        img = rearrange(img, "T H W C -> T C H W") / 255.0          # (T, C, H, W)
        if self.transform:
            img = self.transform(img)
        obs['visual'] = img

        return obs, act, state, {} # infos is None

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
    
    def __len__(self):
        return len(self.seq_lengths)
    


def load_robomimic_slice_train_val(
    transform,
    n_rollout=50,
    data_path='/data/robomimic_datasets',
    task_name='square',
    normalize_action=True,
    done_mode=1,
    shape_rewards=True,
    img_size=64,
    num_exp_trajs=50,
    num_exp_val_trajs=10,
    max_traj_length=200,
    num_hist=0,
    num_pred=0,
    frameskip=0,
): 
    # Construct Dataset
    if isinstance(data_path, str):
        dset = RoboMimicDataset(
            data_path=data_path,
            task_name=task_name,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            done_mode=done_mode,
            shape_rewards=shape_rewards,
            img_size=img_size,
            max_traj_length=max_traj_length,
        )
        
        train_dset = TrajSubset(dset, list(range(num_exp_trajs)))
        val_dset = TrajSubset(dset, list(range(len(dset) - num_exp_val_trajs, len(dset))))
        
    elif OmegaConf.is_list(data_path):
        assert len(data_path) == len(num_exp_trajs) == len(num_exp_val_trajs), "Length of data_path, num_exp_trajs, num_exp_val_trajs must be the same"
        train_dsets = []
        val_dsets = []
        for i in range(len(data_path)):
            dset = RoboMimicDataset(
                data_path=data_path[i],
                task_name=task_name,
                n_rollout=n_rollout,
                transform=transform,
                normalize_action=normalize_action,
                done_mode=done_mode,
                shape_rewards=shape_rewards,
                img_size=img_size,
                max_traj_length=max_traj_length,
            )
            
            train_dset_i = TrajSubset(dset, list(range(num_exp_trajs[i])))
            val_dset_i = TrajSubset(dset, list(range(len(dset) - num_exp_val_trajs[i], len(dset))))
            train_dsets.append(train_dset_i)
            val_dsets.append(val_dset_i)
        
        # Merge datasets
        train_dset = train_dsets[0]
        val_dset = val_dsets[0]
        for i in range(1, len(train_dsets)):
            train_dset = TrajMerge(train_dset, train_dsets[i])
            val_dset = TrajMerge(val_dset, val_dsets[i])
        
    else:
        raise TypeError(f"data_path must be str or ListConfig, got {type(data_path).__name__}")
    
    # Construct Sliced Dataset
    # obs, act, state
    # state: [num_frames, state_dim]
    # act: [num_frames, action_dim * frameskip]
    # obs: {key: [num_frames, C, H, W]}, proprio: [num_frames, proprio_dim]
    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)
    cprint(f"Train slices: {len(train_slices)}, Val slices: {len(val_slices)}", "yellow")
    
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
@hydra.main(config_path="../confings/env", config_name="robomimic")
def test_robomimic_dataset(cfg: OmegaConf):
    num_hist = 3
    num_pred = 1
    frameskip = 5
    datasets, traj_dsets = hydra.utils.call(
        cfg.dataset,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
    )
    
    # visualize the sequences
    pixel_keys = traj_dsets["train"].pixel_keys
    train_dataset = datasets["train"]
    obs = train_dataset[0][0][pixel_keys[0]] # (num_frames, C, H, W)
    obs = rearrange(obs, "T C H W -> T H W C").numpy()  # (num_frames, H, W, C)
    
    plt.figure(figsize=(15, 5))
    for i in range(obs.shape[0]):
        plt.subplot(1, obs.shape[0], i + 1)
        plt.imshow(obs[i])
        plt.axis('off')
    plt.tight_layout()
    
    save_dir = '/coc/flash7/bli678/projects/egowm/external/dino_wm/demos'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'robomimic_demo.png'), bbox_inches='tight')
    plt.close()    
    
    cprint("--------Finished Everything--------", "green", attrs=["bold"])


if __name__ == "__main__":
    test_robomimic_dataset()    
