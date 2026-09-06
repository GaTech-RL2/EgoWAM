import os
import time
import random
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import torch
import numpy as np
from tqdm import tqdm
from termcolor import cprint

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import sys
import glob
sys.path.append(str(Path(__file__).parent.parent))
from dataset.traj_dset import TrajDataset, TrajMerge
from omegaconf import OmegaConf
import hydra



class EgoVerseImageOnlyDataset(TrajDataset):
    """
    EgoVerse dataset that ONLY keeps image observations for reconstruction.
    It does not store states actions proprios or keypoints.

    __getitem__ returns a single image tensor (C,H,W) in [0,1].
    """

    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        valid_ratio: float = 0.1,
        mode: str = "train",
        seed: int = 0,
        repo_id: str = "bli678/egowm",
        cache_root: str = "data/egoverse/processed_wm_v2/recon_images_only",
        pixel_key_idx: int = 0,
    ):
        self.hf_data_path = data_path
        self.transform = transform
        self.valid_ratio = float(valid_ratio)
        self.mode = mode
        self.seed = int(seed)
        self.repo_id = repo_id


        self.data_name = self.hf_data_path.rstrip("/").split("/")[-1]
        os.makedirs(cache_root, exist_ok=True)
        self.cache_dir = os.path.join(cache_root, self.data_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.pixel_key_idx = int(pixel_key_idx)

        self._load_dataset()

        cache_path = os.path.join(self.cache_dir, f"{self.mode}.pt")
        if os.path.exists(cache_path):
            cprint(f"Loading cached image-only tensors from {cache_path}", "yellow")
            blob = torch.load(cache_path, map_location="cpu")
            self.images_list: List[torch.Tensor] = blob["images_list"]     # list of (T,C,H,W) uint8
            self.seq_lengths: torch.Tensor = blob["seq_lengths"]           # (N,) long
            self.frame_index: torch.Tensor = blob["frame_index"]           # (F,2) long
            self.pixel_key: str = blob["pixel_key"]
        else:
            self.images_list, self.seq_lengths = self._load_images_only()
            self.frame_index = self._build_frame_index(self.seq_lengths)
            torch.save(
                {
                    "images_list": self.images_list,
                    "seq_lengths": self.seq_lengths,
                    "frame_index": self.frame_index,
                    "pixel_key": self.pixel_key,
                },
                cache_path,
            )
            cprint(f"Saved cached flattened image index to {cache_path}", "cyan")


    def _load_dataset(self):
        t0 = time.time()
        meta = LeRobotDatasetMetadata(repo_id=self.repo_id, root=self.hf_data_path, local_files_only=True)

        total_episodes = meta.total_episodes
        num_valid = max(int(self.valid_ratio * total_episodes), 1)
        num_train = total_episodes - num_valid

        if self.mode == "train":
            episodes = list(range(0, num_train))
        elif self.mode == "valid":
            episodes = list(range(num_train, total_episodes))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.hf_data_path,
            local_files_only=True,
            episodes=episodes,
        )
        self.dataset = dataset

        dataset_keys = list(meta.features.keys())
        pixel_keys = [k for k in dataset_keys if "observations.images" in k]
        assert len(pixel_keys) > 0, "No observations.images keys found in dataset metadata."

        # pick one camera stream
        assert 0 <= self.pixel_key_idx < len(pixel_keys), f"pixel_key_idx out of range. got {self.pixel_key_idx}, have {len(pixel_keys)} keys"
        self.pixel_key = pixel_keys[self.pixel_key_idx]

        # sequence lengths per episode
        seq_lengths = [int(dataset.meta.episodes[eid]["length"]) for eid in dataset.episodes]
        self.seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)

        t1 = time.time()
        cprint(f"Loaded metadata for {self.data_name} in {t1 - t0:.2f}s", "cyan")
        print(f"Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}, Pixel key: {self.pixel_key}")

    def _load_images_only(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        ds = self.dataset.hf_dataset.with_format("torch")

        images_list: List[torch.Tensor] = []
        seq_lengths_list: List[int] = []

        for ep_idx in tqdm(range(self.dataset.num_episodes), desc=f"Extracting images ({self.mode})"):
            start = self.dataset.episode_data_index["from"][ep_idx].item()
            end = self.dataset.episode_data_index["to"][ep_idx].item()

            batch = ds[start:end]
            imgs = batch[self.pixel_key]  # (T,C,H,W) uint8

            # store as uint8 to save memory; convert to float later
            imgs = imgs.contiguous().cpu()
            images_list.append(imgs)
            seq_lengths_list.append(int(imgs.shape[0]))

        seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.long)
        return images_list, seq_lengths

    @staticmethod
    def _build_frame_index(seq_lengths: torch.Tensor) -> torch.Tensor:
        # frame_index rows: [episode_idx, t]
        pairs = []
        for ep_idx, T in enumerate(seq_lengths.tolist()):
            pairs.append(torch.stack([torch.full((T,), ep_idx, dtype=torch.long),
                                      torch.arange(T, dtype=torch.long)], dim=1))
        return torch.cat(pairs, dim=0)  # (F,2)

    def __len__(self):
        return int(self.frame_index.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        ep_idx, t = self.frame_index[idx].tolist()
        img_u8 = self.images_list[ep_idx][t]          # (C,H,W) uint8
        img = img_u8.float() / 255.0                 # (C,H,W) float in [0,1]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

def load_egoverse_images_train_val(
    transform: Optional[Callable],
    data_path: str,
    valid_ratio: float = 0.1,
    filter: Optional[str] = None,
    mode: str = "train",
    pixel_key_idx: int = 0,
):
    """
    data_path points to a directory that contains multiple processed datasets.
    We will build train and valid datasets from each subfolder and merge them.

    Returns:
      datasets: {"train": merged_train, "valid": merged_valid}
    """
    if filter is None:
        processed_paths = sorted(os.listdir(data_path))
    else:
        processed_paths = sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(data_path, filter))
        )
    cprint(f"Found {len(processed_paths)} processed datasets under {data_path}", "yellow", attrs=["bold"])

    train_dsets = []
    val_dsets = []

    for p in processed_paths:
        p_full = os.path.join(data_path, p)
        cprint(f"Loading image-only dataset from {p}", "yellow")

        if mode == "train":
            train_dsets.append(
                EgoVerseImageOnlyDataset(
                    data_path=p_full,
                    transform=transform,
                    valid_ratio=valid_ratio,
                    mode="train",
                    pixel_key_idx=pixel_key_idx,
                )
            )

        val_dsets.append(
            EgoVerseImageOnlyDataset(
                data_path=p_full,
                transform=transform,
                valid_ratio=valid_ratio,
                mode="valid",
                pixel_key_idx=pixel_key_idx,
            )
        )

    # merge helper: TrajMerge assumes TrajDataset style; but our dataset returns image only.
    # If TrajMerge requires specific fields, you may instead use torch.utils.data.ConcatDataset.
    #
    # We'll use ConcatDataset to be safe.
    from torch.utils.data import ConcatDataset

    datasets = {}
    if mode == "train":
        datasets["train"] = ConcatDataset(train_dsets) if len(train_dsets) > 1 else train_dsets[0]
    else:
        datasets["train"] = []

    datasets["valid"] = ConcatDataset(val_dsets) if len(val_dsets) > 1 else val_dsets[0]
    return datasets



################################
# Main Test
################################
@hydra.main(config_path="./", config_name="egoverse_img")
def test_egoverse_image_dataset(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))
    datasets = hydra.utils.call(
        cfg.dataset,
    )
    
    train_dataset = datasets['train']
    data = train_dataset[0]
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    # data = next(iter(dataloader))
    
    cprint("--------Finished Everything--------", "green", attrs=["bold"])


if __name__ == "__main__":
    test_egoverse_image_dataset()   