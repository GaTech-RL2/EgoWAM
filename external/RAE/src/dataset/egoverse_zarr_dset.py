import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, List
from torch.utils.data import Dataset

from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver, MultiDataset
from egomimic.rldb.embodiment.eva import Eva, _build_eva_bimanual_transform_list


class EgoVerseImageOnlyZarrDataset(Dataset):
    """
    Lightweight EgoVerse Zarr dataset wrapper that returns image only.
    Uses direct indexing without extracting all images upfront.
    
    Args:
        folder_path: Path to zarr dataset root
        key_map: Key mapping for episode resolver
        transform_list: Transform list for episode resolver
        mode: "train" or "valid"
        filters: Optional dict of filters for episodes
        pixel_key: Which image key to extract (e.g., "observations.images.front_img_1")
        transform: Optional image transform
    """
    
    def __init__(
        self,
        folder_path: str,
        key_map: Dict = Eva.get_keymap(action_horizon=45, use_future=True),
        transform_list: List = _build_eva_bimanual_transform_list(
            chunk_length=100, use_future=True
        ),
        mode: str = "train",
        filters: Optional[Dict] = None,
        pixel_key: Optional[str] = None,
        future_pixel_key: Optional[str] = None,
        return_future: bool = False,
        transform: Optional[Callable] = None,
        valid_ratio: float = 0.2,
    ):
        self.folder_path = folder_path
        self.key_map = key_map
        self.transform_list = transform_list
        self.mode = mode
        self.filters = filters or {}
        self.pixel_key = pixel_key
        self.future_pixel_key = future_pixel_key
        self.return_future = return_future
        self.transform = transform
        self.valid_ratio = valid_ratio
        resolver = S3EpisodeResolver(
            folder_path=folder_path,
            key_map=key_map,
            transform_list=transform_list,
        )
        
        self.dataset = MultiDataset._from_resolver(
            resolver,
            filters=self.filters,
            mode=self.mode,
            valid_ratio=valid_ratio
        )
        
    def __len__(self):
        return len(self.dataset)

    def _process_image_tensor(self, img: torch.Tensor) -> torch.Tensor:
        # Handle temporal dimension when sequence tensors are returned.
        if img.dim() == 4:
            img = img[0]

        img = img.float()
        if img.max() > 1.5:
            img = img / 255.0
        img = img.clamp(0, 1)

        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        if self.pixel_key is None:
            raise ValueError("pixel_key must be provided.")

        img = self._process_image_tensor(sample[self.pixel_key])

        if not self.return_future:
            return img

        if self.future_pixel_key is None:
            raise ValueError("future_pixel_key must be provided when return_future=True.")
        future_img = self._process_image_tensor(sample[self.future_pixel_key])
        return img, future_img



def load_egoverse_zarr_images_train_val(
    transform: Optional[Callable],
    data_path: str,
    valid_ratio: float = 0.1,
    filter: Optional[str] = None,
    mode: str = "train",
    pixel_key_idx: int = 0,
    return_future: bool = False,
    pixel_key: str = "observations.images.front_img_1",
    future_pixel_key: str = "future.observations.images.front_img_1",
):
    """
    data_path points to a directory that contains multiple processed datasets.
    We will build train and valid datasets from each subfolder and merge them.

    Returns:
      datasets: {"train": merged_train, "valid": merged_valid}
    """
    
    datasets = {}
    if mode == "train":
        datasets["train"] = EgoVerseImageOnlyZarrDataset(
            folder_path=data_path,
            transform=transform,
            mode="train",
            pixel_key=pixel_key,
            future_pixel_key=future_pixel_key,
            return_future=return_future,
            valid_ratio=valid_ratio,
            filters=filter
        )
    else:
        datasets["train"] = []

    datasets["valid"] = EgoVerseImageOnlyZarrDataset(
        folder_path=data_path,
        transform=transform,
        mode="valid",
        pixel_key=pixel_key,
        future_pixel_key=future_pixel_key,
        return_future=return_future,
        valid_ratio=valid_ratio,
        filters=filter

    )
    return datasets


if __name__ == "__main__":
    # Example usage
    data_path = "/coc/flash7/scratch/egowm/egoverseS3ZarrDataset/eva_cup_saucer/"
    transform = None  # Replace with actual transform if needed
    datasets = load_egoverse_zarr_images_train_val(
        transform=transform,
        data_path=data_path,
        valid_ratio=0.1,
        filter={"task": "cup_on_saucer_success", "robot_name": "eva_bimanual"},
        mode="train",
        pixel_key_idx=0,
        return_future=True,
        pixel_key="observations.images.front_img_1",
        future_pixel_key="future.observations.images.front_img_1",
    )
    print(f"Train dataset length: {len(datasets['train'])}")
    print(f"Valid dataset length: {len(datasets['valid'])}")