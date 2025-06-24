import os
from typing import Optional, Union
import nibabel as nib
import torch
import pandas as pd
from monai.data.dataset import Dataset, PersistentDataset
from monai.transforms.transform import Transform
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

class OCT3DDataset(TorchDataset):
    def __init__(self, data_frame, transform=None):
        self.df = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image_path']
        volume = nib.load(image_path).get_fdata()  # Shape: [512, 496, 193]
        volume = torch.tensor(volume, dtype=torch.float32)  # Convert to torch tensor

        # Optionally permute to match expected shape [C, D, H, W]
        volume = volume.unsqueeze(0)  # Add channel dimension: [1, 512, 496, 193]
        volume = volume.permute(0, 2, 3, 1) # Change to [1, 193, 512, 496]

        if self.transform:
            volume = self.transform(volume)

        return volume


def get_dataset_from_pd(df: pd.DataFrame, transforms_fn: Transform, cache_dir: Optional[str]) -> Union[Dataset,PersistentDataset]: # -> OCT3DDataset:
    """
    If `cache_dir` is defined, returns a `monai.data.PersistenDataset`. 
    Otherwise, returns a simple `monai.data.Dataset`.

    Args:
        df (pd.DataFrame): Dataframe describing each image in the longitudinal dataset.
        transforms_fn (Transform): Set of transformations
        cache_dir (Optional[str]): Cache directory (ensure enough storage is available)

    Returns:
        Dataset|PersistentDataset: The dataset
    """
    assert cache_dir is None or os.path.exists(cache_dir), 'Invalid cache directory path'
    data = df.to_dict(orient='records')
    return Dataset(data=data, transform=transforms_fn) if cache_dir is None \
        else PersistentDataset(data=data, transform=transforms_fn, cache_dir=cache_dir)
    # return OCT3DDataset(data_frame=df, transform=transforms_fn)