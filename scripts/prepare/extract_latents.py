import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from src.brlp import init_autoencoder
from src.brlp import const


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=True)
    parser.add_argument('--aekl_ckpt',   type=str, required=True)
    args = parser.parse_args()

    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()

    transforms_fn = transforms.Compose([
        # DebugLoadImaged(keys=['image_path']),
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.ClipIntensityPercentilesD(keys=['image'], lower=1, upper=99, sharpness_factor=10.),
        # transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        # transforms.ResizeWithPadOrCropD(spatial_size=(192, 512, 512), mode='constant', keys=['image']),
        transforms.ResizeD(spatial_size=const.INPUT_SHAPE_AE, mode='trilinear', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])
    
    df = pd.read_csv(args.dataset_csv)
    # df = df[~df['image_path'].str.contains("OMEGA04/L/V02")]

    with torch.no_grad():
        for image_path in tqdm(df.image_path, total=len(df)):
            destpath = image_path.replace('.nii.gz', '_latent.npz').replace('.nii', '_latent.npz')            
            if os.path.exists(destpath): continue
            oct_tensor = transforms_fn({'image_path': image_path})['image'].to(DEVICE)
            oct_latent, _ = autoencoder.encode(oct_tensor.unsqueeze(0))
            oct_latent = oct_latent.cpu().squeeze(0).numpy()
            np.savez_compressed(destpath, data=oct_latent)