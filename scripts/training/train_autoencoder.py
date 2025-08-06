import os
import argparse
import warnings
from setproctitle import setproctitle
import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from monai.utils.misc import set_determinism
import shutil
import datetime
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter
from monai.metrics.regression import PSNRMetric, SSIMMetric, MSEMetric
from src.brlp.lakefsimaged import LoadLakeFSImaged
from src.brlp.lakefsloader import LakeFSLoader
from src.brlp.datalist import DataList
import yaml
import logging
from src.brlp.training_functions import get_loader
from pathlib import Path
from src.brlp import const
from src.brlp import utils
from src.brlp import (
    KLDivergenceLoss, GradientAccumulation,
    init_autoencoder, init_patch_discriminator,
    get_dataset_from_pd  
)


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = GradScaler(DEVICE)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',    required=True, type=str)
    parser.add_argument('--cache_dir',      default=None, type=str)
    parser.add_argument('--output_dir',     required=True, type=str)
    parser.add_argument('--aekl_ckpt',      default=None,  type=str)
    parser.add_argument('--disc_ckpt',      default=None,  type=str)
    parser.add_argument('--num_workers',    default=8,     type=int)
    parser.add_argument('--n_epochs',       default=5,     type=int)
    parser.add_argument('--max_batch_size', default=2,     type=int)
    parser.add_argument('--batch_size',     default=16,    type=int)
    parser.add_argument('--lr',             default=1e-4,  type=float)
    parser.add_argument('--aug_p',          default=0.8,   type=float)
    parser.add_argument('-C', '--config', default="configs/config.yaml", help="Sets the config file to be used.")
    parser.add_argument('--lakefs', default="configs/lakefs_cfg.yaml", help="Sets the lakefs config file to be used.")
    args = parser.parse_args()

    # set process name
    setproctitle("AE-BrLP")

    src_path = Path(__file__).parent
    project_path = src_path.parent

    with open(src_path / args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(src_path / args.lakefs) as f:
        lakefs_config = yaml.load(f, Loader=yaml.FullLoader)
    learn_cfg = config["learning"]
    model_cfg = config["model"]

    lakefs_loader = LakeFSLoader(
        local_cache_path=lakefs_config["cache_path"],
        repo_name=lakefs_config["data_repository"],
        branch_id=lakefs_config["branch"],
        ca_path=lakefs_config["ca_path"],
        endpoint=lakefs_config["s3_endpoint"],
        secret_key=lakefs_config["secret_key"],
        access_key=lakefs_config["access_key"],
    )

    transforms_fn = transforms.Compose([
        # transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        LoadLakeFSImaged(lakefs_loader=lakefs_loader, image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.ClipIntensityPercentilesD(keys=['image'], lower=1, upper=99, sharpness_factor=10.),
        # transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        # transforms.ResizeWithPadOrCropD(spatial_size=(192, 512, 512), mode='constant', keys=['image']),
        # transforms.SpatialPadD(spatial_size=[193, 512, 512], method='symmetric', mode='constant', keys=['image']),
        transforms.ResizeD(spatial_size=const.INPUT_SHAPE_AE, mode='trilinear', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

    # transforms_fn = transforms.Compose([
    # transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
    # transforms.RandFlip(spatial_axis=0, prob=0.5),
    # transforms.RandFlip(spatial_axis=1, prob=0.5),
    # transforms.RandFlip(spatial_axis=2, prob=0.5),
    # transforms.RandAffine(
    #    prob=0.5,
    #    rotate_range=(0.1, 0.1, 0.1),  # radians
    #    shear_range=(0.05, 0.05, 0.05),
    #    translate_range=(10, 10, 5),
    #    scale_range=(0.1, 0.1, 0.1),
    #    mode='bilinear'
    #),
    # transforms.ResizeWithPadOrCrop(spatial_size=(128, 128, 128), mode='minimum'),
    # transforms.ScaleIntensity(minv=0, maxv=1),
    # ])

    torch.multiprocessing.set_sharing_strategy('file_system') # against memory leak when using more than one worker, see https://github.com/pytorch/pytorch/issues/973

    # dataset_df = pd.read_csv(args.dataset_csv)
    # train_df = dataset_df[ dataset_df.split == 'train' ]
    # train_df = train_df[~train_df['image_path'].str.contains("OMEGA04/L/V02")]
    # trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)

    # create the experiment folder, setup tensorboard, copy the used config
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = config['run_name']
    use_cfg_fold = True
    use_fold = config["data"]["fold"]
    if not use_cfg_fold: experiment_name = experiment_name + f"_fold{use_fold}"
    experiment_path = project_path / 'experiments' / (experiment_name + f"_{timestamp}")
    writer = SummaryWriter(experiment_path)
    shutil.copyfile(src_path / args.config, experiment_path / "config.yaml")
    shutil.copyfile(src_path / args.lakefs, experiment_path / "lakefs_cfg.yaml")

    # create a datalist from the filepath
    input_path = Path(config["input_path"])
    if input_path.is_file() and '.json' in input_path.name.lower():
        datalist  = DataList.from_json(filepath=input_path, lakefs_config=lakefs_config)
    else:
        datalist = DataList.from_lakefs(data_config=config["data"], lakefs_config=lakefs_config, filepath=input_path, include_root=True)
    data_list = datalist.data

    # save datalist
    datalist.save_datalist_to_json(path=experiment_path / "datalist.json")
     # Configure the logging system
    log_handlers = [logging.StreamHandler()]
    if args.log:
        log_handlers.append(logging.FileHandler(experiment_path / "debug.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=log_handlers,
    )

    print(f"training dataset_size is {len(data_list['training'])}")

    train_loader, val_loader = get_loader(
        batch_size=learn_cfg["batch_size"],
        data_list=data_list,
        fold=use_fold,
        num_workers=config["data"]["num_workers"],
        train_transform=transforms_fn,
        val_transform=transforms_fn,
    )

    # train_loader = DataLoader(dataset=trainset, 
    #                          num_workers=config["data"]["num_workers"], 
    #                          batch_size=learn_cfg["batch_size"], 
    #                          shuffle=True, 
    #                          persistent_workers=True, 
    #                          pin_memory=True)


    autoencoder   = init_autoencoder(args.aekl_ckpt).to(DEVICE)
    discriminator = init_patch_discriminator(args.disc_ckpt).to(DEVICE)

    adv_weight          = 0.025
    perceptual_weight   = 0.001
    kl_weight           = 1e-7

    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perc_loss_fn = PerceptualLoss(spatial_dims=3, 
                                      network_type="squeeze", 
                                      is_fake_3d=True, 
                                      fake_3d_ratio=0.2).to(DEVICE)
    
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)


    gradacc_g = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_g, 
                                     grad_scaler=scaler)

    gradacc_d = GradientAccumulation(actual_batch_size=args.max_batch_size,
                                     expect_batch_size=args.batch_size,
                                     loader_len=len(train_loader),
                                     optimizer=optimizer_d, 
                                     grad_scaler=scaler)

    avgloss = utils.AverageLoss()
    # writer = SummaryWriter()
    total_counter = 0

    PSNR = PSNRMetric(max_val=1., reduction='mean')
    # SSIM = StructuralSimilarityIndexMeasure().to(device)
    SSIM = SSIMMetric(spatial_dims=3, data_range=1., reduction='mean')
    MSE = MSEMetric(reduction='mean')

    for epoch in range(args.n_epochs):
        
        autoencoder.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f'Epoch {epoch}')
        mse_batches, psnr_batches, ssim_batches = 0, 0, 0

        for step, batch in progress_bar:

            with autocast('cuda', enabled=True):

                images = batch["image"].to(DEVICE)
                # images = batch.to(DEVICE)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # we use [-1] here because the discriminator also returns 
                # intermediate outputs and we want only the final one.
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                # Computing the loss for the generator. In the Adverarial loss, 
                # if the discriminator works well then the logits are close to 0.
                # Since we use `target_is_real=True`, then the target tensor used
                # for the MSE is a tensor of 1, and minizing this loss will make 
                # the generator better at fooling the discriminator (the discriminator
                # weights are not optimized here).

                rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                per_loss = perceptual_weight * perc_loss_fn(reconstruction.float(), images.float())
                gen_loss = adv_weight * adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                
                loss_g = rec_loss + kld_loss + per_loss + gen_loss
                
            gradacc_g.step(loss_g, step)

            with autocast('cuda', enabled=True):

                # Here we compute the loss for the discriminator. Keep in mind that
                # the loss used is an MSE between the output logits and the expected logits.
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                d_loss_fake = adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_loss_real = adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            gradacc_d.step(loss_d, step)

            # Logging.
            avgloss.put('Generator/reconstruction_loss',    rec_loss.item())
            avgloss.put('Generator/perceptual_loss',        per_loss.item())
            avgloss.put('Generator/adverarial_loss',        gen_loss.item())
            avgloss.put('Generator/kl_regularization',      kld_loss.item())
            avgloss.put('Discriminator/adverarial_loss',    loss_d.item())
            
            if total_counter % len(train_loader) == 0:
                step = total_counter // len(train_loader)
                utils.tb_display_reconstruction(writer, step, images[0].detach().cpu(), reconstruction[0].detach().cpu())  

            total_counter += 1
            avgloss.to_tensorboard(writer, epoch+1)
        
        # Validation phase
        autoencoder.eval()
        progress_bar_eval = tqdm(enumerate(val_loader), total=len(val_loader))
        progress_bar_eval.set_description(f'Validation Epoch {epoch}')
        for step, batch in progress_bar_eval:
            with torch.no_grad():
                images = batch["image"].to(DEVICE)
                reconstruction, _, _ = autoencoder(images)
                # Compute metrics between reconstruction and original image
                mse_batch = MSE(reconstruction.float(), images.float())
                # mse_batches += mse_batch.numpy()
                psnr_batch = PSNR(reconstruction.float(), images.float())
                # psnr_batches += psnr_batch.numpy()
                ssim_batch = SSIM(reconstruction.float(), images.float())
                # ssim_batches += ssim_batch.numpy()
                avgloss.put('Validation_metrics/MSE',             mse_batch.item())
                avgloss.put('Validation_metrics/PSNR',            psnr_batch.item())
                avgloss.put('Validation_metrics/SSIM',            ssim_batch.item())
                avgloss.to_tensorboard(writer, epoch+1)

        # Save the model each 50 epoch.
        if (epoch+1) % 50 == 0 or epoch == args.n_epochs - 1:
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator-ep-{epoch}.pth'))
            torch.save(autoencoder.state_dict(),   os.path.join(args.output_dir, f'autoencoder-ep-{epoch}.pth'))