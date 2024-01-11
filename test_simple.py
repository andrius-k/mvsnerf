import sys
sys.path.append('.')
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data.dtu import MVSDatasetDTU
from models import MVSNet
from utils import sub_selete_data, homo_warp
import cv2
import numpy as np


def decode_batch(batch, idx=list(torch.arange(4))):
        device = "cpu"
        data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
        pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                    'c2ws': data_mvs['c2ws'].squeeze(),'near_fars':data_mvs['near_fars'].squeeze()}

        return data_mvs, pose_ref


def get_depth_values(near_far, imgs, lindisp=False):
    D = 128
    t_vals = torch.linspace(0., 1., steps=D, device=imgs.device, dtype=imgs.dtype)  # (B, D)
    near, far = near_far  # assume batch size==1
    if not lindisp:
        depth_values = near * (1.-t_vals) + far * (t_vals)
    else:
        depth_values = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    depth_values = depth_values.unsqueeze(0)
    return depth_values


def verify(src_img, proj_mat, near_far, depths_h):
    # src_img: torch.Size([1, 3, 128, 160])
    # proj_mat: torch.Size([1, 3, 4])
    # depth_values: torch.Size([1, 128])
    # pad: 0

    idx = 0
    src_img = src_img[:, idx, ...] # [1, 3, 512, 640]
    proj_mat = proj_mat[:, idx, ...] # [1, 3, 4]
    depths_h = depths_h[:, 0, ...] # [1, 512, 640] reference view

    # Params used for rendering an image
    device = src_img.device
    shape=(1,1,3,1,1)
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

    depth_values = get_depth_values(near_far, src_img)
    # homo_warp comes straight from MVSNeRF
    warped_volume, src_grid = homo_warp(src_img, proj_mat, depth_values, pad=0) # [1, 3, 128, 512, 640]

    # Save warped images at a few depth levels.
    # All 3 saved images are identical to the input because proj_mat is identity.
    for d in [0, 64, 127]:
        img = (warped_volume[:, :, d, :, :] - mean) / std
        img = (img[0, 0, ...].permute(1, 2, 0).detach().numpy() * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"depth_{d}.jpg", img)

    # ======== Sample the cost volume to recreate the reference image ========== #

    H = warped_volume.shape[3]
    W = warped_volume.shape[4]

    # Create a 3D grid. Use evenly spaced x,y values and pick a single (GT) depth for z value.
    near, far = near_far[0], near_far[1]
    d0, y0, x0 = torch.meshgrid(
        torch.arange(1).to(warped_volume.device).float(), 
        torch.arange(H).to(warped_volume.device).float(), 
        torch.arange(W).to(warped_volume.device).float())
    x_norm = 2 * (x0 / (W - 1)) - 1
    y_norm = 2 * (y0 / (H - 1)) - 1

    d_norm = (depths_h - near) / (far - near)
    d_norm = (2 * d_norm) - 1

    sample_grid = torch.stack([x_norm, y_norm, d_norm], dim=-1).unsqueeze(dim=0) # (B, D, h, w, 3)
    sampled = F.grid_sample(warped_volume, sample_grid, mode="nearest", padding_mode="border", align_corners=False)
    # sampled: (B, C, 1, h, w)

    # Save the sampled image. Output is recognizable but contains lots of artifacts
    img = (sampled[:, :, 0, :, :] - mean) / std
    img = (img[0, 0, ...].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("sampled.jpg", img)

    # Render depth map to an image for debugging. Matches the depth image provided in the dataset.
    depth = depths_h.squeeze()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth.numpy() * 255.).astype(np.uint8)
    cv2.imwrite(f"depth_map.jpg", depth)

    # Render ref view into an image for debugging.
    img = (src_img - mean) / std
    img = (img[0, 0, ...].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("gt.jpg", img)


def run():
    root_dir="/cluster/work/cvl/haofxu/dynamic_nerf/mvsnerf/dataset/dtu"
    dataset = MVSDatasetDTU(root_dir, split="val", max_len=1, downSample=1.0)
    loader = DataLoader(dataset,
                          shuffle=False,
                          num_workers=1,
                          batch_size=1,
                          pin_memory=True)

    for i, data in enumerate(loader):
        if 'scan' in data.keys():
            data.pop('scan')
        data_mvs, pose_ref = decode_batch(data)

        imgs, proj_mats = data_mvs['images'], data_mvs['proj_mats']
        near_fars, depths_h = data_mvs['near_fars'], data_mvs['depths_h']

        verify(imgs[:, :4], proj_mats[:, :4], near_fars[0,0], depths_h[:, :4])


if __name__ == "__main__":
    run()