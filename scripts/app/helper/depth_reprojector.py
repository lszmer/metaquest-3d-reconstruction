from typing import Iterator
import numpy as np
from tqdm import tqdm


class DepthReprojector:
    @classmethod
    def reproject_depth(
        cls,
        depth_maps: np.ndarray,               # (N, H_d, W_d)
        depth_intrinsics: np.ndarray,         # (N, 3, 3)
        depth_extrinsics: np.ndarray,         # (N, 4, 4)
        rgb_intrinsics: np.ndarray,           # (N, 3, 3)
        rgb_extrinsics: np.ndarray,           # (N, 4, 4)
        out_sizes: tuple,                     # (N, 2)
        batch_size: int = 1,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        import torch
        from torch.utils.data import Dataset, DataLoader

        class ReprojectionDataset(Dataset):
            def __init__(self, depth_maps, depth_intrinsics, depth_extrinsics, rgb_intrinsics, rgb_extrinsics, out_sizes):
                self.depth_maps = depth_maps
                self.depth_intrinsics = depth_intrinsics
                self.depth_extrinsics = depth_extrinsics
                self.rgb_intrinsics = rgb_intrinsics
                self.rgb_extrinsics = rgb_extrinsics
                self.out_sizes = out_sizes

            def __len__(self):
                return len(self.depth_maps)

            def __getitem__(self, idx):
                return (
                    self.depth_maps[idx],
                    self.depth_intrinsics[idx],
                    self.depth_extrinsics[idx],
                    self.rgb_intrinsics[idx],
                    self.rgb_extrinsics[idx],
                    self.out_sizes[idx],
                    idx
                )

        dataset = ReprojectionDataset(
            depth_maps, depth_intrinsics, depth_extrinsics,
            rgb_intrinsics, rgb_extrinsics, out_sizes
        )

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        with torch.no_grad():
            for depth_map_b, depth_intrinsics_b, depth_extrinsics_b, rgb_intrinsics_b, rgb_extrinsics_b, out_size_b, indices in tqdm(loader, desc="Reproject Depth Maps"):
                device = "cuda" if torch.cuda.is_available() else "cpu"

                depth_map_b = depth_map_b.unsqueeze(1).to(device).float()
                depth_intrinsics_b = depth_intrinsics_b.to(device).float()
                depth_extrinsics_b = depth_extrinsics_b.to(device).float()
                rgb_intrinsics_b = rgb_intrinsics_b.to(device).float()
                rgb_extrinsics_b = rgb_extrinsics_b.to(device).float()

                B, _, H_d, W_d = depth_map_b.shape

                H_rgb_b = out_size_b[:, 0]
                W_rgb_b = out_size_b[:, 1]

                H_rgb = int(H_rgb_b.max().item())
                W_rgb = int(W_rgb_b.max().item())

                yy, xx = torch.meshgrid(
                    torch.arange(H_rgb, device=device),
                    torch.arange(W_rgb, device=device),
                    indexing='ij'
                )
                ones = torch.ones_like(xx)
                pix_rgb = torch.stack([xx, yy, ones], dim=0).float().reshape(3, -1)  # (3, N)

                K_rgb_inv = torch.inverse(rgb_intrinsics_b)  # (B, 3, 3)
                rays_rgb = torch.bmm(K_rgb_inv, pix_rgb.expand(B, -1, -1))  # (B, 3, N)
                ones_hom = torch.ones((B, 1, rays_rgb.shape[-1]), device=device)
                rays_rgb_hom = torch.cat([rays_rgb, ones_hom], dim=1)  # (B, 4, N)

                T = torch.inverse(depth_extrinsics_b) @ rgb_extrinsics_b  # (B, 4, 4)
                rays_in_depth = torch.bmm(T, rays_rgb_hom)  # (B, 4, N)
                rays_in_depth = rays_in_depth[:, :3, :]  # (B, 3, N)

                proj = torch.bmm(depth_intrinsics_b, rays_in_depth)  # (B, 3, N)
                z = rays_in_depth[:, 2, :].clamp(min=1e-6)
                u = proj[:, 0, :] / z
                v = proj[:, 1, :] / z

                u_norm = (u / (W_d - 1)) * 2 - 1
                v_norm = (v / (H_d - 1)) * 2 - 1
                grid = torch.stack([u_norm, v_norm], dim=-1)  # (B, N, 2)
                grid = grid.view(B, H_rgb, W_rgb, 2)

                depth_reprojected = torch.nn.functional.grid_sample(
                    depth_map_b, grid, mode='bilinear', align_corners=True
                )  # (B, 1, H_rgb, W_rgb)

                result = depth_reprojected[:, 0].cpu().numpy()  # (B, H_rgb, W_rgb)

                yield indices.numpy(), result