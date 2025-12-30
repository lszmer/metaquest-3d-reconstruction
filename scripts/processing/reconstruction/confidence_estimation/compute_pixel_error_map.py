import numpy as np


def bilinear_interpolate_depth(depth_map: np.ndarray, u: np.ndarray, v: np.ndarray, depth_max: float) -> np.ndarray:
    h, w = depth_map.shape
    
    # Filter out NaN and Inf values before casting to int
    # Also filter out values that are way out of bounds (to prevent int32 overflow)
    # Use very conservative bounds: pixel coordinates should never exceed image dimensions by much
    # Add generous margin (10x image size) to account for projection errors, but prevent overflow
    max_coord = max(w, h) * 10
    valid_coords = (
        np.isfinite(u) & np.isfinite(v) &
        (u >= -max_coord) & (u < max_coord) &
        (v >= -max_coord) & (v < max_coord)
    )
    if not valid_coords.any():
        return np.zeros_like(u, dtype=np.float32)
    
    # Only process valid coordinates
    u_valid = u[valid_coords]
    v_valid = v[valid_coords]
    
    # Clip values to reasonable bounds before floor to prevent int32 overflow
    # Use a safe range that's well within int32 limits but covers all valid pixel coordinates
    max_safe_int32 = 2**30  # 1 billion, well below int32 max
    u_valid = np.clip(u_valid, -max_safe_int32, max_safe_int32)
    v_valid = np.clip(v_valid, -max_safe_int32, max_safe_int32)
    
    # Suppress warnings during floor and cast since we've already validated bounds
    with np.errstate(invalid='ignore', over='ignore'):
        u_floor = np.floor(u_valid)
        v_floor = np.floor(v_valid)
        
        # Clip again after floor to ensure safe casting
        u_floor = np.clip(u_floor, -max_safe_int32, max_safe_int32)
        v_floor = np.clip(v_floor, -max_safe_int32, max_safe_int32)
        
        u0 = u_floor.astype(np.int32)
        v0 = v_floor.astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    # Bounds check: ensure indices are within valid array range
    # We need u0 >= 0, u1 < w, v0 >= 0, v1 < h for valid bilinear interpolation
    valid = (
        (u0 >= 0) & (u1 < w) &
        (v0 >= 0) & (v1 < h) &
        (u0 < w) & (v0 < h)  # Additional safety check
    )

    z = np.zeros_like(u, dtype=np.float32)
    
    if not valid.any():
        return z

    u0v = u0[valid]
    u1v = u1[valid]
    v0v = v0[valid]
    v1v = v1[valid]
    uv = u_valid[valid]
    vv = v_valid[valid]

    # Final safety check: ensure indices are within bounds (defensive programming)
    # This should not be necessary given the valid check above, but prevents crashes
    u0v = np.clip(u0v, 0, w - 1).astype(np.int32)
    u1v = np.clip(u1v, 0, w - 1).astype(np.int32)
    v0v = np.clip(v0v, 0, h - 1).astype(np.int32)
    v1v = np.clip(v1v, 0, h - 1).astype(np.int32)

    Ia = depth_map[v0v, u0v]
    Ib = depth_map[v0v, u1v]
    Ic = depth_map[v1v, u0v]
    Id = depth_map[v1v, u1v]

    is_valid_interp = (
        (Ib > 0) & (Ib <= depth_max) &
        (Ia > 0) & (Ia <= depth_max) &
        (Ic > 0) & (Ic <= depth_max) &
        (Id > 0) & (Id <= depth_max)
    )

    wa = (u1v - uv) * (v1v - vv)
    wb = (uv - u0v) * (v1v - vv)
    wc = (u1v - uv) * (vv - v0v)
    wd = (uv - u0v) * (vv - v0v)

    z_interp = wa * Ia + wb * Ib + wc * Ic + wd * Id
    z_valid_indices = np.where(valid_coords)[0][valid]
    z[z_valid_indices] = np.where(is_valid_interp, z_interp, 0.0)

    return z


def depth_to_pointcloud_numpy(
    depth_map: np.ndarray,   # (h, w)
    intrinsics: np.ndarray,  # (3, 3)
    extrinsics: np.ndarray,  # (4, 4)
    depth_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    valid_mask = (depth_map > 0) & (depth_map <= depth_max)
    v_coords, u_coords = np.where(valid_mask)

    z = depth_map[v_coords, u_coords]
    x = (u_coords - cx) * z / fx
    y = (v_coords - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=1)

    ones = np.ones((points_cam.shape[0], 1))
    points_cam_h = np.concatenate([points_cam, ones], axis=1)
    points_world = (extrinsics @ points_cam_h.T).T[:, :3]

    return points_world, u_coords, v_coords


def compute_pixel_error_map(
    intrinsic_matrices: np.ndarray,
    extrinsic_matrices: np.ndarray,
    extrinsic_matrices_inv: np.ndarray,
    ref_frame_idx: int,
    ref_depth_map: np.ndarray,
    target_frame_idx: int,
    target_depth_map: np.ndarray,
    depth_max: float = 3.0
) -> np.ndarray:
    h, w = ref_depth_map.shape

    # Step 1: target depth -> world point cloud
    ref_points_world, ref_u, ref_v = depth_to_pointcloud_numpy(
        depth_map=ref_depth_map,
        intrinsics=intrinsic_matrices[ref_frame_idx],
        extrinsics=extrinsic_matrices[ref_frame_idx],
        depth_max=depth_max
    )

    # Step 2: world point cloud -> target camera space
    source_extr_inv = extrinsic_matrices_inv[target_frame_idx]
    points_h = np.hstack([ref_points_world, np.ones((ref_points_world.shape[0], 1))])
    target_points = (source_extr_inv @ points_h.T).T[:, :3]

    x, y, z = target_points[:, 0], target_points[:, 1], target_points[:, 2]
    fx, fy = intrinsic_matrices[target_frame_idx][0, 0], intrinsic_matrices[target_frame_idx][1, 1]
    cx, cy = intrinsic_matrices[target_frame_idx][0, 2], intrinsic_matrices[target_frame_idx][1, 2]

    # Step 3: Project points to target depth map pixel coordinates
    valid_project_mask = (z > 0) & np.isfinite(z) & (z <= depth_max) & np.isfinite(x) & np.isfinite(y)
    if not valid_project_mask.any():
        # No valid points to project, return empty confidence map
        return np.full_like(ref_depth_map, fill_value=np.nan, dtype=np.float32)
    
    # Filter to valid points
    x_valid = x[valid_project_mask]
    y_valid = y[valid_project_mask]
    z_valid = z[valid_project_mask]
    
    # Get original indices for tracking
    original_indices = np.where(valid_project_mask)[0]

    u = (x_valid * fx / z_valid) + cx
    v = (y_valid * fy / z_valid) + cy
    
    # Additional validation: ensure u and v are finite
    valid_uv = np.isfinite(u) & np.isfinite(v)
    if not valid_uv.any():
        return np.full_like(ref_depth_map, fill_value=np.nan, dtype=np.float32)
    
    # Further filter to valid UV coordinates
    u_valid = u[valid_uv]
    v_valid = v[valid_uv]
    z_valid_uv = z_valid[valid_uv]
    original_indices_uv = original_indices[valid_uv]

    # Step 4: Bilinear interpolation to get target depth values
    z_target = bilinear_interpolate_depth(
        depth_map=target_depth_map,
        u=u_valid,
        v=v_valid,
        depth_max=depth_max
    )
    valid_target_mask = (z_target > 0) & np.isfinite(z_target)
    if not valid_target_mask.any():
        return np.full_like(ref_depth_map, fill_value=np.nan, dtype=np.float32)

    # Step 5: Compute pixel errors
    u_final = u_valid[valid_target_mask]
    v_final = v_valid[valid_target_mask]
    z_target_final = z_target[valid_target_mask]
    
    x_valid_target = (u_final - cx) * z_target_final / fx
    y_valid_target = (v_final - cy) * z_target_final / fy
    z_valid_target = z_target_final
    target_points = np.stack([x_valid_target, y_valid_target, z_valid_target], axis=1)

    ones = np.ones((target_points.shape[0], 1))
    target_points_h = np.concatenate([target_points, ones], axis=1)
    target_points_world = (extrinsic_matrices[target_frame_idx] @ target_points_h.T).T[:, :3]

    # Get corresponding reference points
    final_original_indices = original_indices_uv[valid_target_mask]
    ref_points_valid_world = ref_points_world[final_original_indices]
    dist = np.linalg.norm(ref_points_valid_world - target_points_world, axis=1)

    # Step 6: Create confidence map
    confidence_map = np.full_like(ref_depth_map, fill_value=np.nan, dtype=np.float32)

    u_ref = ref_u[final_original_indices]
    v_ref = ref_v[final_original_indices]
    inside_mask = (u_ref >= 0) & (u_ref < w) & (v_ref >= 0) & (v_ref < h)

    u_ref_inside = u_ref[inside_mask]
    v_ref_inside = v_ref[inside_mask]
    dist_inside = dist[inside_mask]

    confidence_map[v_ref_inside, u_ref_inside] = dist_inside

    return confidence_map