#!/usr/bin/env python3
"""
Compare a reconstructed mesh scan against a ground truth mesh.

This script computes various metrics to evaluate how well a scan matches
the ground truth, including:
- Chamfer Distance
- Hausdorff Distance
- Point-to-Surface Distance
- F-score (precision/recall)
- Volume metrics

It also provides visualization capabilities:
- Error heatmaps
- Side-by-side comparison
- Overlay visualization
- Export to HTML/PLY with color-coded errors
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

# Handle imports for both module and script usage
try:
    from .mesh_loader import load_mesh, mesh_to_point_cloud
except ImportError:
    # If running as a script, add parent directory to path
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir.parent))
    from evaluation.mesh_loader import load_mesh, mesh_to_point_cloud


@dataclass
class ComparisonMetrics:
    """Metrics comparing a scan mesh to a ground truth mesh."""
    # Chamfer Distance (bidirectional average distance)
    chamfer_distance: float
    
    # Hausdorff Distance (maximum distance)
    hausdorff_distance: float
    
    # Point-to-Surface distances
    mean_point_to_surface: float
    median_point_to_surface: float
    std_point_to_surface: float
    max_point_to_surface: float
    
    # F-score metrics (precision/recall)
    precision: float  # Percentage of scan points within threshold of GT
    recall: float  # Percentage of GT points within threshold of scan
    f_score: float  # Harmonic mean of precision and recall
    
    # Volume metrics
    volume_scan: float
    volume_gt: float
    volume_overlap: float
    volume_iou: float  # Intersection over Union
    
    # Scale and alignment metrics
    bounding_box_ratio: Tuple[float, float, float]  # scan/gt ratios
    center_offset: Tuple[float, float, float]  # Translation between centers
    
    # Sampling info
    num_scan_points: int
    num_gt_points: int
    
    # Surface area and hole metrics
    surface_area_scan: float  # Covered area (Fläche) of scan mesh
    surface_area_gt: float  # Covered area of ground truth mesh
    num_holes_scan: int  # Number of holes in scan mesh
    num_holes_gt: int  # Number of holes in ground truth mesh


@dataclass
class ComparisonReport:
    """Complete comparison report."""
    scan_path: Path
    ground_truth_path: Path
    metrics: ComparisonMetrics
    threshold: float  # Distance threshold used for F-score
    warnings: list[str]
    errors: list[str]


@dataclass
class DualComparisonReport:
    """Comparison report for two scans (fog vs no_fog) against ground truth."""
    ground_truth_path: Path
    no_fog_scan_path: Path
    fog_scan_path: Path
    no_fog_report: ComparisonReport
    fog_report: ComparisonReport
    improvement: dict  # Metrics showing improvement (negative = fog is better)


def compute_chamfer_distance(
    scan_pcd: o3d.geometry.PointCloud,
    gt_pcd: o3d.geometry.PointCloud
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Chamfer Distance between two point clouds.
    
    Returns:
        chamfer_distance: Average bidirectional distance
        scan_to_gt_dists: Distances from scan points to GT
        gt_to_scan_dists: Distances from GT points to scan
    """
    print(f"[Debug] Computing Chamfer distance: scan has {len(scan_pcd.points)} points, GT has {len(gt_pcd.points)} points")
    
    # Compute distances from scan to GT
    scan_to_gt_dists = np.asarray(scan_pcd.compute_point_cloud_distance(gt_pcd))
    print(f"[Debug] Scan to GT distances: min={np.min(scan_to_gt_dists):.6f}, max={np.max(scan_to_gt_dists):.6f}, mean={np.mean(scan_to_gt_dists):.6f}, median={np.median(scan_to_gt_dists):.6f}")
    
    # Compute distances from GT to scan
    gt_to_scan_dists = np.asarray(gt_pcd.compute_point_cloud_distance(scan_pcd))
    print(f"[Debug] GT to scan distances: min={np.min(gt_to_scan_dists):.6f}, max={np.max(gt_to_scan_dists):.6f}, mean={np.mean(gt_to_scan_dists):.6f}, median={np.median(gt_to_scan_dists):.6f}")
    
    # Chamfer distance is the average of both directions
    chamfer_dist = np.mean(scan_to_gt_dists) + np.mean(gt_to_scan_dists)
    print(f"[Debug] Chamfer distance: {chamfer_dist:.6f}")
    
    return chamfer_dist, scan_to_gt_dists, gt_to_scan_dists


def compute_hausdorff_distance(
    scan_pcd: o3d.geometry.PointCloud,
    gt_pcd: o3d.geometry.PointCloud
) -> float:
    """
    Compute Hausdorff Distance (maximum distance) between two point clouds.
    """
    scan_to_gt_dists = np.asarray(scan_pcd.compute_point_cloud_distance(gt_pcd))
    gt_to_scan_dists = np.asarray(gt_pcd.compute_point_cloud_distance(scan_pcd))
    
    hausdorff_dist = max(np.max(scan_to_gt_dists), np.max(gt_to_scan_dists))
    
    return hausdorff_dist


def compute_point_to_surface_distance(
    scan_pcd: o3d.geometry.PointCloud,
    gt_mesh: o3d.geometry.TriangleMesh
) -> np.ndarray:
    """
    Compute distance from each scan point to the nearest surface on GT mesh.
    """
    print(f"[Debug] Computing point-to-surface distance: scan has {len(scan_pcd.points)} points, GT mesh has {len(gt_mesh.vertices)} vertices")
    
    # Create KDTree from GT mesh vertices for fast lookup
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = gt_mesh.vertices
    
    # For each scan point, find nearest GT vertex
    scan_points = np.asarray(scan_pcd.points)
    gt_points = np.asarray(gt_pcd.points)
    
    print(f"[Debug] GT point cloud for KDTree: {len(gt_points)} points")
    
    # Use Open3D's KDTree for efficient nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(gt_pcd)
    
    distances = []
    failed_count = 0
    for i, point in enumerate(scan_points):
        result = kdtree.search_knn_vector_3d(point, 1)
        # result is a tuple: (k, indices, distances)
        if len(result) >= 2 and len(result[1]) > 0:
            idx: int = int(result[1][0])
            nearest_gt_point = gt_points[idx]
            dist = float(np.linalg.norm(point - nearest_gt_point))
            distances.append(dist)
        else:
            failed_count += 1
            distances.append(float('inf'))
    
    if failed_count > 0:
        print(f"[Warning] KDTree search failed for {failed_count} points")
    
    distances_array = np.array(distances)
    valid_distances = distances_array[distances_array != float('inf')]
    if len(valid_distances) > 0:
        print(f"[Debug] Point-to-surface distances: min={np.min(valid_distances):.6f}, max={np.max(valid_distances):.6f}, mean={np.mean(valid_distances):.6f}, median={np.median(valid_distances):.6f}")
    else:
        print(f"[Warning] All point-to-surface distances are invalid!")
    
    return distances_array


def compute_f_score(
    scan_pcd: o3d.geometry.PointCloud,
    gt_pcd: o3d.geometry.PointCloud,
    threshold: float
) -> Tuple[float, float, float]:
    """
    Compute F-score (precision, recall, F-score) at a given distance threshold.
    
    Args:
        scan_pcd: Scan point cloud
        gt_pcd: Ground truth point cloud
        threshold: Distance threshold for considering points as "matched"
        
    Returns:
        precision: Percentage of scan points within threshold of GT
        recall: Percentage of GT points within threshold of scan
        f_score: Harmonic mean of precision and recall
    """
    print(f"[Debug] Computing F-score with threshold: {threshold:.6f}")
    
    # Compute distances
    scan_to_gt_dists = np.asarray(scan_pcd.compute_point_cloud_distance(gt_pcd))
    gt_to_scan_dists = np.asarray(gt_pcd.compute_point_cloud_distance(scan_pcd))
    
    print(f"[Debug] Scan to GT distances for F-score: min={np.min(scan_to_gt_dists):.6f}, max={np.max(scan_to_gt_dists):.6f}, mean={np.mean(scan_to_gt_dists):.6f}")
    print(f"[Debug] GT to scan distances for F-score: min={np.min(gt_to_scan_dists):.6f}, max={np.max(gt_to_scan_dists):.6f}, mean={np.mean(gt_to_scan_dists):.6f}")
    
    # Precision: fraction of scan points within threshold
    precision = np.mean(scan_to_gt_dists < threshold)
    num_within_threshold_scan = np.sum(scan_to_gt_dists < threshold)
    print(f"[Debug] Precision: {num_within_threshold_scan}/{len(scan_to_gt_dists)} points within threshold = {precision:.6f}")
    
    # Recall: fraction of GT points within threshold
    recall = np.mean(gt_to_scan_dists < threshold)
    num_within_threshold_gt = np.sum(gt_to_scan_dists < threshold)
    print(f"[Debug] Recall: {num_within_threshold_gt}/{len(gt_to_scan_dists)} points within threshold = {recall:.6f}")
    
    # F-score: harmonic mean
    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0.0
    
    print(f"[Debug] F-score: {f_score:.6f}")
    
    return precision, recall, f_score


def compute_volume_metrics(
    scan_mesh: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh
) -> Tuple[float, float, float, float]:
    """
    Compute volume metrics for two meshes.
    
    Returns:
        volume_scan: Volume of scan mesh
        volume_gt: Volume of ground truth mesh
        volume_overlap: Estimated overlap volume (simplified)
        volume_iou: Intersection over Union
    """
    # Compute volumes (requires watertight meshes with triangles)
    # Check if meshes have triangles before attempting volume computation
    if len(scan_mesh.triangles) > 0:
        try:
            volume_scan = scan_mesh.get_volume()
        except Exception:
            volume_scan = 0.0
    else:
        volume_scan = 0.0
    
    if len(gt_mesh.triangles) > 0:
        try:
            volume_gt = gt_mesh.get_volume()
        except Exception:
            volume_gt = 0.0
    else:
        volume_gt = 0.0
    
    # For overlap, we use a simplified approach:
    # Sample points from both meshes and count overlap
    # This is an approximation
    scan_pcd = mesh_to_point_cloud(scan_mesh, num_points=10000)
    gt_pcd = mesh_to_point_cloud(gt_mesh, num_points=10000)
    
    # Create KDTree for GT
    kdtree = o3d.geometry.KDTreeFlann(gt_pcd)
    
    # Find scan points near GT (within a small threshold)
    overlap_threshold = 0.01  # 1cm threshold for overlap
    scan_points = np.asarray(scan_pcd.points)
    overlap_count = 0
    
    for point in scan_points:
        result = kdtree.search_radius_vector_3d(point, overlap_threshold)
        # result is a tuple: (k, indices, distances)
        k: int = int(result[0]) if len(result) > 0 else 0
        if k > 0:
            overlap_count += 1
    
    # Estimate overlap volume as fraction of scan volume
    overlap_ratio = overlap_count / len(scan_points) if len(scan_points) > 0 else 0.0
    volume_overlap = volume_scan * overlap_ratio
    
    # IoU = overlap / union
    volume_union = volume_scan + volume_gt - volume_overlap
    volume_iou = volume_overlap / volume_union if volume_union > 0 else 0.0
    
    return volume_scan, volume_gt, volume_overlap, volume_iou


def compute_surface_area(mesh: o3d.geometry.TriangleMesh) -> float:
    """Compute surface area (covered area/Fläche) of a mesh."""
    if len(mesh.triangles) == 0:
        return 0.0
    try:
        return float(mesh.get_surface_area())
    except Exception:
        # Fallback: compute manually
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        total_area = 0.0
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            # Compute triangle area using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            total_area += area
        return float(total_area)


def count_holes(mesh: o3d.geometry.TriangleMesh, min_hole_size_ratio: float = 0.01) -> int:
    """
    Count the number of significant holes (boundary loops) in a mesh.
    
    Args:
        mesh: Input triangle mesh
        min_hole_size_ratio: Minimum hole perimeter relative to mesh bounding box diagonal (default: 1%)
    
    Returns:
        Number of holes above the size threshold
    """
    if len(mesh.triangles) == 0:
        return 0
    
    try:
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        
        # Compute mesh scale (bounding box diagonal) for threshold
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()
        mesh_diagonal = np.linalg.norm(bbox_extent)
        min_hole_perimeter = mesh_diagonal * min_hole_size_ratio
        
        # Build edge to triangle mapping
        edge_to_triangles = {}
        for i, tri in enumerate(triangles):
            edges_in_tri = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            for edge in edges_in_tri:
                if edge not in edge_to_triangles:
                    edge_to_triangles[edge] = []
                edge_to_triangles[edge].append(i)
        
        # Find boundary edges (edges that belong to only one triangle)
        boundary_edges = [edge for edge, tris in edge_to_triangles.items() if len(tris) == 1]
        
        if len(boundary_edges) == 0:
            return 0  # No holes if mesh is watertight
        
        # Build adjacency for boundary edges
        edge_to_vertices = {}
        for edge in boundary_edges:
            v0, v1 = edge[0], edge[1]
            if v0 not in edge_to_vertices:
                edge_to_vertices[v0] = []
            if v1 not in edge_to_vertices:
                edge_to_vertices[v1] = []
            edge_to_vertices[v0].append(v1)
            edge_to_vertices[v1].append(v0)
        
        # Find all closed loops (holes) and compute their perimeters
        visited_edges = set()
        hole_perimeters = []
        
        for start_vertex in list(edge_to_vertices.keys()):
            if start_vertex not in edge_to_vertices:
                continue
            
            # Find an unvisited edge starting from this vertex
            unvisited_neighbor = None
            for neighbor in edge_to_vertices[start_vertex]:
                edge_key = tuple(sorted([start_vertex, neighbor]))
                if edge_key not in visited_edges:
                    unvisited_neighbor = neighbor
                    break
            
            if unvisited_neighbor is None:
                continue
            
            # Traverse the loop
            current = start_vertex
            next_vertex = unvisited_neighbor
            loop_vertices = [start_vertex]
            loop_length = 0
            
            while True:
                edge_key = tuple(sorted([current, next_vertex]))
                if edge_key in visited_edges:
                    break
                
                visited_edges.add(edge_key)
                loop_length += 1
                
                # Move to next vertex
                current = next_vertex
                loop_vertices.append(current)
                
                # Check if we've closed the loop
                if current == start_vertex and loop_length > 2:
                    # Compute hole perimeter
                    perimeter = 0.0
                    for i in range(len(loop_vertices) - 1):
                        v0_idx = loop_vertices[i]
                        v1_idx = loop_vertices[i + 1]
                        edge_length = np.linalg.norm(vertices[v0_idx] - vertices[v1_idx])
                        perimeter += edge_length
                    hole_perimeters.append(perimeter)
                    break
                
                # Find next unvisited neighbor
                if current not in edge_to_vertices:
                    break
                
                neighbors = [v for v in edge_to_vertices[current] 
                            if tuple(sorted([current, v])) not in visited_edges]
                
                if not neighbors:
                    break
                
                next_vertex = neighbors[0]
                
                # Prevent infinite loops
                if loop_length > len(boundary_edges):
                    break
        
        # Filter holes by size
        significant_holes = [p for p in hole_perimeters if p >= min_hole_perimeter]
        num_holes = len(significant_holes)
        
        if len(hole_perimeters) > 0:
            print(f"[Debug] Found {len(hole_perimeters)} total holes, {num_holes} significant (min perimeter: {min_hole_perimeter:.6f}, mesh diagonal: {mesh_diagonal:.6f})")
        
        return num_holes
        
    except Exception as e:
        print(f"[Warning] Error counting holes: {e}")
        return 0


def compute_surface_area_and_holes(
    scan_mesh: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh,
    min_hole_size_ratio: float = 0.01
) -> Tuple[float, float, int, int]:
    """
    Compute surface area and hole count for both meshes.
    
    Args:
        scan_mesh: Scan mesh
        gt_mesh: Ground truth mesh
        min_hole_size_ratio: Minimum hole perimeter relative to mesh bounding box diagonal (default: 1%)
    
    Returns:
        surface_area_scan: Surface area of scan mesh
        surface_area_gt: Surface area of ground truth mesh
        num_holes_scan: Number of significant holes in scan mesh
        num_holes_gt: Number of significant holes in ground truth mesh
    """
    print("[Info] Computing surface area and hole counts...")
    
    surface_area_scan = compute_surface_area(scan_mesh)
    surface_area_gt = compute_surface_area(gt_mesh)
    num_holes_scan = count_holes(scan_mesh, min_hole_size_ratio=min_hole_size_ratio)
    num_holes_gt = count_holes(gt_mesh, min_hole_size_ratio=min_hole_size_ratio)
    
    print(f"[Info] Scan surface area: {surface_area_scan:.6f}, significant holes: {num_holes_scan}")
    print(f"[Info] GT surface area: {surface_area_gt:.6f}, significant holes: {num_holes_gt}")
    
    return surface_area_scan, surface_area_gt, num_holes_scan, num_holes_gt


def compute_scale_and_alignment_metrics(
    scan_mesh: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute scale and alignment metrics.
    
    Returns:
        bounding_box_ratio: (scan/gt) ratios for width, height, depth
        center_offset: Translation between mesh centers
    """
    # Get bounding boxes
    scan_bbox = scan_mesh.get_axis_aligned_bounding_box()
    gt_bbox = gt_mesh.get_axis_aligned_bounding_box()
    
    scan_extent = scan_bbox.get_extent()
    gt_extent = gt_bbox.get_extent()
    
    print(f"[Debug] Scan extent: {scan_extent}")
    print(f"[Debug] GT extent: {gt_extent}")
    
    # Compute ratios (avoid division by zero)
    ratios = tuple(
        float(s / g if g > 0 else 0.0)
        for s, g in zip(scan_extent, gt_extent)
    )
    print(f"[Debug] Bounding box ratios (scan/gt): {ratios}")
    
    # Compute center offset
    scan_center = scan_bbox.get_center()
    gt_center = gt_bbox.get_center()
    offset_array = scan_center - gt_center
    center_offset = (float(offset_array[0]), float(offset_array[1]), float(offset_array[2]))
    print(f"[Debug] Center offset: {center_offset}")
    
    return ratios, center_offset


def align_point_clouds(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    method: str = "center"
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Align source point cloud to target point cloud.
    
    Args:
        source_pcd: Source point cloud to align
        target_pcd: Target point cloud (reference)
        method: Alignment method - "center" (translate to same center), 
                "icp" (ICP registration), or "none" (no alignment)
    
    Returns:
        Aligned source point cloud and transformation matrix (4x4)
    """
    if method == "none":
        return source_pcd, np.eye(4)
    
    aligned_pcd = o3d.geometry.PointCloud(source_pcd)
    transform = np.eye(4)
    
    if method == "center":
        # Translate source to have same center as target
        source_center = np.asarray(source_pcd.get_center())
        target_center = np.asarray(target_pcd.get_center())
        translation = target_center - source_center
        transform[:3, 3] = translation
        aligned_pcd.translate(translation)
        print(f"[Info] Center alignment: translated by {translation}")
    
    elif method == "icp":
        # First center-align, then run ICP
        source_center = np.asarray(source_pcd.get_center())
        target_center = np.asarray(target_pcd.get_center())
        translation = target_center - source_center
        aligned_pcd.translate(translation)
        
        # Run ICP
        print(f"[Info] Running ICP registration...")
        result = o3d.pipelines.registration.registration_icp(
            aligned_pcd, target_pcd,
            max_correspondence_distance=0.1,  # 10cm initial threshold
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        transform_icp = result.transformation
        aligned_pcd.transform(transform_icp)
        
        # Combine transformations
        transform = transform_icp @ transform
        print(f"[Info] ICP alignment: fitness={result.fitness:.4f}, inlier_rmse={result.inlier_rmse:.6f}")
    
    return aligned_pcd, transform


def normalize_scale(
    pcd: o3d.geometry.PointCloud,
    target_scale: Optional[float] = None,
    reference_pcd: Optional[o3d.geometry.PointCloud] = None
) -> Tuple[o3d.geometry.PointCloud, float]:
    """
    Normalize scale of point cloud.
    
    Args:
        pcd: Point cloud to scale
        target_scale: Target scale (if None, uses reference_pcd scale)
        reference_pcd: Reference point cloud to match scale to
    
    Returns:
        Scaled point cloud and scale factor applied
    """
    if target_scale is None and reference_pcd is None:
        return pcd, 1.0
    
    # Compute current scale (bounding box diagonal)
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    current_scale = np.linalg.norm(extent)
    
    if target_scale is None:
        # Use reference scale
        ref_bbox = reference_pcd.get_axis_aligned_bounding_box()
        ref_extent = ref_bbox.get_extent()
        target_scale = np.linalg.norm(ref_extent)
    
    if current_scale == 0:
        return pcd, 1.0
    
    scale_factor = target_scale / current_scale
    
    # Scale around center
    center = bbox.get_center()
    scaled_pcd = o3d.geometry.PointCloud(pcd)
    scaled_pcd.translate(-center)
    scaled_pcd.scale(scale_factor, center=(0, 0, 0))
    scaled_pcd.translate(center)
    
    print(f"[Info] Scale normalization: factor={scale_factor:.6f} (from {current_scale:.6f} to {target_scale:.6f})")
    
    return scaled_pcd, scale_factor


def compare_meshes(
    scan_path: Path,
    ground_truth_path: Path,
    threshold: Optional[float] = None,
    num_sample_points: int = 50000,
    align_method: str = "none",
    scale_normalize: bool = False
) -> ComparisonReport:
    """
    Compare a scan mesh against a ground truth mesh.
    
    Args:
        scan_path: Path to the scan mesh (FBX or PLY)
        ground_truth_path: Path to the ground truth mesh (FBX or PLY)
        threshold: Distance threshold for F-score. If None, uses 1% of GT bounding box diagonal.
        num_sample_points: Number of points to sample from each mesh for comparison
        align_method: Alignment method - "none" (no alignment), "center" (center alignment), 
                      or "icp" (ICP registration)
        scale_normalize: If True, scale scan to match GT scale
        
    Returns:
        ComparisonReport with all computed metrics
    """
    warnings = []
    errors = []
    
    # Load meshes
    print(f"[Info] Loading scan mesh: {scan_path}")
    try:
        scan_mesh = load_mesh(scan_path)
        print(f"[Debug] Scan mesh loaded: {len(scan_mesh.vertices)} vertices, {len(scan_mesh.triangles)} triangles")
        scan_bbox = scan_mesh.get_axis_aligned_bounding_box()
        scan_extent = scan_bbox.get_extent()
        scan_center = scan_bbox.get_center()
        print(f"[Debug] Scan bounding box extent: ({scan_extent[0]:.6f}, {scan_extent[1]:.6f}, {scan_extent[2]:.6f})")
        print(f"[Debug] Scan bounding box center: ({scan_center[0]:.6f}, {scan_center[1]:.6f}, {scan_center[2]:.6f})")
        if len(scan_mesh.vertices) > 0:
            scan_vertices = np.asarray(scan_mesh.vertices)
            print(f"[Debug] Scan vertex range X: [{np.min(scan_vertices[:, 0]):.6f}, {np.max(scan_vertices[:, 0]):.6f}]")
            print(f"[Debug] Scan vertex range Y: [{np.min(scan_vertices[:, 1]):.6f}, {np.max(scan_vertices[:, 1]):.6f}]")
            print(f"[Debug] Scan vertex range Z: [{np.min(scan_vertices[:, 2]):.6f}, {np.max(scan_vertices[:, 2]):.6f}]")
    except Exception as e:
        errors.append(f"Failed to load scan mesh: {e}")
        raise
    
    print(f"[Info] Loading ground truth mesh: {ground_truth_path}")
    try:
        gt_mesh = load_mesh(ground_truth_path)
        print(f"[Debug] GT mesh loaded: {len(gt_mesh.vertices)} vertices, {len(gt_mesh.triangles)} triangles")
        gt_bbox = gt_mesh.get_axis_aligned_bounding_box()
        gt_extent = gt_bbox.get_extent()
        gt_center = gt_bbox.get_center()
        print(f"[Debug] GT bounding box extent: ({gt_extent[0]:.6f}, {gt_extent[1]:.6f}, {gt_extent[2]:.6f})")
        print(f"[Debug] GT bounding box center: ({gt_center[0]:.6f}, {gt_center[1]:.6f}, {gt_center[2]:.6f})")
        if len(gt_mesh.vertices) > 0:
            gt_vertices = np.asarray(gt_mesh.vertices)
            print(f"[Debug] GT vertex range X: [{np.min(gt_vertices[:, 0]):.6f}, {np.max(gt_vertices[:, 0]):.6f}]")
            print(f"[Debug] GT vertex range Y: [{np.min(gt_vertices[:, 1]):.6f}, {np.max(gt_vertices[:, 1]):.6f}]")
            print(f"[Debug] GT vertex range Z: [{np.min(gt_vertices[:, 2]):.6f}, {np.max(gt_vertices[:, 2]):.6f}]")
    except Exception as e:
        errors.append(f"Failed to load ground truth mesh: {e}")
        raise
    
    # Convert meshes to point clouds for distance computations
    print(f"[Info] Sampling {num_sample_points} points from each mesh...")
    scan_pcd = mesh_to_point_cloud(scan_mesh, num_points=num_sample_points)
    gt_pcd = mesh_to_point_cloud(gt_mesh, num_points=num_sample_points)
    print(f"[Debug] Scan point cloud: {len(scan_pcd.points)} points")
    print(f"[Debug] GT point cloud: {len(gt_pcd.points)} points")
    
    # Apply scale normalization if requested
    if scale_normalize:
        print(f"[Info] Normalizing scale...")
        scan_pcd, scale_factor = normalize_scale(scan_pcd, reference_pcd=gt_pcd)
        # Note: We don't transform the original mesh to avoid corruption
        # The point cloud is transformed for comparison, which is sufficient
    
    # Apply alignment if requested
    if align_method != "none":
        print(f"[Info] Aligning point clouds using method: {align_method}")
        scan_pcd, transform = align_point_clouds(scan_pcd, gt_pcd, method=align_method)
        # Note: We don't transform the original mesh to avoid corruption
        # The point cloud is transformed for comparison, which is sufficient
    
    # Check point cloud bounds
    if len(scan_pcd.points) > 0:
        scan_pcd_points = np.asarray(scan_pcd.points)
        print(f"[Debug] Scan PCD range X: [{np.min(scan_pcd_points[:, 0]):.6f}, {np.max(scan_pcd_points[:, 0]):.6f}]")
        print(f"[Debug] Scan PCD range Y: [{np.min(scan_pcd_points[:, 1]):.6f}, {np.max(scan_pcd_points[:, 1]):.6f}]")
        print(f"[Debug] Scan PCD range Z: [{np.min(scan_pcd_points[:, 2]):.6f}, {np.max(scan_pcd_points[:, 2]):.6f}]")
    if len(gt_pcd.points) > 0:
        gt_pcd_points = np.asarray(gt_pcd.points)
        print(f"[Debug] GT PCD range X: [{np.min(gt_pcd_points[:, 0]):.6f}, {np.max(gt_pcd_points[:, 0]):.6f}]")
        print(f"[Debug] GT PCD range Y: [{np.min(gt_pcd_points[:, 1]):.6f}, {np.max(gt_pcd_points[:, 1]):.6f}]")
        print(f"[Debug] GT PCD range Z: [{np.min(gt_pcd_points[:, 2]):.6f}, {np.max(gt_pcd_points[:, 2]):.6f}]")
    
    # Determine threshold if not provided
    if threshold is None:
        gt_bbox = gt_mesh.get_axis_aligned_bounding_box()
        gt_extent = gt_bbox.get_extent()
        gt_diagonal = np.linalg.norm(gt_extent)
        print(f"[Debug] GT bounding box extent: {gt_extent}")
        print(f"[Debug] GT bounding box diagonal: {gt_diagonal:.6f}")
        threshold = gt_diagonal * 0.01  # 1% of bounding box diagonal
        if threshold == 0.0:
            # Fallback: use a small fixed threshold or compute from point cloud distances
            print(f"[Warning] Computed threshold is 0.0, using fallback threshold")
            if len(gt_pcd.points) > 0:
                # Use 1% of the range of distances between GT points
                gt_points = np.asarray(gt_pcd.points)
                if len(gt_points) > 1:
                    # Compute pairwise distances (sample a subset for efficiency)
                    sample_size = min(1000, len(gt_points))
                    sample_indices = np.random.choice(len(gt_points), size=sample_size, replace=False)
                    sample_points = gt_points[sample_indices]
                    # Compute distances to nearest neighbors
                    kdtree = o3d.geometry.KDTreeFlann(gt_pcd)
                    sample_dists = []
                    for point in sample_points[:100]:  # Limit to 100 for speed
                        result = kdtree.search_knn_vector_3d(point, 2)  # 2 because point itself is included
                        if len(result) >= 2 and len(result[1]) > 1:
                            idx = result[1][1]  # Second nearest (first is itself)
                            dist = np.linalg.norm(point - gt_points[idx])
                            sample_dists.append(dist)
                    if len(sample_dists) > 0:
                        threshold = np.percentile(sample_dists, 50) * 0.1  # 10% of median nearest neighbor distance
                        print(f"[Debug] Using fallback threshold based on point spacing: {threshold:.6f}")
                    else:
                        threshold = 0.01  # Default 1cm
                else:
                    threshold = 0.01  # Default 1cm
            else:
                threshold = 0.01  # Default 1cm
        print(f"[Info] Using automatic threshold: {threshold:.6f} (1% of GT bounding box diagonal)")
    
    # Compute Chamfer Distance
    print("[Info] Computing Chamfer Distance...")
    chamfer_dist, scan_to_gt_dists, gt_to_scan_dists = compute_chamfer_distance(scan_pcd, gt_pcd)
    
    # Compute Hausdorff Distance
    print("[Info] Computing Hausdorff Distance...")
    hausdorff_dist = compute_hausdorff_distance(scan_pcd, gt_pcd)
    
    # Compute Point-to-Surface Distance
    print("[Info] Computing Point-to-Surface Distance...")
    point_to_surface_dists = compute_point_to_surface_distance(scan_pcd, gt_mesh)
    
    # Compute F-score
    print("[Info] Computing F-score...")
    precision, recall, f_score = compute_f_score(scan_pcd, gt_pcd, threshold)
    
    # Compute Volume Metrics
    print("[Info] Computing Volume Metrics...")
    volume_scan, volume_gt, volume_overlap, volume_iou = compute_volume_metrics(scan_mesh, gt_mesh)
    
    # Compute Scale and Alignment Metrics
    print("[Info] Computing Scale and Alignment Metrics...")
    bbox_ratio, center_offset = compute_scale_and_alignment_metrics(scan_mesh, gt_mesh)
    
    # Compute Surface Area and Hole Count
    surface_area_scan, surface_area_gt, num_holes_scan, num_holes_gt = compute_surface_area_and_holes(scan_mesh, gt_mesh)
    
    # Create metrics object
    metrics = ComparisonMetrics(
        chamfer_distance=float(chamfer_dist),
        hausdorff_distance=float(hausdorff_dist),
        mean_point_to_surface=float(np.mean(point_to_surface_dists)),
        median_point_to_surface=float(np.median(point_to_surface_dists)),
        std_point_to_surface=float(np.std(point_to_surface_dists)),
        max_point_to_surface=float(np.max(point_to_surface_dists)),
        precision=float(precision),
        recall=float(recall),
        f_score=float(f_score),
        volume_scan=float(volume_scan),
        volume_gt=float(volume_gt),
        volume_overlap=float(volume_overlap),
        volume_iou=float(volume_iou),
        bounding_box_ratio=tuple(float(x) for x in bbox_ratio),
        center_offset=tuple(float(x) for x in center_offset),
        num_scan_points=len(scan_pcd.points),
        num_gt_points=len(gt_pcd.points),
        surface_area_scan=float(surface_area_scan),
        surface_area_gt=float(surface_area_gt),
        num_holes_scan=int(num_holes_scan),
        num_holes_gt=int(num_holes_gt),
    )
    
    # Create report
    report = ComparisonReport(
        scan_path=scan_path,
        ground_truth_path=ground_truth_path,
        metrics=metrics,
        threshold=threshold,
        warnings=warnings,
        errors=errors,
    )
    
    return report


def create_error_heatmap(
    scan_pcd: o3d.geometry.PointCloud,
    gt_pcd: o3d.geometry.PointCloud,
    colormap: str = "jet"
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Create a point cloud colored by distance error (heatmap).
    
    Args:
        scan_pcd: Scan point cloud
        gt_pcd: Ground truth point cloud
        colormap: Colormap name ("jet", "viridis", "plasma", etc.)
        
    Returns:
        Colored point cloud with error visualization
    """
    # Compute distances
    distances = np.asarray(scan_pcd.compute_point_cloud_distance(gt_pcd))
    
    # Normalize distances to [0, 1] for colormap
    if np.max(distances) > 0:
        normalized = distances / np.max(distances)
    else:
        normalized = np.zeros_like(distances)
    
    # Apply colormap
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        colors = cmap(normalized)[:, :3]  # RGB only, no alpha
    except ImportError:
        # Fallback to simple colormap if matplotlib not available
        colors = np.zeros((len(normalized), 3))
        colors[:, 0] = normalized  # Red channel
        colors[:, 1] = 1.0 - normalized  # Green channel
    
    # Create colored point cloud
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = scan_pcd.points
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return colored_pcd, distances


def _create_html_report(report: ComparisonReport, distances: np.ndarray) -> str:
    """Create an HTML report with metrics and statistics."""
    m = report.metrics
    
    # Statistics about distances
    dist_mean = float(np.mean(distances))
    dist_median = float(np.median(distances))
    dist_std = float(np.std(distances))
    dist_min = float(np.min(distances))
    dist_max = float(np.max(distances))
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Mesh Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            color: #333;
        }}
        .file-info {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .file-info strong {{
            color: #2196F3;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .good {{ color: #4CAF50; font-weight: bold; }}
        .medium {{ color: #FF9800; font-weight: bold; }}
        .poor {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>3D Mesh Comparison Report</h1>
        
        <div class="file-info">
            <p><strong>Scan:</strong> {report.scan_path}</p>
            <p><strong>Ground Truth:</strong> {report.ground_truth_path}</p>
            <p><strong>Evaluation Threshold:</strong> {report.threshold:.6f}</p>
        </div>
        
        <h2>Distance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Chamfer Distance</div>
                <div class="metric-value">{m.chamfer_distance:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Hausdorff Distance</div>
                <div class="metric-value">{m.hausdorff_distance:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Point-to-Surface</div>
                <div class="metric-value">{m.mean_point_to_surface:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Median Point-to-Surface</div>
                <div class="metric-value">{m.median_point_to_surface:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Point-to-Surface</div>
                <div class="metric-value">{m.max_point_to_surface:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Std Point-to-Surface</div>
                <div class="metric-value">{m.std_point_to_surface:.6f}</div>
            </div>
        </div>
        
        <h2>F-Score Metrics (Threshold: {report.threshold:.6f})</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{m.precision:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{m.recall:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F-Score</div>
                <div class="metric-value">{m.f_score:.4f}</div>
            </div>
        </div>
        
        <h2>Volume Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Scan Volume</div>
                <div class="metric-value">{m.volume_scan:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GT Volume</div>
                <div class="metric-value">{m.volume_gt:.6f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volume IoU</div>
                <div class="metric-value">{m.volume_iou:.4f}</div>
            </div>
        </div>
        
        <h2>Scale and Alignment</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Bounding Box Ratio (scan/gt) - X</td>
                <td>{m.bounding_box_ratio[0]:.4f}</td>
            </tr>
            <tr>
                <td>Bounding Box Ratio (scan/gt) - Y</td>
                <td>{m.bounding_box_ratio[1]:.4f}</td>
            </tr>
            <tr>
                <td>Bounding Box Ratio (scan/gt) - Z</td>
                <td>{m.bounding_box_ratio[2]:.4f}</td>
            </tr>
            <tr>
                <td>Center Offset - X</td>
                <td>{m.center_offset[0]:.6f}</td>
            </tr>
            <tr>
                <td>Center Offset - Y</td>
                <td>{m.center_offset[1]:.6f}</td>
            </tr>
            <tr>
                <td>Center Offset - Z</td>
                <td>{m.center_offset[2]:.6f}</td>
            </tr>
        </table>
        
        <h2>Distance Statistics</h2>
        <table>
            <tr>
                <th>Statistic</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Mean</td>
                <td>{dist_mean:.6f}</td>
            </tr>
            <tr>
                <td>Median</td>
                <td>{dist_median:.6f}</td>
            </tr>
            <tr>
                <td>Standard Deviation</td>
                <td>{dist_std:.6f}</td>
            </tr>
            <tr>
                <td>Minimum</td>
                <td>{dist_min:.6f}</td>
            </tr>
            <tr>
                <td>Maximum</td>
                <td>{dist_max:.6f}</td>
            </tr>
        </table>
        
        <h2>Sampling Information</h2>
        <p>Scan points sampled: {m.num_scan_points:,}</p>
        <p>Ground truth points sampled: {m.num_gt_points:,}</p>
        
        <h2>Visualization Files</h2>
        <p>Error heatmap saved as: <code>error_heatmap.ply</code></p>
        <p>Open this file in a 3D viewer (e.g., MeshLab, CloudCompare) to see the color-coded error visualization.</p>
    </div>
</body>
</html>
"""
    return html


def visualize_comparison(
    report: ComparisonReport,
    output_dir: Optional[Path] = None,
    interactive: bool = True
) -> None:
    """
    Visualize the comparison results.
    
    Args:
        report: Comparison report
        output_dir: Directory to save visualization files. If None, uses scan directory.
        interactive: Whether to show interactive Open3D visualizations
    """
    if output_dir is None:
        output_dir = report.scan_path.parent / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load meshes
    scan_mesh = load_mesh(report.scan_path)
    gt_mesh = load_mesh(report.ground_truth_path)
    
    # Convert to point clouds
    scan_pcd = mesh_to_point_cloud(scan_mesh, num_points=50000)
    gt_pcd = mesh_to_point_cloud(gt_mesh, num_points=50000)
    
    # Create error heatmap
    print("[Info] Creating error heatmap...")
    colored_pcd, distances = create_error_heatmap(scan_pcd, gt_pcd)
    
    # Save error heatmap as PLY
    heatmap_path = output_dir / "error_heatmap.ply"
    o3d.io.write_point_cloud(str(heatmap_path), colored_pcd)
    print(f"[Info] Saved error heatmap to: {heatmap_path}")
    
    # Create side-by-side visualization
    if interactive:
        print("[Info] Opening interactive visualizations...")
        print("[Info] Press 'Q' or close window to continue")
        
        # Visualization 1: Error heatmap
        vis1 = o3d.visualization.Visualizer()
        vis1.create_window(window_name="Error Heatmap - Scan vs Ground Truth", width=1200, height=800)
        vis1.add_geometry(colored_pcd)
        vis1.add_geometry(gt_pcd)
        vis1.run()
        vis1.destroy_window()
        
        # Visualization 2: Side-by-side
        vis2 = o3d.visualization.Visualizer()
        vis2.create_window(window_name="Side-by-Side Comparison", width=1200, height=800)
        
        # Translate scan to the right
        scan_pcd_translated = o3d.geometry.PointCloud(scan_pcd)
        translation = np.array([2.0, 0.0, 0.0])  # Move 2 units to the right
        scan_pcd_translated.translate(translation)
        
        vis2.add_geometry(gt_pcd)
        vis2.add_geometry(scan_pcd_translated)
        vis2.run()
        vis2.destroy_window()
        
        # Visualization 3: Overlay (semi-transparent)
        vis3 = o3d.visualization.Visualizer()
        vis3.create_window(window_name="Overlay Comparison", width=1200, height=800)
        
        # Make GT semi-transparent (gray)
        gt_pcd_overlay = o3d.geometry.PointCloud(gt_pcd)
        gt_colors = np.ones((len(gt_pcd_overlay.points), 3)) * 0.5  # Gray
        gt_pcd_overlay.colors = o3d.utility.Vector3dVector(gt_colors)
        
        vis3.add_geometry(gt_pcd_overlay)
        vis3.add_geometry(colored_pcd)
        vis3.run()
        vis3.destroy_window()
    
    # Save metrics to JSON
    metrics_path = output_dir / "comparison_metrics.json"
    metrics_dict = asdict(report.metrics)
    # Convert tuples to lists for JSON serialization
    metrics_dict['bounding_box_ratio'] = list(metrics_dict['bounding_box_ratio'])
    metrics_dict['center_offset'] = list(metrics_dict['center_offset'])
    
    report_dict = {
        'scan_path': str(report.scan_path),
        'ground_truth_path': str(report.ground_truth_path),
        'threshold': report.threshold,
        'metrics': metrics_dict,
        'warnings': report.warnings,
        'errors': report.errors,
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"[Info] Saved metrics to: {metrics_path}")
    
    # Create HTML report
    html_path = output_dir / "comparison_report.html"
    html_content = _create_html_report(report, distances)
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"[Info] Saved HTML report to: {html_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Chamfer Distance: {report.metrics.chamfer_distance:.6f}")
    print(f"Hausdorff Distance: {report.metrics.hausdorff_distance:.6f}")
    print(f"Mean Point-to-Surface: {report.metrics.mean_point_to_surface:.6f}")
    print(f"Median Point-to-Surface: {report.metrics.median_point_to_surface:.6f}")
    print(f"Max Point-to-Surface: {report.metrics.max_point_to_surface:.6f}")
    print(f"\nF-score (threshold={report.threshold:.6f}):")
    print(f"  Precision: {report.metrics.precision:.4f}")
    print(f"  Recall: {report.metrics.recall:.4f}")
    print(f"  F-score: {report.metrics.f_score:.4f}")
    print(f"\nVolume Metrics:")
    print(f"  Scan Volume: {report.metrics.volume_scan:.6f}")
    print(f"  GT Volume: {report.metrics.volume_gt:.6f}")
    print(f"  Volume IoU: {report.metrics.volume_iou:.4f}")
    print(f"\nBounding Box Ratio (scan/gt): {report.metrics.bounding_box_ratio}")
    print(f"Center Offset: {report.metrics.center_offset}")
    print("="*80)


def compare_dual_scans(
    no_fog_scan_path: Path,
    fog_scan_path: Path,
    ground_truth_path: Path,
    threshold: Optional[float] = None,
    num_sample_points: int = 50000,
    align_method: str = "none",
    scale_normalize: bool = False
) -> DualComparisonReport:
    """
    Compare two scans (no_fog and fog) against ground truth.
    
    Args:
        no_fog_scan_path: Path to the no_fog (bad) scan mesh
        fog_scan_path: Path to the fog (good) scan mesh
        ground_truth_path: Path to the ground truth mesh
        threshold: Distance threshold for F-score
        num_sample_points: Number of points to sample
        align_method: Alignment method
        scale_normalize: Whether to normalize scale
        
    Returns:
        DualComparisonReport with both comparisons
    """
    print("="*80)
    print("COMPARING NO_FOG SCAN (BAD) AGAINST GROUND TRUTH")
    print("="*80)
    no_fog_report = compare_meshes(
        scan_path=no_fog_scan_path,
        ground_truth_path=ground_truth_path,
        threshold=threshold,
        num_sample_points=num_sample_points,
        align_method=align_method,
        scale_normalize=scale_normalize
    )
    
    print("\n" + "="*80)
    print("COMPARING FOG SCAN (GOOD) AGAINST GROUND TRUTH")
    print("="*80)
    fog_report = compare_meshes(
        scan_path=fog_scan_path,
        ground_truth_path=ground_truth_path,
        threshold=threshold,
        num_sample_points=num_sample_points,
        align_method=align_method,
        scale_normalize=scale_normalize
    )
    
    # Compute improvement metrics (negative = fog is better)
    no_fog_metrics = no_fog_report.metrics
    fog_metrics = fog_report.metrics
    
    improvement = {
        'chamfer_distance': fog_metrics.chamfer_distance - no_fog_metrics.chamfer_distance,  # Negative = better
        'hausdorff_distance': fog_metrics.hausdorff_distance - no_fog_metrics.hausdorff_distance,
        'mean_point_to_surface': fog_metrics.mean_point_to_surface - no_fog_metrics.mean_point_to_surface,
        'median_point_to_surface': fog_metrics.median_point_to_surface - no_fog_metrics.median_point_to_surface,
        'max_point_to_surface': fog_metrics.max_point_to_surface - no_fog_metrics.max_point_to_surface,
        'f_score': fog_metrics.f_score - no_fog_metrics.f_score,  # Positive = better
        'precision': fog_metrics.precision - no_fog_metrics.precision,  # Positive = better
        'recall': fog_metrics.recall - no_fog_metrics.recall,  # Positive = better
        'surface_area_scan': fog_metrics.surface_area_scan - no_fog_metrics.surface_area_scan,  # Positive = better (more coverage)
        'num_holes_scan': fog_metrics.num_holes_scan - no_fog_metrics.num_holes_scan,  # Negative = better (fewer holes)
    }
    
    dual_report = DualComparisonReport(
        ground_truth_path=ground_truth_path,
        no_fog_scan_path=no_fog_scan_path,
        fog_scan_path=fog_scan_path,
        no_fog_report=no_fog_report,
        fog_report=fog_report,
        improvement=improvement
    )
    
    return dual_report


def print_dual_comparison_summary(dual_report: DualComparisonReport):
    """Print a summary comparing both scans."""
    print("\n" + "="*80)
    print("DUAL COMPARISON SUMMARY: NO_FOG (BAD) vs FOG (GOOD)")
    print("="*80)
    
    no_fog = dual_report.no_fog_report.metrics
    fog = dual_report.fog_report.metrics
    imp = dual_report.improvement
    
    # Primary metrics: Surface Area and Hole Count
    print("\n" + "="*80)
    print("PRIMARY METRICS: Surface Area (Coverage) and Hole Count")
    print("="*80)
    print(f"\n{'Metric':<35} {'No_Fog (Bad)':<25} {'Fog (Good)':<25} {'Improvement':<25} {'Winner':<10}")
    print("-" * 120)
    
    # Surface Area (higher is better - more coverage)
    surface_area_ratio_no_fog = no_fog.surface_area_scan / no_fog.surface_area_gt if no_fog.surface_area_gt > 0 else 0.0
    surface_area_ratio_fog = fog.surface_area_scan / fog.surface_area_gt if fog.surface_area_gt > 0 else 0.0
    winner = "Fog ✓" if imp['surface_area_scan'] > 0 else "No_Fog"
    print(f"{'Surface Area (Coverage)':<35} {no_fog.surface_area_scan:<25.6f} {fog.surface_area_scan:<25.6f} {imp['surface_area_scan']:<25.6f} {winner:<10}")
    print(f"{'  (vs GT, ratio)':<35} {surface_area_ratio_no_fog:<25.4f} {surface_area_ratio_fog:<25.4f} {'':<25} {'':<10}")
    
    # Hole Count (lower is better - fewer holes)
    winner = "Fog ✓" if imp['num_holes_scan'] < 0 else "No_Fog"
    print(f"{'Number of Holes':<35} {no_fog.num_holes_scan:<25} {fog.num_holes_scan:<25} {imp['num_holes_scan']:<25} {winner:<10}")
    
    print("\n" + "="*80)
    print("SECONDARY METRICS: Geometric Accuracy")
    print("="*80)
    print(f"\n{'Metric':<35} {'No_Fog (Bad)':<25} {'Fog (Good)':<25} {'Improvement':<25} {'Winner':<10}")
    print("-" * 120)
    
    # Chamfer Distance (lower is better)
    winner = "Fog ✓" if imp['chamfer_distance'] < 0 else "No_Fog"
    print(f"{'Chamfer Distance':<35} {no_fog.chamfer_distance:<25.6f} {fog.chamfer_distance:<25.6f} {imp['chamfer_distance']:<25.6f} {winner:<10}")
    
    # Hausdorff Distance (lower is better)
    winner = "Fog ✓" if imp['hausdorff_distance'] < 0 else "No_Fog"
    print(f"{'Hausdorff Distance':<35} {no_fog.hausdorff_distance:<25.6f} {fog.hausdorff_distance:<25.6f} {imp['hausdorff_distance']:<25.6f} {winner:<10}")
    
    # Mean Point-to-Surface (lower is better)
    winner = "Fog ✓" if imp['mean_point_to_surface'] < 0 else "No_Fog"
    print(f"{'Mean Point-to-Surface':<35} {no_fog.mean_point_to_surface:<25.6f} {fog.mean_point_to_surface:<25.6f} {imp['mean_point_to_surface']:<25.6f} {winner:<10}")
    
    # F-score (higher is better)
    winner = "Fog ✓" if imp['f_score'] > 0 else "No_Fog"
    print(f"{'F-score':<35} {no_fog.f_score:<25.6f} {fog.f_score:<25.6f} {imp['f_score']:<25.6f} {winner:<10}")
    
    # Precision (higher is better)
    winner = "Fog ✓" if imp['precision'] > 0 else "No_Fog"
    print(f"{'Precision':<35} {no_fog.precision:<25.6f} {fog.precision:<25.6f} {imp['precision']:<25.6f} {winner:<10}")
    
    # Recall (higher is better)
    winner = "Fog ✓" if imp['recall'] > 0 else "No_Fog"
    print(f"{'Recall':<35} {no_fog.recall:<25.6f} {fog.recall:<25.6f} {imp['recall']:<25.6f} {winner:<10}")
    
    print("="*80)
    
    # Count wins in primary metrics
    primary_wins = sum([
        imp['surface_area_scan'] > 0,  # More coverage is better
        imp['num_holes_scan'] < 0,  # Fewer holes is better
    ])
    
    # Count wins in secondary metrics
    secondary_wins = sum([
        imp['chamfer_distance'] < 0,
        imp['hausdorff_distance'] < 0,
        imp['mean_point_to_surface'] < 0,
        imp['f_score'] > 0,
        imp['precision'] > 0,
        imp['recall'] > 0,
    ])
    
    print(f"\nPRIMARY METRICS: Fog scan wins {primary_wins}/2 (Surface Area, Holes)")
    print(f"SECONDARY METRICS: Fog scan wins {secondary_wins}/6 (Geometric Accuracy)")
    
    total_wins = primary_wins + secondary_wins
    if primary_wins == 2:
        print("✓✓ Fog scan is SIGNIFICANTLY BETTER (wins both primary metrics)")
    elif primary_wins == 1 and secondary_wins >= 4:
        print("✓ Fog scan is BETTER (wins 1 primary + majority of secondary)")
    elif primary_wins == 1:
        print("~ Fog scan is MIXED (wins 1 primary but loses most secondary)")
    elif secondary_wins >= 4:
        print("~ Fog scan is MIXED (loses primary but wins most secondary)")
    elif total_wins >= 5:
        print("~ Fog scan is SLIGHTLY BETTER (wins majority overall)")
    elif total_wins <= 3:
        print("✗ Fog scan is WORSE than no_fog scan")
    else:
        print("~ Fog scan is MIXED compared to no_fog scan")


def save_dual_comparison_report(dual_report: DualComparisonReport, output_dir: Path):
    """Save dual comparison report to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    report_dict = {
        'ground_truth_path': str(dual_report.ground_truth_path),
        'no_fog_scan_path': str(dual_report.no_fog_scan_path),
        'fog_scan_path': str(dual_report.fog_scan_path),
        'no_fog_metrics': asdict(dual_report.no_fog_report.metrics),
        'fog_metrics': asdict(dual_report.fog_report.metrics),
        'improvement': dual_report.improvement,
        'no_fog_threshold': dual_report.no_fog_report.threshold,
        'fog_threshold': dual_report.fog_report.threshold,
    }
    
    # Convert tuples to lists for JSON
    report_dict['no_fog_metrics']['bounding_box_ratio'] = list(report_dict['no_fog_metrics']['bounding_box_ratio'])
    report_dict['no_fog_metrics']['center_offset'] = list(report_dict['no_fog_metrics']['center_offset'])
    report_dict['fog_metrics']['bounding_box_ratio'] = list(report_dict['fog_metrics']['bounding_box_ratio'])
    report_dict['fog_metrics']['center_offset'] = list(report_dict['fog_metrics']['center_offset'])
    
    metrics_path = output_dir / "dual_comparison_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"[Info] Saved dual comparison metrics to: {metrics_path}")


def main():
    """Main entry point for the comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare reconstructed mesh scan(s) against a ground truth mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scan comparison (no alignment, no scaling)
  python compare_mesh_to_ground_truth.py --scan scan.fbx --gt ground_truth.fbx
  
  # Dual scan comparison (fog vs no_fog)
  python compare_mesh_to_ground_truth.py --no-fog-scan bad.fbx --fog-scan good.fbx --gt gt.fbx
  
  # With center alignment and scale normalization
  python compare_mesh_to_ground_truth.py --scan scan.fbx --gt gt.fbx \\
      --align center --normalize-scale
  
  # With ICP registration for full alignment
  python compare_mesh_to_ground_truth.py --scan scan.fbx --gt gt.fbx \\
      --align icp --normalize-scale
  
  # Non-interactive (no visualization windows)
  python compare_mesh_to_ground_truth.py --scan scan.fbx --gt gt.fbx --no-interactive
        """
    )
    
    # Single scan mode
    parser.add_argument(
        "--scan", "-s",
        type=Path,
        default=None,
        help="Path to the scan mesh (FBX or PLY) - for single scan mode"
    )
    
    # Dual scan mode
    parser.add_argument(
        "--no-fog-scan",
        type=Path,
        default=None,
        help="Path to the no_fog (bad) scan mesh (FBX or PLY) - for dual scan mode"
    )
    parser.add_argument(
        "--fog-scan",
        type=Path,
        default=None,
        help="Path to the fog (good) scan mesh (FBX or PLY) - for dual scan mode"
    )
    
    parser.add_argument(
        "--gt", "-g",
        type=Path,
        required=True,
        help="Path to the ground truth mesh (FBX or PLY)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Distance threshold for F-score. If not specified, uses 1%% of GT bounding box diagonal."
    )
    parser.add_argument(
        "--num-points", "-n",
        type=int,
        default=50000,
        help="Number of points to sample from each mesh (default: 50000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for visualization files. Default: scan directory / evaluation_results"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive visualizations (only save files)"
    )
    parser.add_argument(
        "--align",
        choices=["none", "center", "icp"],
        default="none",
        help="Alignment method: 'none' (no alignment), 'center' (center alignment), or 'icp' (ICP registration). Default: none"
    )
    parser.add_argument(
        "--normalize-scale",
        action="store_true",
        help="Scale scan mesh to match ground truth scale (based on bounding box diagonal)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.gt.exists():
        print(f"[Error] Ground truth file not found: {args.gt}")
        sys.exit(1)
    
    # Determine mode: dual scan or single scan
    dual_mode = args.no_fog_scan is not None and args.fog_scan is not None
    single_mode = args.scan is not None
    
    if not dual_mode and not single_mode:
        print("[Error] Must provide either --scan (single mode) or both --no-fog-scan and --fog-scan (dual mode)")
        sys.exit(1)
    
    if dual_mode and single_mode:
        print("[Error] Cannot use both single scan mode (--scan) and dual scan mode (--no-fog-scan/--fog-scan)")
        sys.exit(1)
    
    try:
        if dual_mode:
            # Dual scan mode
            if not args.no_fog_scan.exists():
                print(f"[Error] No_fog scan file not found: {args.no_fog_scan}")
                sys.exit(1)
            
            if not args.fog_scan.exists():
                print(f"[Error] Fog scan file not found: {args.fog_scan}")
                sys.exit(1)
            
            # Run dual comparison
            dual_report = compare_dual_scans(
                no_fog_scan_path=args.no_fog_scan,
                fog_scan_path=args.fog_scan,
                ground_truth_path=args.gt,
                threshold=args.threshold,
                num_sample_points=args.num_points,
                align_method=args.align,
                scale_normalize=args.normalize_scale
            )
            
            # Print summary
            print_dual_comparison_summary(dual_report)
            
            # Save reports
            if args.output is None:
                output_dir = args.fog_scan.parent / "evaluation_results"
            else:
                output_dir = args.output
            
            # Save individual reports
            visualize_comparison(
                report=dual_report.no_fog_report,
                output_dir=output_dir / "no_fog",
                interactive=not args.no_interactive
            )
            visualize_comparison(
                report=dual_report.fog_report,
                output_dir=output_dir / "fog",
                interactive=not args.no_interactive
            )
            
            # Save dual comparison report
            save_dual_comparison_report(dual_report, output_dir)
            
            print("\n[Info] Dual comparison complete!")
        
        else:
            # Single scan mode
            if not args.scan.exists():
                print(f"[Error] Scan file not found: {args.scan}")
                sys.exit(1)
            
            report = compare_meshes(
                scan_path=args.scan,
                ground_truth_path=args.gt,
                threshold=args.threshold,
                num_sample_points=args.num_points,
                align_method=args.align,
                scale_normalize=args.normalize_scale
            )
            
            # Visualize results
            visualize_comparison(
                report=report,
                output_dir=args.output,
                interactive=not args.no_interactive
            )
            
            print("\n[Info] Comparison complete!")
        
    except Exception as e:
        print(f"[Error] Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


