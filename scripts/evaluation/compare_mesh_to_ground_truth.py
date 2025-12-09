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
import tempfile
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
    import sys
    from pathlib import Path
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


@dataclass
class ComparisonReport:
    """Complete comparison report."""
    scan_path: Path
    ground_truth_path: Path
    metrics: ComparisonMetrics
    threshold: float  # Distance threshold used for F-score
    warnings: list[str]
    errors: list[str]


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
    # Compute distances from scan to GT
    scan_to_gt_dists = np.asarray(scan_pcd.compute_point_cloud_distance(gt_pcd))
    
    # Compute distances from GT to scan
    gt_to_scan_dists = np.asarray(gt_pcd.compute_point_cloud_distance(scan_pcd))
    
    # Chamfer distance is the average of both directions
    chamfer_dist = np.mean(scan_to_gt_dists) + np.mean(gt_to_scan_dists)
    
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
    # Create KDTree from GT mesh vertices for fast lookup
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = gt_mesh.vertices
    
    # For each scan point, find nearest GT vertex
    scan_points = np.asarray(scan_pcd.points)
    gt_points = np.asarray(gt_pcd.points)
    
    # Use Open3D's KDTree for efficient nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(gt_pcd)
    
    distances = []
    for point in scan_points:
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        nearest_gt_point = gt_points[idx[0]]
        dist = np.linalg.norm(point - nearest_gt_point)
        distances.append(dist)
    
    return np.array(distances)


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
    # Compute distances
    scan_to_gt_dists = np.asarray(scan_pcd.compute_point_cloud_distance(gt_pcd))
    gt_to_scan_dists = np.asarray(gt_pcd.compute_point_cloud_distance(scan_pcd))
    
    # Precision: fraction of scan points within threshold
    precision = np.mean(scan_to_gt_dists < threshold)
    
    # Recall: fraction of GT points within threshold
    recall = np.mean(gt_to_scan_dists < threshold)
    
    # F-score: harmonic mean
    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0.0
    
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
    # Compute volumes (requires watertight meshes)
    try:
        volume_scan = scan_mesh.get_volume()
    except Exception:
        volume_scan = 0.0
    
    try:
        volume_gt = gt_mesh.get_volume()
    except Exception:
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
        [k, _, _] = kdtree.search_radius_vector_3d(point, overlap_threshold)
        if k > 0:
            overlap_count += 1
    
    # Estimate overlap volume as fraction of scan volume
    overlap_ratio = overlap_count / len(scan_points) if len(scan_points) > 0 else 0.0
    volume_overlap = volume_scan * overlap_ratio
    
    # IoU = overlap / union
    volume_union = volume_scan + volume_gt - volume_overlap
    volume_iou = volume_overlap / volume_union if volume_union > 0 else 0.0
    
    return volume_scan, volume_gt, volume_overlap, volume_iou


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
    
    # Compute ratios (avoid division by zero)
    ratios = tuple(
        s / g if g > 0 else 0.0
        for s, g in zip(scan_extent, gt_extent)
    )
    
    # Compute center offset
    scan_center = scan_bbox.get_center()
    gt_center = gt_bbox.get_center()
    center_offset = tuple((scan_center - gt_center).tolist())
    
    return ratios, center_offset


def compare_meshes(
    scan_path: Path,
    ground_truth_path: Path,
    threshold: Optional[float] = None,
    num_sample_points: int = 50000
) -> ComparisonReport:
    """
    Compare a scan mesh against a ground truth mesh.
    
    Args:
        scan_path: Path to the scan mesh (FBX or PLY)
        ground_truth_path: Path to the ground truth mesh (FBX or PLY)
        threshold: Distance threshold for F-score. If None, uses 1% of GT bounding box diagonal.
        num_sample_points: Number of points to sample from each mesh for comparison
        
    Returns:
        ComparisonReport with all computed metrics
    """
    warnings = []
    errors = []
    
    # Load meshes
    print(f"[Info] Loading scan mesh: {scan_path}")
    try:
        scan_mesh = load_mesh(scan_path)
    except Exception as e:
        errors.append(f"Failed to load scan mesh: {e}")
        raise
    
    print(f"[Info] Loading ground truth mesh: {ground_truth_path}")
    try:
        gt_mesh = load_mesh(ground_truth_path)
    except Exception as e:
        errors.append(f"Failed to load ground truth mesh: {e}")
        raise
    
    # Convert meshes to point clouds for distance computations
    print(f"[Info] Sampling {num_sample_points} points from each mesh...")
    scan_pcd = mesh_to_point_cloud(scan_mesh, num_points=num_sample_points)
    gt_pcd = mesh_to_point_cloud(gt_mesh, num_points=num_sample_points)
    
    # Determine threshold if not provided
    if threshold is None:
        gt_bbox = gt_mesh.get_axis_aligned_bounding_box()
        gt_diagonal = np.linalg.norm(gt_bbox.get_extent())
        threshold = gt_diagonal * 0.01  # 1% of bounding box diagonal
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
        bounding_box_ratio=bbox_ratio,
        center_offset=center_offset,
        num_scan_points=len(scan_pcd.points),
        num_gt_points=len(gt_pcd.points),
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
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    colors = cmap(normalized)[:, :3]  # RGB only, no alpha
    
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


def main():
    """Main entry point for the comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare a reconstructed mesh scan against a ground truth mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python compare_mesh_to_ground_truth.py --scan scan.fbx --gt ground_truth.fbx
  
  # With custom threshold and output directory
  python compare_mesh_to_ground_truth.py --scan scan.fbx --gt gt.fbx \\
      --threshold 0.01 --output results/
  
  # Non-interactive (no visualization windows)
  python compare_mesh_to_ground_truth.py --scan scan.fbx --gt gt.fbx --no-interactive
        """
    )
    
    parser.add_argument(
        "--scan", "-s",
        type=Path,
        required=True,
        help="Path to the scan mesh (FBX or PLY)"
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
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.scan.exists():
        print(f"[Error] Scan file not found: {args.scan}")
        sys.exit(1)
    
    if not args.gt.exists():
        print(f"[Error] Ground truth file not found: {args.gt}")
        sys.exit(1)
    
    # Run comparison
    try:
        report = compare_meshes(
            scan_path=args.scan,
            ground_truth_path=args.gt,
            threshold=args.threshold,
            num_sample_points=args.num_points
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

