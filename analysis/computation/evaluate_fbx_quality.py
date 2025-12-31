#!/usr/bin/env python3
"""
Intrinsic FBX mesh quality evaluation (geometry, smoothness, completeness, color).

This script implements the refined quality score described in the thesis:

    Q = 0.50 * S_geom + 0.25 * S_smooth + 0.15 * S_complete + 0.10 * S_color

with batch-wise min–max normalization of all underlying metrics.

ARGUMENTS:
    meshes              Unpaired mesh paths (FBX/PLY). Ignored in pair mode if --pair or --from-csv is provided.
    --pair FOG NOFOG    Fog and no-fog mesh paths for a pair (can be used multiple times).
    --from-csv PATH     Load fog/no-fog pairs automatically from CSV file. Only pairs where both meshes exist will be included.
    --csv PATH          Optional path to write detailed CSV with all metrics and scores.
    --out-dir PATH      Output directory for batch artifacts (plots, pairwise summary). Default: analysis/mesh_quality_batch
    --num-workers INT   Number of parallel workers for mesh processing. Defaults to CPU count. Use 1 for sequential processing.

Three usage modes are supported:

1) Single / unpaired meshes
   - Evaluate one or more meshes independently.
   - Example:
        python analysis/computation/evaluate_fbx_quality.py mesh1.fbx mesh2.fbx --num-workers 4

2) Paired fog / no-fog evaluation (manual pairs)
   - Explicitly pass fog / no-fog meshes as pairs. The script will:
       * use names "<pair_index>_fog" and "<pair_index>_nofog" for reporting
       * compute relative rankings across the whole batch
       * print per-pair improvements similar to the thesis examples
   - Example:
        python analysis/computation/evaluate_fbx_quality.py \\
            --pair fog1.fbx nofog1.fbx \\
            --pair fog2.fbx nofog2.fbx \\
            --num-workers 4

3) Automatic pair loading from CSV (recommended for batch analysis)
   - Load pairs automatically from master_fog_no_fog_report.csv
   - Only pairs where both meshes exist (color_mesh_present=True) are included
   - Example:
        python analysis/computation/evaluate_fbx_quality.py --from-csv analysis/data/master_fog_no_fog_report.csv --num-workers 4
"""

from __future__ import annotations

import argparse
import csv
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any
import multiprocessing

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

try:
    from analysis.computation.mesh_loader import load_mesh
except ImportError:
    # Script execution fallback - try relative import
    try:
        from .mesh_loader import load_mesh
    except ImportError:
        # Last resort - try importing from same directory
        import sys
        script_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(script_dir))
        from mesh_loader import load_mesh


# -----------------------------------------------------------------------------
# Low-level geometry helpers
# -----------------------------------------------------------------------------


def compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in radians."""
    v1n = np.divide(v1, (np.linalg.norm(v1) + 1e-12))
    v2n = np.divide(v2, (np.linalg.norm(v2) + 1e-12))
    cos = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    return float(np.arccos(cos))


def triangle_aspect_ratio(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute triangle aspect ratio: longest edge / shortest edge."""
    edges = [
        float(np.linalg.norm(v1 - v0)),
        float(np.linalg.norm(v2 - v1)),
        float(np.linalg.norm(v0 - v2)),
    ]
    min_e = max(min(edges), 1e-12)
    return float(max(edges) / min_e)


def triangle_skewness(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute equiangle skewness (0 = equilateral, 1 = degenerate).

    Implementation follows the guidance:
        skew = max(
            (theta_ideal - min_angle) / theta_ideal,
            (max_angle - theta_ideal) / theta_ideal
        )
    """
    angles = [
        compute_angle(v1 - v0, v2 - v0),  # at v0
        compute_angle(v2 - v1, v0 - v1),  # at v1
        compute_angle(v0 - v2, v1 - v2),  # at v2
    ]
    theta_ideal = np.radians(60.0)
    skew = max(
        (theta_ideal - min(angles)) / theta_ideal,
        (max(angles) - theta_ideal) / theta_ideal,
    )
    return float(np.clip(skew, 0.0, 1.0))


def dihedral_angle(tri1_normal: np.ndarray, tri2_normal: np.ndarray) -> float:
    """Angle between two adjacent triangle normals (0–180°)."""
    n1 = tri1_normal / (np.linalg.norm(tri1_normal) + 1e-12)
    n2 = tri2_normal / (np.linalg.norm(tri2_normal) + 1e-12)
    cos_angle = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] with min–max scaling.

    Degenerate (constant) arrays map to 0.5 everywhere.
    """
    if values.size == 0:
        return np.zeros_like(values, dtype=float)
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if np.isclose(v_min, v_max):
        return np.full_like(values, 0.5, dtype=float)
    return (values - v_min) / (v_max - v_min)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# -----------------------------------------------------------------------------
# Topological / smoothness feature extraction
# -----------------------------------------------------------------------------


@dataclass
class RawMeshMetrics:
    """Per-mesh raw statistics before batch normalization."""

    name: str
    path: Path

    # Shape / topology
    mean_aspect_ratio: float
    mean_skewness: float
    degenerate_triangles: int
    non_manifold_edges: int
    boundary_edge_ratio: float
    component_count: int
    total_edges: int

    # Smoothness
    normal_deviation_avg_deg: float
    dihedral_min_deg: float
    dihedral_max_deg: float
    dihedral_penalty: float
    surface_roughness: float  # here: stddev of dihedral angles

    # Completeness
    is_single_component: bool
    vertex_density_stddev: float

    # Color
    has_color: bool
    uncolored_vertex_ratio: float
    color_gradient_stddev: float

    # Derived booleans
    is_manifold: bool
    is_watertight: bool

    # Basic size diagnostics (for reporting)
    num_vertices: int
    num_triangles: int


def build_topology(
    triangles: np.ndarray, num_vertices: int
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[int]]]:
    """
    Build edge→faces and vertex→vertices adjacency from triangle indices.
    """
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    vertex_adj: Dict[int, List[int]] = {i: [] for i in range(num_vertices)}

    for f_idx, tri in enumerate(triangles):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        edges = [(i0, i1), (i1, i2), (i2, i0)]
        for u, v in edges:
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(e, []).append(f_idx)
            if v not in vertex_adj[u]:
                vertex_adj[u].append(v)
            if u not in vertex_adj[v]:
                vertex_adj[v].append(u)

    return edge_to_faces, vertex_adj


def count_components(vertex_adj: Dict[int, List[int]]) -> int:
    """Count connected components in the vertex adjacency graph."""
    visited = set()
    components = 0

    for v in vertex_adj.keys():
        if v in visited:
            continue
        components += 1
        stack = [v]
        visited.add(v)
        while stack:
            cur = stack.pop()
            for nb in vertex_adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
    return components


def compute_vertex_normals(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Ensure vertex normals exist and return as np.ndarray."""
    local_mesh = o3d.geometry.TriangleMesh(mesh)  # copy
    local_mesh.compute_vertex_normals()
    v_normals = np.asarray(local_mesh.vertex_normals)
    # Explicitly drop temporary mesh to help GC with large geometries
    del local_mesh
    return v_normals


def compute_triangle_normals(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Ensure triangle normals exist and return as np.ndarray."""
    local_mesh = o3d.geometry.TriangleMesh(mesh)  # copy
    local_mesh.compute_triangle_normals()
    t_normals = np.asarray(local_mesh.triangle_normals)
    del local_mesh
    return t_normals


def compute_raw_metrics_for_mesh(path: Path, name: str) -> RawMeshMetrics:
    """Extract all raw per-mesh metrics from a mesh file."""
    mesh = load_mesh(path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles, dtype=int)

    num_vertices = vertices.shape[0]
    num_triangles = triangles.shape[0]

    if num_vertices == 0 or num_triangles == 0:
        raise ValueError(f"Mesh {path} has no geometry (vertices={num_vertices}, triangles={num_triangles})")

    # ------------------------------------------------------------------
    # Shape metrics: aspect ratio & skewness over all triangles
    # ------------------------------------------------------------------
    aspect_ratios: List[float] = []
    skewness_vals: List[float] = []
    degenerate_triangles = 0

    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

        # Detect degenerates: repeated vertices or tiny area
        if i0 == i1 or i1 == i2 or i2 == i0:
            degenerate_triangles += 1
            continue
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        if area < 1e-10:
            degenerate_triangles += 1
            continue

        ar = triangle_aspect_ratio(v0, v1, v2)
        sk = triangle_skewness(v0, v1, v2)
        aspect_ratios.append(ar)
        skewness_vals.append(sk)

    mean_aspect_ratio = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0
    mean_skewness = float(np.mean(skewness_vals)) if skewness_vals else 0.0

    # ------------------------------------------------------------------
    # Topology: edges, boundary, manifoldness, components
    # ------------------------------------------------------------------
    edge_to_faces, vertex_adj = build_topology(triangles, num_vertices)
    total_edges = len(edge_to_faces)

    boundary_edges = 0
    non_manifold_edges = 0
    for faces in edge_to_faces.values():
        if len(faces) == 1:
            boundary_edges += 1
        elif len(faces) > 2:
            non_manifold_edges += 1

    boundary_edge_ratio = float(boundary_edges / total_edges) if total_edges > 0 else 0.0
    component_count = count_components(vertex_adj)

    is_manifold = non_manifold_edges == 0
    is_watertight = is_manifold and boundary_edges == 0 and component_count == 1

    # ------------------------------------------------------------------
    # Smoothness: normal deviation, dihedral extremes, surface roughness
    # ------------------------------------------------------------------
    v_normals = compute_vertex_normals(mesh)
    t_normals = compute_triangle_normals(mesh)

    # Vertex normal deviation along edges
    normal_deviations: List[float] = []
    for (u, v) in edge_to_faces.keys():
        n1 = v_normals[u]
        n2 = v_normals[v]
        angle_rad = np.arccos(
            np.clip(
                float(np.dot(n1, n2) / ((np.linalg.norm(n1) + 1e-12) * (np.linalg.norm(n2) + 1e-12))),
                -1.0,
                1.0,
            )
        )
        normal_deviations.append(float(np.degrees(angle_rad)))

    normal_deviation_avg = float(np.mean(normal_deviations)) if normal_deviations else 0.0

    # Dihedral angles
    dihedral_angles: List[float] = []
    for (u, v), faces in edge_to_faces.items():
        if len(faces) == 2:
            f0, f1 = faces[0], faces[1]
            a = dihedral_angle(t_normals[f0], t_normals[f1])
            dihedral_angles.append(a)

    if dihedral_angles:
        dihed_min = float(np.min(dihedral_angles))
        dihed_max = float(np.max(dihedral_angles))
        dihedral_penalty = max(0.0, 30.0 - dihed_min) + max(0.0, dihed_max - 170.0)
        surface_roughness = float(np.std(dihedral_angles))
    else:
        dihed_min = 180.0
        dihed_max = 0.0
        dihedral_penalty = 0.0
        surface_roughness = 0.0

    # ------------------------------------------------------------------
    # Completeness: vertex density variation across a voxel grid
    # ------------------------------------------------------------------
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    # Guard against zero-size dimensions
    extent[extent == 0.0] = 1e-6

    grid_res = 10  # 10 x 10 x 10 grid
    voxel_sizes = extent / grid_res
    voxel_vol = np.prod(voxel_sizes)
    if voxel_vol <= 0.0:
        voxel_vol = 1.0

    # Map each vertex to its voxel index
    rel_pos = (vertices - np.asarray(bbox.min_bound)) / voxel_sizes
    indices = np.floor(rel_pos).astype(int)
    indices = np.clip(indices, 0, grid_res - 1)

    counts = np.zeros((grid_res, grid_res, grid_res), dtype=int)
    for ix, iy, iz in indices:
        counts[ix, iy, iz] += 1

    densities = counts.astype(float) / voxel_vol
    flat = densities.flatten()
    # Consider only non-empty voxels for stddev to avoid dominating zeros
    non_zero = flat[flat > 0]
    if non_zero.size == 0:
        vertex_density_stddev = 0.0
    else:
        vertex_density_stddev = float(np.std(non_zero))

    is_single_component = component_count == 1

    # ------------------------------------------------------------------
    # Color metrics
    # ------------------------------------------------------------------
    has_color = mesh.has_vertex_colors()
    if has_color:
        colors = np.asarray(mesh.vertex_colors)
        if colors.shape[0] != num_vertices:
            # Defensive: mismatch shouldn't happen, but guard it
            colors = np.resize(colors, (num_vertices, 3))
        # Treat presence of any color as colored; no clear notion of "uncolored vertex"
        uncolored_vertex_ratio = 0.0

        # Compute color gradients along edges
        diffs: List[float] = []
        for (u, v) in edge_to_faces.keys():
            c1 = colors[u]
            c2 = colors[v]
            diffs.append(float(np.linalg.norm(c1 - c2)))
        color_gradient_stddev = float(np.std(diffs)) if diffs else 0.0
    else:
        uncolored_vertex_ratio = 1.0
        color_gradient_stddev = 0.0

    # Help GC by removing large intermediate arrays and mesh references
    del vertices, triangles, edge_to_faces, vertex_adj, counts, densities, flat, non_zero
    del v_normals, t_normals
    del mesh
    gc.collect()

    return RawMeshMetrics(
        name=name,
        path=path,
        mean_aspect_ratio=mean_aspect_ratio,
        mean_skewness=mean_skewness,
        degenerate_triangles=degenerate_triangles,
        non_manifold_edges=non_manifold_edges,
        boundary_edge_ratio=boundary_edge_ratio,
        component_count=component_count,
        total_edges=total_edges,
        normal_deviation_avg_deg=normal_deviation_avg,
        dihedral_min_deg=dihed_min,
        dihedral_max_deg=dihed_max,
        dihedral_penalty=dihedral_penalty,
        surface_roughness=surface_roughness,
        is_single_component=is_single_component,
        vertex_density_stddev=vertex_density_stddev,
        has_color=has_color,
        uncolored_vertex_ratio=uncolored_vertex_ratio,
        color_gradient_stddev=color_gradient_stddev,
        is_manifold=is_manifold,
        is_watertight=is_watertight,
        num_vertices=num_vertices,
        num_triangles=num_triangles,
    )


# -----------------------------------------------------------------------------
# Batch normalization and quality score computation
# -----------------------------------------------------------------------------


@dataclass
class QualityScores:
    """Per-mesh quality scores after batch normalization."""

    name: str
    path: Path

    # Subscores
    S_shape: float
    S_topology: float
    S_bonuses: float
    S_geom: float
    S_smooth: float
    S_complete: float
    S_color: float

    # Overall score (absolute, before batch normalization)
    Q_raw: float

    # Relative score (normalized Q over batch)
    Q_norm: float

    # Raw metrics (for reporting / CSV)
    raw: RawMeshMetrics


def compute_quality_scores(raw_metrics: List[RawMeshMetrics]) -> List[QualityScores]:
    """Compute batch-normalized quality scores for all meshes."""
    n = len(raw_metrics)
    if n == 0:
        return []

    # Collect raw arrays for batch normalization
    ar = np.array([m.mean_aspect_ratio for m in raw_metrics], dtype=float)
    skew = np.array([m.mean_skewness for m in raw_metrics], dtype=float)
    deg_count = np.array([m.degenerate_triangles for m in raw_metrics], dtype=float)
    nonmanifold = np.array([m.non_manifold_edges for m in raw_metrics], dtype=float)
    boundary_ratio = np.array([m.boundary_edge_ratio for m in raw_metrics], dtype=float)
    comp_minus1 = np.array([max(0, m.component_count - 1) for m in raw_metrics], dtype=float)

    normal_dev = np.array([m.normal_deviation_avg_deg for m in raw_metrics], dtype=float)
    dihedral_pen = np.array([m.dihedral_penalty for m in raw_metrics], dtype=float)
    surface_rough = np.array([m.surface_roughness for m in raw_metrics], dtype=float)

    vertex_density_std = np.array([m.vertex_density_stddev for m in raw_metrics], dtype=float)

    uncolored_ratio = np.array([m.uncolored_vertex_ratio for m in raw_metrics], dtype=float)
    color_grad_std = np.array([m.color_gradient_stddev for m in raw_metrics], dtype=float)

    # Normalize all "badness" metrics (higher = worse) into [0, 1]
    ar_norm = min_max_normalize(ar)
    skew_norm = min_max_normalize(skew)
    deg_norm = min_max_normalize(deg_count)
    nonmanifold_norm = min_max_normalize(nonmanifold)
    boundary_norm = min_max_normalize(boundary_ratio)
    comp_norm = min_max_normalize(comp_minus1)

    normal_dev_norm = min_max_normalize(normal_dev)
    dihedral_pen_norm = min_max_normalize(dihedral_pen)
    surface_rough_norm = min_max_normalize(surface_rough)

    vertex_density_std_norm = min_max_normalize(vertex_density_std)

    uncolored_ratio_norm = min_max_normalize(uncolored_ratio)
    color_grad_std_norm = min_max_normalize(color_grad_std)

    scores: List[QualityScores] = []

    for i, m in enumerate(raw_metrics):
        # -----------------------------
        # Geometry
        # -----------------------------
        S_shape = 0.5 * (1.0 - ar_norm[i]) + 0.5 * (1.0 - skew_norm[i])

        S_topology = (
            0.4 * (1.0 - deg_norm[i])
            + 0.3 * (1.0 - nonmanifold_norm[i])
            + 0.2 * (1.0 - boundary_norm[i])
            + 0.1 * (1.0 - comp_norm[i])
        )

        S_bonuses = 0.5 * (1.0 if m.is_manifold else 0.0) + 0.5 * (1.0 if m.is_watertight else 0.0)

        S_geom = 0.25 * S_shape + 0.15 * S_topology + 0.10 * S_bonuses

        # -----------------------------
        # Smoothness
        # -----------------------------
        S_smooth = (
            0.48 * (1.0 - normal_dev_norm[i])
            + 0.32 * (1.0 - dihedral_pen_norm[i])
            + 0.20 * (1.0 - surface_rough_norm[i])
        )

        # -----------------------------
        # Completeness
        # -----------------------------
        S_complete = (
            0.50 * (1.0 - m.boundary_edge_ratio)
            + 0.30 * (1.0 if m.is_single_component else 0.0)
            + 0.20 * (1.0 - vertex_density_std_norm[i])
        )

        # -----------------------------
        # Color
        # -----------------------------
        if m.has_color:
            S_color = 0.5 * (1.0 - uncolored_ratio_norm[i]) + 0.5 * (1.0 - color_grad_std_norm[i])
        else:
            # Neutral score when no color is available
            S_color = 0.5

        # -----------------------------
        # Overall quality
        # -----------------------------
        Q_raw = 0.50 * S_geom + 0.25 * S_smooth + 0.15 * S_complete + 0.10 * S_color

        scores.append(
            QualityScores(
                name=m.name,
                path=m.path,
                S_shape=S_shape,
                S_topology=S_topology,
                S_bonuses=S_bonuses,
                S_geom=S_geom,
                S_smooth=S_smooth,
                S_complete=S_complete,
                S_color=S_color,
                Q_raw=Q_raw,
                Q_norm=0.0,  # filled below
                raw=m,
            )
        )

    # Batch-normalize Q for relative ranking
    Q_arr = np.array([s.Q_raw for s in scores], dtype=float)
    Q_norm = min_max_normalize(Q_arr)
    for i, s in enumerate(scores):
        s.Q_norm = float(Q_norm[i])

    return scores


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def write_csv(scores: List[QualityScores], csv_path: Path) -> None:
    """Write detailed metrics and scores to CSV for downstream analysis."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = [
        "name",
        "path",
        "Q_raw",
        "Q_norm",
        "S_geom",
        "S_smooth",
        "S_complete",
        "S_color",
        "S_shape",
        "S_topology",
        "S_bonuses",
        "mean_aspect_ratio",
        "mean_skewness",
        "degenerate_triangles",
        "non_manifold_edges",
        "boundary_edge_ratio",
        "component_count",
        "total_edges",
        "normal_deviation_avg_deg",
        "dihedral_min_deg",
        "dihedral_max_deg",
        "dihedral_penalty",
        "surface_roughness",
        "is_single_component",
        "vertex_density_stddev",
        "has_color",
        "uncolored_vertex_ratio",
        "color_gradient_stddev",
        "is_manifold",
        "is_watertight",
        "num_vertices",
        "num_triangles",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scores:
            m = s.raw
            row = {
                "name": s.name,
                "path": str(s.path),
                "Q_raw": s.Q_raw,
                "Q_norm": s.Q_norm,
                "S_geom": s.S_geom,
                "S_smooth": s.S_smooth,
                "S_complete": s.S_complete,
                "S_color": s.S_color,
                "S_shape": s.S_shape,
                "S_topology": s.S_topology,
                "S_bonuses": s.S_bonuses,
            }
            row.update(asdict(m))
            writer.writerow(row)


def print_batch_summary(scores: List[QualityScores]) -> None:
    """Print ranking summary across all meshes."""
    if not scores:
        return

    sorted_scores = sorted(scores, key=lambda s: s.Q_norm, reverse=True)
    print("\n" + "=" * 80)
    print("BATCH QUALITY RANKING (higher Q_norm is better)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Name':<25} {'Q_norm':<8} {'Q_raw':<8} {'S_geom':<8} {'S_smooth':<8} {'S_complete':<10} {'S_color':<8}")
    print("-" * 80)

    for rank, s in enumerate(sorted_scores, start=1):
        print(
            f"{rank:<6} {s.name:<25} "
            f"{s.Q_norm:>7.3f} {s.Q_raw:>7.3f} "
            f"{s.S_geom:>7.3f} {s.S_smooth:>7.3f} "
            f"{s.S_complete:>9.3f} {s.S_color:>7.3f}"
        )


def print_pair_summaries(pairs: List[Tuple[QualityScores, QualityScores]]) -> None:
    """
    Print per-pair fog / no-fog summaries.

    Each pair is (fog_score, nofog_score) and is expected to have names
    "<idx>_fog" and "<idx>_nofog" respectively.
    """
    if not pairs:
        return

    all_scores: List[QualityScores] = [s for pair in pairs for s in pair]
    # For rank reporting, use the batch ranking order
    sorted_all = sorted(all_scores, key=lambda s: s.Q_norm, reverse=True)
    rank_map: Dict[str, int] = {s.name: i + 1 for i, s in enumerate(sorted_all)}

    total = len(all_scores)

    print("\n" + "=" * 80)
    print("FOG vs NO-FOG PAIR ANALYSIS")
    print("=" * 80)

    for idx, (fog, nofog) in enumerate(pairs, start=1):
        fog_rank = rank_map.get(fog.name, -1)
        nofog_rank = rank_map.get(nofog.name, -1)
        improvement = fog.Q_norm - nofog.Q_norm

        print(f"\nPair {idx}:")
        print(f"  {fog.name:<20}: Q = {fog.Q_norm:.3f} (rank: {fog_rank}/{total})")
        print(f"  {nofog.name:<20}: Q = {nofog.Q_norm:.3f} (rank: {nofog_rank}/{total})")
        print(f"  Improvement (fog - nofog): {improvement:+.3f} (relative)")

        # Optional short absolute summary for interpretability
        fm = fog.raw
        nm = nofog.raw
        print("  Fog mesh absolute metrics:")
        print(
            f"    - Aspect ratio (mean): {fm.mean_aspect_ratio:.2f} "
            f"(degenerate tris: {fm.degenerate_triangles})"
        )
        print(f"    - Skewness (mean):     {fm.mean_skewness:.2f}")
        print(f"    - Normal deviation:    {fm.normal_deviation_avg_deg:.2f}°")
        print(
            f"    - Dihedral range:      min={fm.dihedral_min_deg:.1f}°, max={fm.dihedral_max_deg:.1f}°"
        )
        print(
            f"    - Components:          {fm.component_count}, "
            f"boundary edges: {fm.boundary_edge_ratio * 100.0:.1f}%"
        )

        print("  No-fog mesh absolute metrics:")
        print(
            f"    - Aspect ratio (mean): {nm.mean_aspect_ratio:.2f} "
            f"(degenerate tris: {nm.degenerate_triangles})"
        )
        print(f"    - Skewness (mean):     {nm.mean_skewness:.2f}")
        print(f"    - Normal deviation:    {nm.normal_deviation_avg_deg:.2f}°")
        print(
            f"    - Dihedral range:      min={nm.dihedral_min_deg:.1f}°, max={nm.dihedral_max_deg:.1f}°"
        )
        print(
            f"    - Components:          {nm.component_count}, "
            f"boundary edges: {nm.boundary_edge_ratio * 100.0:.1f}%"
        )


def write_pairwise_reports(
    scores: List[QualityScores],
    pair_indices: List[Tuple[int, int]],
    pair_meta: List[Dict[str, str]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create name-to-score mapping for reliable lookup
    scores_by_name: Dict[str, QualityScores] = {s.name: s for s in scores}

    summary_csv = out_dir / "pairwise_summary.csv"
    rows = []
    labels = []
    fog_vals = []
    nofog_vals = []
    deltas = []

    for i, (fog_idx, nofog_idx) in enumerate(pair_indices):
        meta = pair_meta[i] if i < len(pair_meta) else {
            "participant": f"pair{i+1}",
            "pair_id": f"pair{i+1}",
            "fog_name": "",
            "nofog_name": "",
        }
        
        # Use names from metadata for reliable matching (more robust than indices)
        fog_name = meta.get("fog_name", "")
        nofog_name = meta.get("nofog_name", "")
        
        # Try to find scores by name first (most reliable)
        if fog_name and nofog_name:
            fog_s = scores_by_name.get(fog_name)
            nofog_s = scores_by_name.get(nofog_name)
            
            if fog_s is None or nofog_s is None:
                # Fallback to indices if names don't match (shouldn't happen, but defensive)
                print(f"[Warning] Could not find scores by name for pair {i+1}, using indices")
                if fog_idx < len(scores) and nofog_idx < len(scores):
                    fog_s = scores[fog_idx]
                    nofog_s = scores[nofog_idx]
                else:
                    print(f"[Error] Invalid indices for pair {i+1}: fog_idx={fog_idx}, nofog_idx={nofog_idx}")
                    continue
        else:
            # Fallback to indices if metadata doesn't have names
            if fog_idx < len(scores) and nofog_idx < len(scores):
                fog_s = scores[fog_idx]
                nofog_s = scores[nofog_idx]
                # Update metadata with actual names
                meta["fog_name"] = fog_s.name
                meta["nofog_name"] = nofog_s.name
            else:
                print(f"[Error] Invalid indices for pair {i+1}: fog_idx={fog_idx}, nofog_idx={nofog_idx}")
                continue
        
        delta_nofog_minus_fog = nofog_s.Q_norm - fog_s.Q_norm
        delta_fog_minus_nofog = fog_s.Q_norm - nofog_s.Q_norm

        # Extract just the date ID from mesh names for cleaner output
        # Pattern: {participant}_{nofog_session}__{fog_session}_{condition}
        # For fog: extract fog_session (second part after "__")
        fog_base = fog_s.name[:-4] if fog_s.name.endswith("_fog") else fog_s.name
        fog_parts = fog_base.split("__")
        fog_session = fog_parts[1] if len(fog_parts) == 2 else fog_s.name

        # For nofog: extract nofog_session (last 15 chars of first part before "__")
        nofog_base = nofog_s.name[:-6] if nofog_s.name.endswith("_nofog") else nofog_s.name
        nofog_parts = nofog_base.split("__")
        if len(nofog_parts) == 2:
            nofog_part = nofog_parts[0]  # e.g., "Kilian Kozerke_20251212_191346"
            # Extract just the date ID (last 15 characters: YYYYMMDD_HHMMSS)
            nofog_session = nofog_part[-15:] if len(nofog_part) >= 15 and "_" in nofog_part[-15:] else nofog_s.name
        else:
            nofog_session = nofog_s.name

        rows.append([
            meta.get("participant", ""),
            meta.get("pair_id", ""),
            fog_session,
            f"{fog_s.Q_norm:.6f}",
            nofog_session,
            f"{nofog_s.Q_norm:.6f}",
            f"{delta_nofog_minus_fog:.6f}",
            f"{delta_fog_minus_nofog:.6f}",
        ])
        labels.append(meta.get("participant") or meta.get("pair_id") or f"pair{i+1}")
        fog_vals.append(fog_s.Q_norm)
        nofog_vals.append(nofog_s.Q_norm)
        deltas.append(delta_nofog_minus_fog)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["participant", "pair_id", "fog_name", "fog_Q_norm", "nofog_name", "nofog_Q_norm", "delta_nofog_minus_fog", "delta_fog_minus_nofog"])
        writer.writerows(rows)
    print(f"[Info] Wrote pairwise summary CSV: {summary_csv}")

    pngs: Dict[str, str] = {}

    # Bar chart per pair
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, fog_vals, width=0.4, label="Fog")
    ax.bar(x + 0.2, nofog_vals, width=0.4, label="NoFog")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Q_norm")
    ax.set_title("Quality scores per pair (normalized)")
    ax.legend()
    pngs["Per-pair scores"] = _fig_to_base64(fig)

    # Delta plot
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["green" if d >= 0 else "red" for d in deltas]
    ax.bar(x, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Delta (NoFog - Fog)")
    ax.set_title("Score delta per pair (Q_norm)")
    pngs["Score delta"] = _fig_to_base64(fig)

    # Box plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([fog_vals, nofog_vals])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Fog", "NoFog"])
    ax.set_ylabel("Q_norm")
    ax.set_title("Score distribution")
    pngs["Distribution"] = _fig_to_base64(fig)

    html_parts = [
        "<html><head><title>Fog vs NoFog Mesh Quality</title>",
        "<style>table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:6px;}th{background:#eee;}</style>",
        "</head><body>",
        "<h2>Fog vs NoFog Mesh Quality (normalized scores)</h2>",
        "<table>",
        "<tr><th>Participant</th><th>Pair ID</th><th>Fog</th><th>Fog Q_norm</th><th>NoFog</th><th>NoFog Q_norm</th><th>Delta (NoFog-Fog)</th><th>Delta (Fog-NoFog)</th></tr>",
    ]
    for r in rows:
        html_parts.append(
            f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td><td>{r[7]}</td></tr>"
        )
    html_parts.append("</table><br/>")
    for title, b64 in pngs.items():
        html_parts.append(f"<h3>{title}</h3><img src='data:image/png;base64,{b64}' style='max-width:100%;'/>")
    html_parts.append("</body></html>")

    html_path = out_dir / "pairwise_quality_report.html"
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"[Info] Wrote pairwise HTML report: {html_path}")


# -----------------------------------------------------------------------------
# Progressive master CSV updates (fog / no-fog report)
# -----------------------------------------------------------------------------


def update_master_fog_report(
    master_csv: Path,
    scores: List[QualityScores],
    pair_meta: List[Dict[str, str]],
) -> None:
    """
    Progressively write per-mesh and relative metrics back into
    master_fog_no_fog_report.csv.

    This function is intended to be called repeatedly as more meshes are
    evaluated. It will:
      - Fill *_evaluate_quality_score_placeholder with Q_norm
      - Add comprehensive quality metrics for both fog and nofog:
        - Raw scores (Q_raw) and normalized scores (Q_norm)
        - All sub-scores: S_shape, S_topology, S_bonuses, S_geom, S_smooth, S_complete, S_color
      - Maintain / create a relative delta column: relative_quality_delta_nofog_minus_fog
    Only rows for which both fog and no-fog meshes have been evaluated are
    updated; other rows are left untouched.
    """
    if not master_csv.exists():
        # Nothing to update (defensive; should exist in --from-csv mode)
        return

    # Fast lookup for scores and pair metadata
    scores_by_name: Dict[str, QualityScores] = {s.name: s for s in scores}
    meta_index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for m in pair_meta:
        participant = (m.get("participant") or "").strip()
        pair_id = (m.get("pair_id") or "").strip()
        if participant and pair_id:
            meta_index[(participant, pair_id)] = m

    # Read entire CSV into memory, update rows in-place, then rewrite.
    with master_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # Define all the quality metric columns we want to add
    quality_columns = [
        "Q_raw", "Q_norm",  # Overall scores
        "S_geom", "S_smooth", "S_complete", "S_color",  # Main sub-scores
        "S_shape", "S_topology", "S_bonuses"  # Detailed sub-scores
    ]

    # Add columns for both fog and nofog conditions
    new_columns = []
    for condition in ["fog", "nofog"]:
        for col in quality_columns:
            new_col = f"{condition}_{col}"
            if new_col not in fieldnames:
                fieldnames.append(new_col)
                new_columns.append(new_col)

    delta_col = "relative_quality_delta_nofog_minus_fog"
    if delta_col not in fieldnames:
        fieldnames.append(delta_col)
        new_columns.append(delta_col)

    for row in rows:
        participant = (row.get("participant") or "").strip()
        pair_id = (row.get("pair_id") or "").strip()
        meta = meta_index.get((participant, pair_id))
        if not meta:
            continue

        fog_name = meta.get("fog_name") or ""
        nofog_name = meta.get("nofog_name") or ""
        fog_score = scores_by_name.get(fog_name)
        nofog_score = scores_by_name.get(nofog_name)

        # Only update if we have both sides evaluated so far
        if fog_score is None or nofog_score is None:
            continue

        fog_q_norm = fog_score.Q_norm
        nofog_q_norm = nofog_score.Q_norm
        delta = nofog_q_norm - fog_q_norm

        # Fill existing placeholder columns with the normalized scores (backward compatibility)
        if "fog_evaluate_quality_score_placeholder" in row:
            row["fog_evaluate_quality_score_placeholder"] = f"{fog_q_norm:.6f}"
        if "nofog_evaluate_quality_score_placeholder" in row:
            row["nofog_evaluate_quality_score_placeholder"] = f"{nofog_q_norm:.6f}"

        # Fill comprehensive quality metrics for fog condition
        row["fog_Q_raw"] = f"{fog_score.Q_raw:.6f}"
        row["fog_Q_norm"] = f"{fog_score.Q_norm:.6f}"
        row["fog_S_geom"] = f"{fog_score.S_geom:.6f}"
        row["fog_S_smooth"] = f"{fog_score.S_smooth:.6f}"
        row["fog_S_complete"] = f"{fog_score.S_complete:.6f}"
        row["fog_S_color"] = f"{fog_score.S_color:.6f}"
        row["fog_S_shape"] = f"{fog_score.S_shape:.6f}"
        row["fog_S_topology"] = f"{fog_score.S_topology:.6f}"
        row["fog_S_bonuses"] = f"{fog_score.S_bonuses:.6f}"

        # Fill comprehensive quality metrics for nofog condition
        row["nofog_Q_raw"] = f"{nofog_score.Q_raw:.6f}"
        row["nofog_Q_norm"] = f"{nofog_score.Q_norm:.6f}"
        row["nofog_S_geom"] = f"{nofog_score.S_geom:.6f}"
        row["nofog_S_smooth"] = f"{nofog_score.S_smooth:.6f}"
        row["nofog_S_complete"] = f"{nofog_score.S_complete:.6f}"
        row["nofog_S_color"] = f"{nofog_score.S_color:.6f}"
        row["nofog_S_shape"] = f"{nofog_score.S_shape:.6f}"
        row["nofog_S_topology"] = f"{nofog_score.S_topology:.6f}"
        row["nofog_S_bonuses"] = f"{nofog_score.S_bonuses:.6f}"

        # Optionally fill report path placeholders with the batch CSV path
        # (kept simple: point to the global quality_scores.csv in analysis dir)
        if "fog_evaluate_report_path_placeholder" in row:
            row["fog_evaluate_report_path_placeholder"] = "analysis/mesh_quality_batch/quality_scores.csv"
        if "nofog_evaluate_report_path_placeholder" in row:
            row["nofog_evaluate_report_path_placeholder"] = "analysis/mesh_quality_batch/quality_scores.csv"

        row[delta_col] = f"{delta:.6f}"

    with master_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def load_pairs_from_csv(csv_path: Path) -> List[Tuple[Path, Path, str, str]]:
    """
    Load fog/no-fog pairs from master_fog_no_fog_report.csv.
    
    Returns list of tuples: (fog_path, nofog_path, participant, session_id_pair)
    Only includes pairs where both meshes exist (color_mesh_present == True).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    pairs: List[Tuple[Path, Path, str, str]] = []

    def _mesh_present(value: Any) -> bool:
        """Accept bools and truthy strings."""
        if isinstance(value, bool):
            return value
        normalized = str(value or "").strip().lower()
        return normalized in ("true", "1", "yes", "y")

    def _extract_prefixed_mesh(row: Dict[str, Any], prefix: str) -> Optional[Path]:
        for key in (f"{prefix}_color_mesh_fbx_path", f"{prefix}_color_mesh_ply_path"):
            candidate = (row.get(key) or "").strip()
            if candidate:
                return Path(candidate)
        return None

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    has_symmetric_schema = any(
        "nofog_color_mesh_fbx_path" in r or "fog_color_mesh_fbx_path" in r for r in rows
    )

    if has_symmetric_schema:
        for row in rows:
            participant = (row.get("participant") or "").strip()
            fog_path = _extract_prefixed_mesh(row, "fog")
            nofog_path = _extract_prefixed_mesh(row, "nofog")

            fog_present = _mesh_present(row.get("fog_color_mesh_present", True))
            nofog_present = _mesh_present(row.get("nofog_color_mesh_present", True))

            if not participant or fog_path is None or nofog_path is None:
                continue

            if not fog_present or not fog_path.exists():
                print(f"[Warning] Skipping participant '{participant}' (fog mesh missing): {fog_path}")
                continue
            if not nofog_present or not nofog_path.exists():
                print(f"[Warning] Skipping participant '{participant}' (nofog mesh missing): {nofog_path}")
                continue

            pair_id = (
                row.get("pair_id")
                or f"{row.get('nofog_session_id', '')}_{row.get('fog_session_id', '')}".strip("_")
                or f"{participant}_pair"
            )
            pairs.append((fog_path, nofog_path, participant, pair_id))
    else:
        # Legacy long/stacked schema (one row per condition)
        session_map: Dict[str, Dict[str, Any]] = {}
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row.get("session_id", "").strip()
                if not session_id:
                    continue
                
                if not _mesh_present(row.get("color_mesh_present", "")):
                    continue
                
                mesh_path_str = row.get("color_mesh_fbx_path", "").strip()
                if not mesh_path_str:
                    continue
                
                mesh_path = Path(mesh_path_str)
                if not mesh_path.exists():
                    print(f"[Warning] Mesh path from CSV does not exist: {mesh_path}")
                    continue
                
                condition = row.get("condition", "").strip()
                paired_session_id = row.get("paired_session_id", "").strip()
                participant = row.get("participant", "").strip()
                
                session_map[session_id] = {
                    "condition": condition,
                    "paired_session_id": paired_session_id,
                    "mesh_path": mesh_path,
                    "participant": participant,
                }
        
        processed_pairs = set()
        for session_id, session_data in session_map.items():
            paired_id = session_data["paired_session_id"]
            
            if session_id in processed_pairs or paired_id in processed_pairs:
                continue
            
            if paired_id not in session_map:
                continue
            
            paired_data = session_map[paired_id]
            if paired_data["paired_session_id"] != session_id:
                continue
            
            cond1 = session_data["condition"]
            cond2 = paired_data["condition"]
            
            if cond1 == "Fog" and cond2 == "NoFog":
                fog_path = session_data["mesh_path"]
                nofog_path = paired_data["mesh_path"]
                participant = session_data["participant"]
                pair_id = f"{session_id}_{paired_id}"
            elif cond1 == "NoFog" and cond2 == "Fog":
                nofog_path = session_data["mesh_path"]
                fog_path = paired_data["mesh_path"]
                participant = session_data["participant"]
                pair_id = f"{paired_id}_{session_id}"
            else:
                print(f"[Warning] Skipping pair {session_id}/{paired_id}: unexpected conditions ({cond1}/{cond2})")
                continue
            
            pairs.append((fog_path, nofog_path, participant, pair_id))
            processed_pairs.add(session_id)
            processed_pairs.add(paired_id)
    
    return pairs


def _process_single_mesh_worker(mesh_data: Tuple[Path, str]) -> RawMeshMetrics:
    """
    Worker function for processing a single mesh in parallel.
    Must be at module level for multiprocessing pickling.
    """
    path, name = mesh_data
    print(f"[Info] Processing mesh: {name} ({path})")
    try:
        raw = compute_raw_metrics_for_mesh(path, name)
        print(f"[Info] Completed processing: {name}")
        return raw
    except Exception as e:
        print(f"[Error] Failed to process mesh {name}: {e}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate intrinsic quality of FBX/PLY meshes (geometry, smoothness, completeness, color).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Evaluate a set of meshes independently
  python evaluate_fbx_quality.py scan1.fbx scan2.fbx

  # Evaluate 13 fog/no-fog pairs
  python evaluate_fbx_quality.py \\
      --pair fog_scan1.fbx nofog_scan1.fbx \\
      --pair fog_scan2.fbx nofog_scan2.fbx \\
      ... (repeat 13 times) ...

  # Automatically load pairs from master_fog_no_fog_report.csv
  python evaluate_fbx_quality.py --from-csv analysis/master_fog_no_fog_report.csv

  # Save results to a specific CSV path
  python evaluate_fbx_quality.py scan1.fbx scan2.fbx --csv results/quality_report.csv
        """,
    )

    parser.add_argument(
        "meshes",
        nargs="*",
        type=Path,
        help="Unpaired mesh paths (FBX/PLY). Ignored in pair mode if --pair or --from-csv is provided.",
    )

    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("FOG", "NOFOG"),
        help="Fog and no-fog mesh paths for a pair (can be used multiple times).",
    )

    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Load fog/no-fog pairs automatically from master_fog_no_fog_report.csv. "
             "Only pairs where both meshes exist (color_mesh_present=True) will be included.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write detailed CSV with all metrics and scores.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/mesh_quality_batch"),
        help="Output directory for batch artifacts (plots, pairwise summary).",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"Number of parallel workers for mesh processing. "
             f"Defaults to CPU count ({multiprocessing.cpu_count()}). Use 1 for sequential processing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mesh_paths: List[Tuple[Path, str]] = []
    pair_indices: List[Tuple[int, int]] = []  # indices into mesh_paths referencing (fog, nofog)
    pair_meta: List[Dict[str, str]] = []      # metadata per pair: participant, pair_id

    # CSV-based pair loading mode
    if args.from_csv:
        print(f"[Info] Loading pairs from CSV: {args.from_csv}")
        csv_pairs = load_pairs_from_csv(args.from_csv)
        
        if not csv_pairs:
            raise SystemExit(f"No valid pairs found in CSV file: {args.from_csv}")
        
        print(f"[Info] Found {len(csv_pairs)} valid pairs in CSV")
        
        for idx, (fog_path, nofog_path, participant, pair_id) in enumerate(csv_pairs, start=1):
            if not fog_path.exists():
                raise FileNotFoundError(f"Fog mesh not found: {fog_path}")
            if not nofog_path.exists():
                raise FileNotFoundError(f"No-fog mesh not found: {nofog_path}")

            # Use participant and pair_id for naming
            fog_name = f"{participant}_{pair_id}_fog"
            nofog_name = f"{participant}_{pair_id}_nofog"

            fog_index = len(mesh_paths)
            mesh_paths.append((fog_path, fog_name))
            nofog_index = len(mesh_paths)
            mesh_paths.append((nofog_path, nofog_name))
            pair_indices.append((fog_index, nofog_index))
            pair_meta.append({
                "participant": participant,
                "pair_id": pair_id,
                "fog_name": fog_name,
                "nofog_name": nofog_name,
            })

    # Manual pair mode
    if args.pair:
        for idx, (fog_str, nofog_str) in enumerate(args.pair, start=1):
            fog_path = Path(fog_str)
            nofog_path = Path(nofog_str)
            if not fog_path.exists():
                raise FileNotFoundError(f"Fog mesh not found: {fog_path}")
            if not nofog_path.exists():
                raise FileNotFoundError(f"No-fog mesh not found: {nofog_path}")

            fog_name = f"pair{idx}_fog"
            nofog_name = f"pair{idx}_nofog"

            fog_index = len(mesh_paths)
            mesh_paths.append((fog_path, fog_name))
            nofog_index = len(mesh_paths)
            mesh_paths.append((nofog_path, nofog_name))
            pair_indices.append((fog_index, nofog_index))
            pair_meta.append({
                "participant": f"pair{idx}",
                "pair_id": f"pair{idx}",
                "fog_name": fog_name,
                "nofog_name": nofog_name,
            })

    # Unpaired meshes
    for p in args.meshes:
        if not p.exists():
            raise FileNotFoundError(f"Mesh not found: {p}")
        name = p.stem
        mesh_paths.append((p, name))

    if not mesh_paths:
        raise SystemExit("No meshes provided. Use positional arguments, --pair, or --from-csv.")

    # Determine number of workers
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    elif num_workers <= 0:
        num_workers = 1

    print(f"[Info] Using {num_workers} worker(s) for parallel mesh processing")

    # Compute raw metrics for all meshes (parallel processing)
    raw_list: List[RawMeshMetrics] = []

    if num_workers == 1:
        # Sequential processing (original behavior)
        print("[Info] Using sequential processing")
        for idx, (path, name) in enumerate(mesh_paths, start=1):
            print(f"[Info] Computing raw metrics for {name} ({path})")
            raw = compute_raw_metrics_for_mesh(path, name)
            raw_list.append(raw)

            # Progressive batch scoring and CSV updates to keep
            # master_fog_no_fog_report.csv in sync as we go.
            scores_so_far = compute_quality_scores(raw_list)
            # Only meaningful when we are in pair / from-csv modes
            if args.from_csv is not None and pair_meta:
                try:
                    update_master_fog_report(args.from_csv, scores_so_far, pair_meta)
                except Exception as exc:
                    print(f"[Warning] Failed to update master fog report after mesh {idx}: {exc}")
    else:
        # Parallel processing
        print(f"[Info] Processing {len(mesh_paths)} meshes in parallel using {num_workers} workers")
        # Create name-to-index mapping for maintaining order
        name_to_index: Dict[str, int] = {name: idx for idx, (_, name) in enumerate(mesh_paths)}
        # Store results by name to maintain correct pairing
        raw_by_name: Dict[str, RawMeshMetrics] = {}
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_mesh = {
                executor.submit(_process_single_mesh_worker, mesh_data): mesh_data
                for mesh_data in mesh_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_mesh):
                mesh_data = future_to_mesh[future]
                try:
                    raw = future.result()
                    raw_by_name[raw.name] = raw
                    print(f"[Info] Completed processing: {raw.name}")

                    # Reconstruct raw_list in correct order for progressive scoring
                    raw_list_ordered = [raw_by_name[name] for _, name in mesh_paths if name in raw_by_name]
                    
                    # Progressive batch scoring and CSV updates
                    scores_so_far = compute_quality_scores(raw_list_ordered)
                    # Only meaningful when we are in pair / from-csv modes
                    if args.from_csv is not None and pair_meta:
                        try:
                            update_master_fog_report(args.from_csv, scores_so_far, pair_meta)
                        except Exception as exc:
                            print(f"[Warning] Failed to update master fog report: {exc}")

                    # Force garbage collection to prevent memory buildup
                    gc.collect()

                except Exception as exc:
                    path, name = mesh_data
                    print(f"[Error] Mesh {name} generated an exception: {exc}")
                    raise

        # Reconstruct raw_list in the correct order matching mesh_paths
        raw_list = [raw_by_name[name] for _, name in mesh_paths]
        print(f"[Info] All {len(mesh_paths)} meshes processed in parallel")

        # Encourage Python to release memory between large meshes
        gc.collect()

    # Compute batch-normalized quality scores
    scores = compute_quality_scores(raw_list)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV output
    if args.csv is not None:
        write_csv(scores, args.csv)
        print(f"[Info] Wrote CSV report to: {args.csv}")
    else:
        default_csv = out_dir / "quality_scores.csv"
        write_csv(scores, default_csv)
        print(f"[Info] Wrote CSV report to: {default_csv}")

    # Print overall ranking
    print_batch_summary(scores)

    # Print pair-wise summaries when in pair mode
    if pair_indices:
        # Create name-to-score mapping for reliable lookup
        scores_by_name: Dict[str, QualityScores] = {s.name: s for s in scores}
        
        pairs: List[Tuple[QualityScores, QualityScores]] = []
        for i, (fog_idx, nofog_idx) in enumerate(pair_indices):
            meta = pair_meta[i] if i < len(pair_meta) else None
            
            # Use names from metadata for reliable matching
            if meta:
                fog_name = meta.get("fog_name", "")
                nofog_name = meta.get("nofog_name", "")
                fog_s = scores_by_name.get(fog_name) if fog_name else None
                nofog_s = scores_by_name.get(nofog_name) if nofog_name else None
                
                if fog_s is None or nofog_s is None:
                    # Fallback to indices
                    if fog_idx < len(scores) and nofog_idx < len(scores):
                        fog_s = scores[fog_idx]
                        nofog_s = scores[nofog_idx]
                    else:
                        print(f"[Error] Could not find scores for pair {i+1}")
                        continue
            else:
                # Fallback to indices if no metadata
                if fog_idx < len(scores) and nofog_idx < len(scores):
                    fog_s = scores[fog_idx]
                    nofog_s = scores[nofog_idx]
                else:
                    print(f"[Error] Invalid indices for pair {i+1}")
                    continue
            
            pairs.append((fog_s, nofog_s))
        print_pair_summaries(pairs)
        write_pairwise_reports(scores, pair_indices, pair_meta, out_dir)


if __name__ == "__main__":
    main()


