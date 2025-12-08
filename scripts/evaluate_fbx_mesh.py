#!/usr/bin/env python3
"""
FBX Mesh Quality Evaluation Script

Evaluates colored FBX mesh files using no-reference metrics and displays
a comprehensive validation report in a GUI popup.

Metrics computed:
- Geometric quality: topology, triangle quality, mesh statistics
- Color quality: coverage, distribution, smoothness
- File quality: size, compression
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import open3d as o3d
from dataclasses import dataclass
import json
import time

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("[Warning] tkinter not available. Report will be printed to console only.")

# Try to import aspose.threed for FBX support (same library used for conversion)
try:
    import aspose.threed as a3d
    ASPOSE_AVAILABLE = True
except ImportError:
    ASPOSE_AVAILABLE = False
    print("[Warning] aspose-3d not available. FBX loading may fail. Install with: pip install aspose-3d")


@dataclass
class GeometricMetrics:
    """Geometric quality metrics for the mesh."""
    vertex_count: int
    face_count: int
    edge_count: int
    is_watertight: bool
    is_manifold: bool
    is_orientable: bool
    num_connected_components: int
    num_non_manifold_edges: int
    num_non_manifold_vertices: int
    num_self_intersecting_triangles: int
    num_boundary_edges: int
    num_holes: int  # Number of holes (boundary loops)
    genus: int  # Number of holes/tunnels
    surface_area: float
    volume: float
    bounding_box: Tuple[float, float, float]  # (width, height, depth)
    center_of_mass: Tuple[float, float, float]
    avg_edge_length: float
    std_edge_length: float
    min_edge_length: float
    max_edge_length: float
    avg_triangle_area: float
    std_triangle_area: float
    min_triangle_area: float
    max_triangle_area: float
    avg_aspect_ratio: float
    min_aspect_ratio: float
    degenerate_triangles: int


@dataclass
class ColorMetrics:
    """Color quality metrics for the mesh."""
    has_vertex_colors: bool
    vertex_color_coverage: float  # Percentage of vertices with colors
    mean_rgb: Tuple[float, float, float]
    std_rgb: Tuple[float, float, float]
    min_rgb: Tuple[float, float, float]
    max_rgb: Tuple[float, float, float]
    color_variance: float
    avg_color_gradient_magnitude: float  # Color smoothness
    color_consistency: float  # Neighboring vertex color similarity


@dataclass
class SmoothnessMetrics:
    """Smoothness and noise quality metrics for the mesh."""
    avg_normal_deviation: float  # Average deviation of vertex normals from neighbors
    std_normal_deviation: float  # Standard deviation of normal deviations
    avg_curvature_variation: float  # Average curvature variation
    std_curvature_variation: float  # Standard deviation of curvature variation
    surface_roughness: float  # Estimated surface roughness (neighbor position deviation)
    normal_consistency: float  # Average cosine similarity of neighboring normals


@dataclass
class FileMetrics:
    """File quality metrics."""
    file_size_mb: float
    vertices_per_mb: float
    faces_per_mb: float
    overall_density: float  # Combined density metric


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    file_path: Path
    geometric: GeometricMetrics
    color: ColorMetrics
    smoothness: SmoothnessMetrics
    file: FileMetrics
    warnings: List[str]
    errors: List[str]
    quality_score: float = 0.0  # Overall quality score (0-1)


def compute_triangle_aspect_ratio(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Compute aspect ratio for each triangle (vectorized for performance).
    Aspect ratio = (4 * sqrt(3) * area) / (sum of squared edge lengths)
    For equilateral triangle, aspect ratio = 1.0
    For degenerate triangle, aspect ratio approaches 0.0
    """
    num_triangles = len(triangles)
    aspect_ratios = np.zeros(num_triangles)
    
    # Process in chunks to show progress for large meshes
    chunk_size = max(100000, num_triangles // 10)  # Process in chunks, show ~10 progress updates
    
    for chunk_start in range(0, num_triangles, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_triangles)
        chunk_triangles = triangles[chunk_start:chunk_end]
        
        # Vectorized computation for chunk
        v0 = vertices[chunk_triangles[:, 0]]
        v1 = vertices[chunk_triangles[:, 1]]
        v2 = vertices[chunk_triangles[:, 2]]
        
        # Edge lengths (vectorized)
        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)
        
        # Triangle area using cross product (more stable than Heron's)
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        
        # Sum of squared edge lengths
        edge_sum_sq = e0**2 + e1**2 + e2**2
        
        # Compute aspect ratios
        valid_mask = edge_sum_sq > 1e-10
        aspect_ratios[chunk_start:chunk_end] = np.where(
            valid_mask,
            (4.0 * np.sqrt(3.0) * areas) / edge_sum_sq,
            0.0
        )
        
        if chunk_end % (chunk_size * 5) == 0 or chunk_end == num_triangles:
            print(f"[Progress]   Processed {chunk_end:,} / {num_triangles:,} triangles ({100*chunk_end/num_triangles:.1f}%)")
    
    return aspect_ratios


def compute_color_gradient(mesh: o3d.geometry.TriangleMesh) -> float:
    """
    Compute average color gradient magnitude across edges (vectorized).
    Measures color smoothness - lower values indicate smoother colors.
    """
    if not mesh.has_vertex_colors():
        return 0.0
    
    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    
    # Vectorized computation: get all edges from triangles
    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])
    
    # Compute color differences for all edges (vectorized)
    color_diffs = colors[edges[:, 0]] - colors[edges[:, 1]]
    gradients = np.linalg.norm(color_diffs, axis=1)
    
    return np.mean(gradients) if len(gradients) > 0 else 0.0


def compute_color_consistency(mesh: o3d.geometry.TriangleMesh) -> float:
    """
    Compute color consistency as average similarity between neighboring vertices (vectorized).
    Higher values indicate more consistent colors.
    """
    if not mesh.has_vertex_colors():
        return 0.0
    
    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    
    # Get all edges from triangles (vectorized)
    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])
    
    # Get colors for edge endpoints
    c0 = colors[edges[:, 0]]
    c1 = colors[edges[:, 1]]
    
    # Compute cosine similarity (vectorized)
    dot_products = np.sum(c0 * c1, axis=1)
    norms0 = np.linalg.norm(c0, axis=1)
    norms1 = np.linalg.norm(c1, axis=1)
    
    # Avoid division by zero
    valid_mask = (norms0 > 1e-10) & (norms1 > 1e-10)
    similarities = np.where(
        valid_mask,
        dot_products / (norms0 * norms1),
        0.0
    )
    
    return float(np.mean(similarities[valid_mask])) if np.any(valid_mask) else 0.0


def compute_geometric_metrics(mesh: o3d.geometry.TriangleMesh, skip_detailed_non_manifold: bool = False) -> GeometricMetrics:
    """Compute all geometric quality metrics."""
    print("[Progress] Starting geometric metrics computation...")
    start_time = time.time()
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    vertex_count = len(vertices)
    face_count = len(triangles)
    print(f"[Progress] Mesh has {vertex_count:,} vertices and {face_count:,} faces")
    
    # Basic topology checks
    print("[Progress] Computing topology checks (watertight, manifold, orientable)...")
    topo_start = time.time()
    is_watertight = mesh.is_watertight()
    # Open3D 0.19.0 uses is_edge_manifold() and is_vertex_manifold() instead of is_manifold()
    is_edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    is_vertex_manifold = mesh.is_vertex_manifold()
    # A mesh is manifold if both edge and vertex are manifold
    is_manifold = is_edge_manifold and is_vertex_manifold
    # Check orientability (may not be available in all versions)
    try:
        is_orientable = mesh.is_orientable()
    except AttributeError:
        # If not available, assume orientable if edge-manifold
        is_orientable = is_edge_manifold
    print(f"[Progress] Topology checks completed in {time.time() - topo_start:.2f}s")
    
    # Non-manifold elements
    # Optimization: If mesh is already manifold, skip detailed detection (no non-manifold elements exist)
    # Also skip if explicitly requested (for very large meshes)
    if skip_detailed_non_manifold or (is_edge_manifold and is_vertex_manifold):
        if skip_detailed_non_manifold:
            print("[Progress] Skipping detailed non-manifold detection (requested or mesh is large)...")
        else:
            print("[Progress] Skipping detailed non-manifold detection (mesh is already manifold)...")
        num_non_manifold_edges = 0
        num_non_manifold_vertices = 0
        num_self_intersecting_triangles = 0
    else:
        print("[Progress] Finding non-manifold elements (this may take a while for large meshes)...")
        nm_start = time.time()
        
        # Compute sequentially with progress reporting
        # Note: Open3D's internal operations are already optimized, but these can still be slow for very large meshes
        print("[Progress]   Computing non-manifold edges...")
        non_manifold_edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        num_non_manifold_edges = len(non_manifold_edges)
        print(f"[Progress]   Found {num_non_manifold_edges:,} non-manifold edges")
        
        print("[Progress]   Computing non-manifold vertices...")
        non_manifold_vertices = mesh.get_non_manifold_vertices()
        num_non_manifold_vertices = len(non_manifold_vertices)
        print(f"[Progress]   Found {num_non_manifold_vertices:,} non-manifold vertices")
        
        print("[Progress]   Checking for self-intersecting triangles...")
        try:
            self_intersecting = mesh.get_self_intersecting_triangles()
        except AttributeError:
            self_intersecting = []
        num_self_intersecting_triangles = len(self_intersecting)
        print(f"[Progress]   Found {num_self_intersecting_triangles:,} self-intersecting triangles")
        
        print(f"[Progress] Non-manifold analysis completed in {time.time() - nm_start:.2f}s")
    
    # Connected components
    print("[Progress] Computing connected components...")
    cc_start = time.time()
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    num_connected_components = len(cluster_n_triangles)
    print(f"[Progress] Found {num_connected_components} connected components in {time.time() - cc_start:.2f}s")
    
    # Boundary edges - optimized: use edge_to_faces only for boundary detection
    print("[Progress] Computing boundary edges and holes...")
    boundary_start = time.time()
    # Build edge_to_faces map efficiently using numpy
    # Create all edges from triangles
    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])
    # Sort edges to make them canonical
    edges_sorted = np.sort(edges, axis=1)
    # Count occurrences of each edge
    unique_edges, edge_counts = np.unique(edges_sorted, axis=0, return_counts=True)
    # Boundary edges are those that appear only once
    boundary_mask = edge_counts == 1
    num_boundary_edges = np.sum(boundary_mask)
    edge_count = len(unique_edges)
    
    # Count holes (boundary loops)
    # A hole is a closed boundary loop
    num_holes = 0
    if num_boundary_edges > 0:
        # Get boundary edges
        boundary_edge_list = unique_edges[boundary_mask]
        
        # Build edge adjacency for boundary edges only
        edge_to_vertices = {}
        for edge in boundary_edge_list:
            v0, v1 = edge[0], edge[1]
            if v0 not in edge_to_vertices:
                edge_to_vertices[v0] = []
            if v1 not in edge_to_vertices:
                edge_to_vertices[v1] = []
            edge_to_vertices[v0].append(v1)
            edge_to_vertices[v1].append(v0)
        
        # Count closed loops (holes)
        # A hole is a closed boundary loop (cycle)
        visited_edges = set()
        num_holes = 0
        
        for start_vertex in edge_to_vertices.keys():
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
            loop_length = 0
            
            while True:
                edge_key = tuple(sorted([current, next_vertex]))
                if edge_key in visited_edges:
                    break
                
                visited_edges.add(edge_key)
                loop_length += 1
                
                # Move to next vertex
                current = next_vertex
                
                # Check if we've closed the loop
                if current == start_vertex and loop_length > 2:
                    num_holes += 1
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
                if loop_length > num_boundary_edges:
                    break
    
    print(f"[Progress] Boundary edge analysis completed in {time.time() - boundary_start:.2f}s ({num_boundary_edges:,} boundary edges, {num_holes} holes)")
    
    # Genus (Euler characteristic)
    # χ = V - E + F = 2 - 2g - b (for closed surfaces)
    # g = (2 - χ - b) / 2
    if is_watertight:
        euler_char = vertex_count - edge_count + face_count
        genus = max(0, (2 - euler_char) // 2)
    else:
        genus = -1  # Not applicable for non-watertight meshes
    
    # Surface area and volume
    print("[Progress] Computing surface area and volume...")
    area_start = time.time()
    surface_area = mesh.get_surface_area()
    volume = mesh.get_volume() if is_watertight else 0.0
    print(f"[Progress] Surface area/volume computed in {time.time() - area_start:.2f}s")
    
    # Bounding box
    print("[Progress] Computing bounding box...")
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_extent()
    bbox_center = bbox.get_center()
    
    # Edge lengths - vectorized computation
    print("[Progress] Computing edge length statistics...")
    edge_start = time.time()
    # Use the edges we already computed
    edge_vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    avg_edge_length = np.mean(edge_lengths)
    std_edge_length = np.std(edge_lengths)
    min_edge_length = np.min(edge_lengths)
    max_edge_length = np.max(edge_lengths)
    print(f"[Progress] Edge length statistics completed in {time.time() - edge_start:.2f}s")
    
    # Triangle areas - vectorized computation
    print("[Progress] Computing triangle area statistics...")
    area_start = time.time()
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    # Vectorized cross product and norm
    triangle_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    avg_triangle_area = np.mean(triangle_areas)
    std_triangle_area = np.std(triangle_areas)
    min_triangle_area = np.min(triangle_areas)
    max_triangle_area = np.max(triangle_areas)
    print(f"[Progress] Triangle area statistics completed in {time.time() - area_start:.2f}s")
    
    # Triangle aspect ratios - vectorized where possible
    print("[Progress] Computing triangle aspect ratios (this may take a while for large meshes)...")
    aspect_start = time.time()
    aspect_ratios = compute_triangle_aspect_ratio(vertices, triangles)
    avg_aspect_ratio = np.mean(aspect_ratios)
    min_aspect_ratio = np.min(aspect_ratios)
    degenerate_threshold = 0.01
    degenerate_triangles = np.sum(aspect_ratios < degenerate_threshold)
    print(f"[Progress] Aspect ratio computation completed in {time.time() - aspect_start:.2f}s")
    
    total_time = time.time() - start_time
    print(f"[Progress] Geometric metrics computation completed in {total_time:.2f}s total")
    
    return GeometricMetrics(
        vertex_count=vertex_count,
        face_count=face_count,
        edge_count=edge_count,
        is_watertight=is_watertight,
        is_manifold=is_manifold,
        is_orientable=is_orientable,
        num_connected_components=num_connected_components,
        num_non_manifold_edges=num_non_manifold_edges,
        num_non_manifold_vertices=num_non_manifold_vertices,
        num_self_intersecting_triangles=num_self_intersecting_triangles,
        num_boundary_edges=num_boundary_edges,
        num_holes=num_holes,
        genus=genus,
        surface_area=surface_area,
        volume=volume,
        bounding_box=tuple(bbox_size),
        center_of_mass=tuple(bbox_center),
        avg_edge_length=float(avg_edge_length),
        std_edge_length=float(std_edge_length),
        min_edge_length=float(min_edge_length),
        max_edge_length=float(max_edge_length),
        avg_triangle_area=float(avg_triangle_area),
        std_triangle_area=float(std_triangle_area),
        min_triangle_area=float(min_triangle_area),
        max_triangle_area=float(max_triangle_area),
        avg_aspect_ratio=float(avg_aspect_ratio),
        min_aspect_ratio=float(min_aspect_ratio),
        degenerate_triangles=int(degenerate_triangles)
    )


def compute_color_metrics(mesh: o3d.geometry.TriangleMesh) -> ColorMetrics:
    """Compute all color quality metrics."""
    print("[Progress] Starting color metrics computation...")
    color_start = time.time()
    
    has_vertex_colors = mesh.has_vertex_colors()
    
    if not has_vertex_colors:
        print(f"[Progress] Color metrics computation completed in {time.time() - color_start:.2f}s")
        return ColorMetrics(
            has_vertex_colors=False,
            vertex_color_coverage=0.0,
            mean_rgb=(0.0, 0.0, 0.0),
            std_rgb=(0.0, 0.0, 0.0),
            min_rgb=(0.0, 0.0, 0.0),
            max_rgb=(0.0, 0.0, 0.0),
            color_variance=0.0,
            avg_color_gradient_magnitude=0.0,
            color_consistency=0.0
        )
    
    # Process vertex colors
    print("[Progress] Processing vertex colors...")
    vertex_colors = np.asarray(mesh.vertex_colors)
    vertex_color_coverage = 100.0  # All vertices have colors
    
    # Color statistics
    print("[Progress] Computing color statistics...")
    mean_rgb = tuple(np.mean(vertex_colors, axis=0))
    std_rgb = tuple(np.std(vertex_colors, axis=0))
    min_rgb = tuple(np.min(vertex_colors, axis=0))
    max_rgb = tuple(np.max(vertex_colors, axis=0))
    
    # Overall color variance
    color_variance = np.var(vertex_colors)
    
    # Color smoothness and consistency
    print("[Progress] Computing color smoothness (gradient)...")
    avg_color_gradient = compute_color_gradient(mesh)
    print("[Progress] Computing color consistency...")
    color_consistency = compute_color_consistency(mesh)
    
    print(f"[Progress] Color metrics computation completed in {time.time() - color_start:.2f}s")
    
    return ColorMetrics(
        has_vertex_colors=has_vertex_colors,
        vertex_color_coverage=vertex_color_coverage,
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        min_rgb=min_rgb,
        max_rgb=max_rgb,
        color_variance=float(color_variance),
        avg_color_gradient_magnitude=avg_color_gradient,
        color_consistency=color_consistency
    )


def compute_smoothness_metrics(mesh: o3d.geometry.TriangleMesh) -> SmoothnessMetrics:
    """Compute smoothness and noise quality metrics."""
    print("[Progress] Starting smoothness metrics computation...")
    smooth_start = time.time()
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Compute vertex normals if not present
    if not mesh.has_vertex_normals():
        print("[Progress] Computing vertex normals...")
        mesh.compute_vertex_normals()
    
    normals = np.asarray(mesh.vertex_normals)
    
    # Build vertex-to-triangle adjacency
    print("[Progress] Building vertex adjacency...")
    vertex_to_triangles = {}
    for i, tri in enumerate(triangles):
        for v_idx in tri:
            if v_idx not in vertex_to_triangles:
                vertex_to_triangles[v_idx] = []
            vertex_to_triangles[v_idx].append(i)
    
    # Compute normal deviations (deviation from neighbors)
    print("[Progress] Computing normal deviations...")
    normal_deviations = []
    normal_similarities = []
    
    # Sample vertices for performance (use all for small meshes, sample for large)
    num_vertices = len(vertices)
    sample_size = min(50000, num_vertices) if num_vertices > 50000 else num_vertices
    sample_indices = np.linspace(0, num_vertices - 1, sample_size, dtype=np.int32) if num_vertices > 50000 else np.arange(num_vertices)
    
    for v_idx in sample_indices:
        if v_idx not in vertex_to_triangles:
            continue
        
        # Get neighboring vertices through shared triangles
        neighbor_indices = set()
        for tri_idx in vertex_to_triangles[v_idx]:
            tri = triangles[tri_idx]
            for n_idx in tri:
                if n_idx != v_idx:
                    neighbor_indices.add(n_idx)
        
        if len(neighbor_indices) == 0:
            continue
        
        neighbor_indices = list(neighbor_indices)
        v_normal = normals[v_idx]
        neighbor_normals = normals[neighbor_indices]
        
        # Compute deviations (angle between normals)
        dot_products = np.dot(neighbor_normals, v_normal)
        # Clamp to avoid numerical issues
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products)
        deviations = np.abs(angles)
        
        normal_deviations.extend(deviations.tolist())
        
        # Compute consistency (cosine similarity)
        similarities = dot_products
        normal_similarities.extend(similarities.tolist())
    
    avg_normal_deviation = np.mean(normal_deviations) if normal_deviations else 0.0
    std_normal_deviation = np.std(normal_deviations) if normal_deviations else 0.0
    normal_consistency = np.mean(normal_similarities) if normal_similarities else 0.0
    
    # Compute curvature variation (using triangle area variation as proxy)
    print("[Progress] Computing curvature variation...")
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    # Compute local curvature variation per vertex
    curvature_variations = []
    for v_idx in sample_indices[:min(10000, len(sample_indices))]:  # Limit for performance
        if v_idx not in vertex_to_triangles:
            continue
        
        # Get areas of adjacent triangles
        adjacent_areas = []
        for tri_idx in vertex_to_triangles[v_idx]:
            adjacent_areas.append(triangle_areas[tri_idx])
        
        if len(adjacent_areas) > 1:
            variation = np.std(adjacent_areas) / (np.mean(adjacent_areas) + 1e-10)
            curvature_variations.append(variation)
    
    avg_curvature_variation = np.mean(curvature_variations) if curvature_variations else 0.0
    std_curvature_variation = np.std(curvature_variations) if curvature_variations else 0.0
    
    # Compute surface roughness (neighbor position deviation)
    print("[Progress] Computing surface roughness...")
    position_deviations = []
    for v_idx in sample_indices[:min(10000, len(sample_indices))]:
        if v_idx not in vertex_to_triangles:
            continue
        
        neighbor_indices = set()
        for tri_idx in vertex_to_triangles[v_idx]:
            tri = triangles[tri_idx]
            for n_idx in tri:
                if n_idx != v_idx:
                    neighbor_indices.add(n_idx)
        
        if len(neighbor_indices) == 0:
            continue
        
        neighbor_indices = list(neighbor_indices)
        v_pos = vertices[v_idx]
        neighbor_positions = vertices[neighbor_indices]
        
        # Compute distances to neighbors
        distances = np.linalg.norm(neighbor_positions - v_pos, axis=1)
        avg_distance = np.mean(distances)
        
        # Deviation from average distance indicates roughness
        deviations = np.abs(distances - avg_distance)
        position_deviations.extend(deviations.tolist())
    
    surface_roughness = np.mean(position_deviations) if position_deviations else 0.0
    
    print(f"[Progress] Smoothness metrics computation completed in {time.time() - smooth_start:.2f}s")
    
    return SmoothnessMetrics(
        avg_normal_deviation=float(avg_normal_deviation),
        std_normal_deviation=float(std_normal_deviation),
        avg_curvature_variation=float(avg_curvature_variation),
        std_curvature_variation=float(std_curvature_variation),
        surface_roughness=float(surface_roughness),
        normal_consistency=float(normal_consistency)
    )


def compute_file_metrics(file_path: Path, geometric: GeometricMetrics) -> FileMetrics:
    """Compute file quality metrics."""
    file_size_bytes = file_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    vertices_per_mb = geometric.vertex_count / file_size_mb if file_size_mb > 0 else 0.0
    faces_per_mb = geometric.face_count / file_size_mb if file_size_mb > 0 else 0.0
    
    # Overall density: geometric mean of vertices and faces per MB
    overall_density = np.sqrt(vertices_per_mb * faces_per_mb) if vertices_per_mb > 0 and faces_per_mb > 0 else 0.0
    
    return FileMetrics(
        file_size_mb=file_size_mb,
        vertices_per_mb=vertices_per_mb,
        faces_per_mb=faces_per_mb,
        overall_density=overall_density
    )


def evaluate_fbx_mesh(mesh_path: Path, skip_detailed_non_manifold: bool = False) -> EvaluationReport:
    """
    Load mesh (FBX/PLY) and compute all evaluation metrics.
    
    Args:
        mesh_path: Path to the mesh file (FBX or PLY)
        skip_detailed_non_manifold: Skip detailed non-manifold detection for performance
        
    Returns:
        EvaluationReport with all computed metrics
    """
    warnings = []
    errors = []
    
    if not mesh_path.exists():
        errors.append(f"File not found: {mesh_path}")
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    file_ext = mesh_path.suffix.lower()
    if file_ext not in ['.fbx', '.ply']:
        warnings.append(f"File extension is not .fbx or .ply: {file_ext}")
    
    # Load mesh - try different methods based on file format
    mesh = None
    
    if file_ext == '.fbx':
        # FBX files - use Aspose.3D to convert to PLY temporarily, then load with Open3D
        # This is the most reliable method since Aspose.3D handles FBX well and Open3D handles PLY well
        if ASPOSE_AVAILABLE:
            try:
                print(f"[Info] Loading FBX file using Aspose.3D (converting to PLY temporarily)...")
                import tempfile
                
                # Load FBX scene with Aspose
                scene = a3d.Scene.from_file(str(mesh_path))
                
                # Save to temporary PLY file
                with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
                    tmp_ply_path = tmp_file.name
                
                try:
                    # Save scene as PLY (preserves vertex colors)
                    scene.save(tmp_ply_path, a3d.FileFormat.PLY)  # type: ignore
                    
                    # Load PLY with Open3D (which has excellent PLY support including vertex colors)
                    mesh = o3d.io.read_triangle_mesh(tmp_ply_path)
                    
                    # Clean up temp file
                    Path(tmp_ply_path).unlink()
                    
                    if len(mesh.vertices) == 0:
                        raise ValueError("Loaded mesh has no vertices")
                    
                    print(f"[Info] Successfully loaded FBX mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
                    if mesh.has_vertex_colors():
                        print(f"[Info] Vertex colors detected: {len(mesh.vertex_colors)} colors")
                    
                except Exception as e:
                    # Clean up temp file on error
                    if Path(tmp_ply_path).exists():
                        Path(tmp_ply_path).unlink()
                    raise ValueError(f"Failed to convert/load FBX via PLY: {e}")
                
            except Exception as e:
                errors.append(f"Failed to load FBX with Aspose.3D: {e}")
                # Fall through to try other methods
                print(f"[Warning] Aspose.3D loading failed: {e}. Trying alternative methods...")
        
        # Final fallback: Try Open3D directly
        if mesh is None:
            try:
                print(f"[Info] Trying to load FBX file using Open3D...")
                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                if len(mesh.vertices) == 0:
                    raise ValueError("Open3D loaded empty mesh - FBX support may be limited")
                print(f"[Info] Successfully loaded FBX mesh with Open3D: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
            except Exception as e:
                raise ValueError(f"Failed to load FBX file with all methods. "
                               f"Please install aspose-3d: pip install aspose-3d")
    elif file_ext == '.ply':
        # PLY files - use Open3D directly
        try:
            print(f"[Info] Loading PLY file using Open3D...")
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if len(mesh.vertices) == 0:
                raise ValueError("Loaded mesh has no vertices")
            print(f"[Info] Successfully loaded PLY mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
            if mesh.has_vertex_colors():
                print(f"[Info] Vertex colors detected: {len(mesh.vertex_colors)} colors")
        except Exception as e:
            errors.append(f"Failed to load PLY file: {e}")
            raise
    else:
        # Try Open3D for other formats (OBJ, etc.)
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        except Exception as e:
            errors.append(f"Failed to load mesh: {e}")
            raise
    
    if len(mesh.vertices) == 0:
        errors.append("Mesh has no vertices")
        raise ValueError("Empty mesh")
    
    if len(mesh.triangles) == 0:
        errors.append("Mesh has no triangles")
        raise ValueError("Mesh has no faces")
    
    # Compute metrics
    print("[Info] Starting metric computation...")
    metric_start = time.time()
    
    # Auto-skip detailed non-manifold detection for very large meshes (>1M faces)
    # This can save significant time
    face_count = len(mesh.triangles)
    auto_skip = face_count > 1_000_000
    if auto_skip:
        print(f"[Info] Large mesh detected ({face_count:,} faces). Skipping detailed non-manifold detection for performance.")
        skip_detailed_non_manifold = True
    
    geometric = compute_geometric_metrics(mesh, skip_detailed_non_manifold=skip_detailed_non_manifold)
    color = compute_color_metrics(mesh)
    smoothness = compute_smoothness_metrics(mesh)
    file = compute_file_metrics(mesh_path, geometric)
    
    print(f"[Info] All metrics computed in {time.time() - metric_start:.2f}s")
    
    # Generate warnings based on metrics
    if not geometric.is_watertight:
        warnings.append("Mesh is not watertight (has holes/boundaries)")
    
    if not geometric.is_manifold:
        warnings.append("Mesh is not manifold (has non-manifold edges/vertices)")
        if skip_detailed_non_manifold or auto_skip:
            warnings.append("(Detailed non-manifold element counts skipped for performance)")
    
    if geometric.num_non_manifold_edges > 0:
        warnings.append(f"Mesh has {geometric.num_non_manifold_edges:,} non-manifold edges")
    
    if geometric.num_non_manifold_vertices > 0:
        warnings.append(f"Mesh has {geometric.num_non_manifold_vertices:,} non-manifold vertices")
    
    if geometric.num_self_intersecting_triangles > 0:
        warnings.append(f"Mesh has {geometric.num_self_intersecting_triangles} self-intersecting triangles")
    
    if geometric.num_connected_components > 1:
        warnings.append(f"Mesh consists of {geometric.num_connected_components} disconnected components")
    
    if geometric.degenerate_triangles > 0:
        warnings.append(f"Mesh has {geometric.degenerate_triangles} degenerate triangles (aspect ratio < 0.01)")
    
    if not color.has_vertex_colors:
        warnings.append("Mesh has no vertex colors")
    
    if geometric.avg_aspect_ratio < 0.3:
        warnings.append(f"Low average triangle aspect ratio ({geometric.avg_aspect_ratio:.3f}), indicates poor triangle quality")
    
    return EvaluationReport(
        file_path=mesh_path,
        geometric=geometric,
        color=color,
        smoothness=smoothness,
        file=file,
        warnings=warnings,
        errors=errors,
        quality_score=0.0  # Will be computed during comparison
    )


def format_report(report: EvaluationReport) -> str:
    """Format evaluation report as a readable string."""
    lines = []
    lines.append("=" * 80)
    lines.append("FBX MESH QUALITY EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append(f"\nFile: {report.file_path}")
    lines.append(f"File Size: {report.file.file_size_mb:.2f} MB")
    lines.append("")
    
    # Geometric Metrics
    lines.append("-" * 80)
    lines.append("GEOMETRIC QUALITY METRICS")
    lines.append("-" * 80)
    lines.append(f"Vertices: {report.geometric.vertex_count:,}")
    lines.append(f"Faces: {report.geometric.face_count:,}")
    lines.append(f"Edges: {report.geometric.edge_count:,}")
    lines.append("")
    
    lines.append("Topology:")
    lines.append(f"  Watertight: {'✓' if report.geometric.is_watertight else '✗'}")
    lines.append(f"  Manifold: {'✓' if report.geometric.is_manifold else '✗'}")
    lines.append(f"  Orientable: {'✓' if report.geometric.is_orientable else '✗'}")
    lines.append(f"  Connected Components: {report.geometric.num_connected_components}")
    if report.geometric.genus >= 0:
        lines.append(f"  Genus (holes/tunnels): {report.geometric.genus}")
    lines.append(f"  Boundary Edges: {report.geometric.num_boundary_edges}")
    lines.append(f"  Holes: {report.geometric.num_holes}")
    lines.append(f"  Non-manifold Edges: {report.geometric.num_non_manifold_edges}")
    lines.append(f"  Non-manifold Vertices: {report.geometric.num_non_manifold_vertices}")
    lines.append(f"  Self-intersecting Triangles: {report.geometric.num_self_intersecting_triangles}")
    lines.append("")
    
    lines.append("Geometry:")
    lines.append(f"  Surface Area: {report.geometric.surface_area:.6f}")
    if report.geometric.volume > 0:
        lines.append(f"  Volume: {report.geometric.volume:.6f}")
    lines.append(f"  Bounding Box: {report.geometric.bounding_box[0]:.3f} × {report.geometric.bounding_box[1]:.3f} × {report.geometric.bounding_box[2]:.3f}")
    lines.append(f"  Center of Mass: ({report.geometric.center_of_mass[0]:.3f}, {report.geometric.center_of_mass[1]:.3f}, {report.geometric.center_of_mass[2]:.3f})")
    lines.append("")
    
    lines.append("Edge Lengths:")
    lines.append(f"  Average: {report.geometric.avg_edge_length:.6f}")
    lines.append(f"  Std Dev: {report.geometric.std_edge_length:.6f}")
    lines.append(f"  Range: [{report.geometric.min_edge_length:.6f}, {report.geometric.max_edge_length:.6f}]")
    lines.append("")
    
    lines.append("Triangle Quality:")
    lines.append(f"  Average Area: {report.geometric.avg_triangle_area:.6f}")
    lines.append(f"  Area Std Dev: {report.geometric.std_triangle_area:.6f}")
    lines.append(f"  Area Range: [{report.geometric.min_triangle_area:.6f}, {report.geometric.max_triangle_area:.6f}]")
    lines.append(f"  Average Aspect Ratio: {report.geometric.avg_aspect_ratio:.3f} (1.0 = equilateral)")
    lines.append(f"  Min Aspect Ratio: {report.geometric.min_aspect_ratio:.3f}")
    lines.append(f"  Degenerate Triangles: {report.geometric.degenerate_triangles}")
    lines.append("")
    
    # Smoothness Metrics
    lines.append("-" * 80)
    lines.append("SMOOTHNESS & NOISE METRICS")
    lines.append("-" * 80)
    lines.append("Normal Consistency:")
    lines.append(f"  Average Normal Deviation: {report.smoothness.avg_normal_deviation:.6f} radians (lower = smoother)")
    lines.append(f"  Std Dev of Normal Deviation: {report.smoothness.std_normal_deviation:.6f}")
    lines.append(f"  Normal Consistency: {report.smoothness.normal_consistency:.3f} (higher = more consistent)")
    lines.append("")
    lines.append("Curvature Variation:")
    lines.append(f"  Average: {report.smoothness.avg_curvature_variation:.6f}")
    lines.append(f"  Std Dev: {report.smoothness.std_curvature_variation:.6f}")
    lines.append("")
    lines.append("Surface Roughness:")
    lines.append(f"  Estimated Roughness: {report.smoothness.surface_roughness:.6f} (lower = smoother)")
    lines.append("")
    
    # Color Metrics
    lines.append("-" * 80)
    lines.append("COLOR QUALITY METRICS")
    lines.append("-" * 80)
    lines.append(f"Has Vertex Colors: {'✓' if report.color.has_vertex_colors else '✗'}")
    lines.append(f"Vertex Color Coverage: {report.color.vertex_color_coverage:.1f}%")
    lines.append("")
    
    if report.color.has_vertex_colors:
        lines.append("Color Statistics (RGB):")
        lines.append(f"  Mean: ({report.color.mean_rgb[0]:.3f}, {report.color.mean_rgb[1]:.3f}, {report.color.mean_rgb[2]:.3f})")
        lines.append(f"  Std Dev: ({report.color.std_rgb[0]:.3f}, {report.color.std_rgb[1]:.3f}, {report.color.std_rgb[2]:.3f})")
        lines.append(f"  Min: ({report.color.min_rgb[0]:.3f}, {report.color.min_rgb[1]:.3f}, {report.color.min_rgb[2]:.3f})")
        lines.append(f"  Max: ({report.color.max_rgb[0]:.3f}, {report.color.max_rgb[1]:.3f}, {report.color.max_rgb[2]:.3f})")
        lines.append(f"  Overall Variance: {report.color.color_variance:.6f}")
        lines.append("")
        lines.append("Color Quality:")
        lines.append(f"  Average Gradient Magnitude: {report.color.avg_color_gradient_magnitude:.6f} (lower = smoother)")
        lines.append(f"  Color Consistency: {report.color.color_consistency:.3f} (higher = more consistent)")
        lines.append("")
    
    # File Metrics
    lines.append("-" * 80)
    lines.append("FILE QUALITY METRICS")
    lines.append("-" * 80)
    lines.append(f"Vertices per MB: {report.file.vertices_per_mb:.1f}")
    lines.append(f"Faces per MB: {report.file.faces_per_mb:.1f}")
    lines.append(f"Overall Density: {report.file.overall_density:.1f}")
    lines.append("")
    
    # Quality Score (if computed)
    if report.quality_score > 0:
        lines.append("-" * 80)
        lines.append("OVERALL QUALITY SCORE")
        lines.append("-" * 80)
        lines.append(f"Quality Score: {report.quality_score:.4f} / 1.0000")
        lines.append("")
    
    # Warnings and Errors
    if report.warnings:
        lines.append("-" * 80)
        lines.append("WARNINGS")
        lines.append("-" * 80)
        for warning in report.warnings:
            lines.append(f"  ⚠ {warning}")
        lines.append("")
    
    if report.errors:
        lines.append("-" * 80)
        lines.append("ERRORS")
        lines.append("-" * 80)
        for error in report.errors:
            lines.append(f"  ✗ {error}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


class EvaluationReportGUI:
    """GUI window for displaying evaluation report."""
    
    def __init__(self, report: EvaluationReport):
        self.report = report
        self.root = tk.Tk()
        self.root.title("FBX Mesh Quality Evaluation Report")
        self.root.geometry("900x700")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header
        header_label = ttk.Label(
            main_frame,
            text=f"Evaluation Report: {report.file_path.name}",
            font=("Arial", 14, "bold")
        )
        header_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Text area with scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        text_area = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Courier", 10),
            width=100,
            height=35
        )
        text_area.grid(row=0, column=0, sticky="nsew")
        
        # Insert report text
        report_text = format_report(report)
        text_area.insert(tk.END, report_text)
        text_area.config(state=tk.DISABLED)  # Make read-only
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, sticky=tk.E, pady=(10, 0))
        
        # Save button
        save_button = ttk.Button(
            button_frame,
            text="Save Report",
            command=self.save_report
        )
        save_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Close button
        close_button = ttk.Button(
            button_frame,
            text="Close",
            command=self.root.destroy
        )
        close_button.pack(side=tk.LEFT)
        
        # Status indicator
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        
        if report.errors:
            status_label = ttk.Label(
                status_frame,
                text="Status: ERRORS FOUND",
                foreground="red",
                font=("Arial", 10, "bold")
            )
        elif report.warnings:
            status_label = ttk.Label(
                status_frame,
                text="Status: WARNINGS FOUND",
                foreground="orange",
                font=("Arial", 10, "bold")
            )
        else:
            status_label = ttk.Label(
                status_frame,
                text="Status: NO ISSUES DETECTED",
                foreground="green",
                font=("Arial", 10, "bold")
            )
        status_label.pack(side=tk.LEFT)
    
    def save_report(self):
        """Save report to file."""
        from tkinter import filedialog
        
        default_path = self.report.file_path.with_suffix('.eval_report.txt')
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=default_path.name
        )
        
        if file_path:
            try:
                report_text = format_report(self.report)
                Path(file_path).write_text(report_text, encoding="utf-8")
                messagebox.showinfo("Success", f"Report saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report:\n{e}")
    
    def show(self):
        """Display the GUI window."""
        self.root.mainloop()


def compute_quality_score(report: EvaluationReport, all_reports: List[EvaluationReport] | None = None) -> float:
    """
    Compute overall quality score (0-1) for a mesh.
    Higher score = better quality.
    
    Weights:
    - Geometry: 0.6
    - Smoothness: 0.2
    - Color: 0.1
    - Completeness/File: 0.1
    """
    if all_reports is None:
        all_reports = [report]
    
    # Normalize metrics across all meshes
    def normalize(value, min_val, max_val, higher_better=True):
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return normalized if higher_better else 1.0 - normalized
    
    # Extract metric arrays for normalization
    aspect_ratios = [r.geometric.avg_aspect_ratio for r in all_reports]
    degenerate_counts = [r.geometric.degenerate_triangles for r in all_reports]
    non_manifold_edges = [r.geometric.num_non_manifold_edges for r in all_reports]
    non_manifold_vertices = [r.geometric.num_non_manifold_vertices for r in all_reports]
    self_intersecting = [r.geometric.num_self_intersecting_triangles for r in all_reports]
    boundary_edges = [r.geometric.num_boundary_edges for r in all_reports]
    connected_components = [r.geometric.num_connected_components for r in all_reports]
    normal_deviations = [r.smoothness.avg_normal_deviation for r in all_reports]
    surface_roughness = [r.smoothness.surface_roughness for r in all_reports]
    normal_consistency = [r.smoothness.normal_consistency for r in all_reports]
    color_coverage = [r.color.vertex_color_coverage for r in all_reports]
    color_consistency = [r.color.color_consistency for r in all_reports if r.color.has_vertex_colors]
    overall_density = [r.file.overall_density for r in all_reports]
    
    # Geometry score (0.6 weight)
    geom_score = 0.0
    
    # Aspect ratio (higher is better, max 1.0)
    geom_score += 0.15 * normalize(
        report.geometric.avg_aspect_ratio,
        min(aspect_ratios), max(aspect_ratios),
        higher_better=True
    )
    
    # Degenerate triangles (lower is better)
    geom_score += 0.10 * normalize(
        report.geometric.degenerate_triangles,
        min(degenerate_counts), max(degenerate_counts),
        higher_better=False
    )
    
    # Non-manifold elements (lower is better)
    geom_score += 0.10 * normalize(
        report.geometric.num_non_manifold_edges + report.geometric.num_non_manifold_vertices,
        min(non_manifold_edges) + min(non_manifold_vertices),
        max(non_manifold_edges) + max(non_manifold_vertices),
        higher_better=False
    )
    
    # Self-intersecting triangles (lower is better)
    geom_score += 0.05 * normalize(
        report.geometric.num_self_intersecting_triangles,
        min(self_intersecting), max(self_intersecting),
        higher_better=False
    )
    
    # Boundary edges (lower is better for watertight meshes)
    geom_score += 0.05 * normalize(
        report.geometric.num_boundary_edges,
        min(boundary_edges), max(boundary_edges),
        higher_better=False
    )
    
    # Connected components (lower is better, ideally 1)
    geom_score += 0.05 * normalize(
        report.geometric.num_connected_components,
        min(connected_components), max(connected_components),
        higher_better=False
    )
    
    # Manifold and watertight bonuses
    geom_score += 0.10 * (1.0 if report.geometric.is_manifold else 0.0)
    geom_score += 0.10 * (1.0 if report.geometric.is_watertight else 0.0)
    
    # Smoothness score (0.2 weight)
    smooth_score = 0.0
    if len(normal_deviations) > 0:
        smooth_score += 0.10 * normalize(
            report.smoothness.avg_normal_deviation,
            min(normal_deviations), max(normal_deviations),
            higher_better=False
        )
    
    if len(surface_roughness) > 0:
        smooth_score += 0.05 * normalize(
            report.smoothness.surface_roughness,
            min(surface_roughness), max(surface_roughness),
            higher_better=False
        )
    
    if len(normal_consistency) > 0:
        smooth_score += 0.05 * normalize(
            report.smoothness.normal_consistency,
            min(normal_consistency), max(normal_consistency),
            higher_better=True
        )
    
    # Color score (0.1 weight)
    color_score = 0.0
    if report.color.has_vertex_colors:
        if len(color_coverage) > 0:
            color_score += 0.05 * normalize(
                report.color.vertex_color_coverage,
                min(color_coverage), max(color_coverage),
                higher_better=True
            )
        
        if len(color_consistency) > 0:
            color_score += 0.05 * normalize(
                report.color.color_consistency,
                min(color_consistency), max(color_consistency),
                higher_better=True
            )
    else:
        # No penalty if no colors (mesh might not need them)
        color_score = 0.05
    
    # Completeness/File score (0.1 weight)
    file_score = 0.0
    if len(overall_density) > 0 and max(overall_density) > 0:
        file_score += 0.10 * normalize(
            report.file.overall_density,
            min(overall_density), max(overall_density),
            higher_better=True
        )
    
    # Combine scores
    total_score = (
        0.6 * geom_score +
        0.2 * smooth_score +
        0.1 * color_score +
        0.1 * file_score
    )
    
    return max(0.0, min(1.0, total_score))  # Clamp to [0, 1]


def compare_meshes(reports: List[EvaluationReport], output_format: str = "text") -> str:
    """
    Compare multiple meshes and generate a ranking report.
    
    Args:
        reports: List of evaluation reports
        output_format: "text", "json", "markdown", or "html"
    
    Returns:
        Formatted comparison report string
    """
    # Compute quality scores for all meshes
    for report in reports:
        report.quality_score = compute_quality_score(report, reports)
    
    # Sort by quality score (descending)
    sorted_reports = sorted(reports, key=lambda r: r.quality_score, reverse=True)
    
    if output_format == "json":
        return format_comparison_json(sorted_reports)
    elif output_format == "markdown":
        return format_comparison_markdown(sorted_reports)
    elif output_format == "html":
        return format_comparison_html(sorted_reports)
    else:
        return format_comparison_text(sorted_reports)


def format_comparison_text(reports: List[EvaluationReport]) -> str:
    """Format comparison report as plain text."""
    lines = []
    lines.append("=" * 100)
    lines.append("MESH QUALITY COMPARISON REPORT")
    lines.append("=" * 100)
    lines.append(f"\nComparing {len(reports)} meshes\n")
    
    # Ranking table
    lines.append("-" * 100)
    lines.append("RANKING (sorted by quality score)")
    lines.append("-" * 100)
    lines.append(f"{'Rank':<6} {'Score':<8} {'File':<50} {'Vertices':<12} {'Faces':<12}")
    lines.append("-" * 100)
    
    for rank, report in enumerate(reports, 1):
        file_name = report.file_path.name[:47] + "..." if len(report.file_path.name) > 50 else report.file_path.name
        lines.append(
            f"{rank:<6} {report.quality_score:<8.4f} {file_name:<50} "
            f"{report.geometric.vertex_count:<12,} {report.geometric.face_count:<12,}"
        )
    
    # Detailed comparison table
    lines.append("\n" + "=" * 100)
    lines.append("DETAILED METRIC COMPARISON")
    lines.append("=" * 100)
    
    # Key metrics table
    lines.append("\nGeometric Quality Metrics:")
    lines.append("-" * 100)
    lines.append(f"{'Mesh':<30} {'Aspect':<8} {'Degenerate':<12} {'Non-Manifold':<15} {'Boundary':<10} {'Components':<12}")
    lines.append("-" * 100)
    for report in reports:
        name = report.file_path.name[:28] + ".." if len(report.file_path.name) > 30 else report.file_path.name
        lines.append(
            f"{name:<30} {report.geometric.avg_aspect_ratio:<8.3f} "
            f"{report.geometric.degenerate_triangles:<12} "
            f"{report.geometric.num_non_manifold_edges + report.geometric.num_non_manifold_vertices:<15} "
            f"{report.geometric.num_boundary_edges:<10} "
            f"{report.geometric.num_connected_components:<12}"
        )
    
    lines.append("\nSmoothness Metrics:")
    lines.append("-" * 100)
    lines.append(f"{'Mesh':<30} {'Normal Dev':<12} {'Roughness':<12} {'Normal Consist':<15}")
    lines.append("-" * 100)
    for report in reports:
        name = report.file_path.name[:28] + ".." if len(report.file_path.name) > 30 else report.file_path.name
        lines.append(
            f"{name:<30} {report.smoothness.avg_normal_deviation:<12.6f} "
            f"{report.smoothness.surface_roughness:<12.6f} "
            f"{report.smoothness.normal_consistency:<15.3f}"
        )
    
    lines.append("\nColor Metrics:")
    lines.append("-" * 100)
    lines.append(f"{'Mesh':<30} {'Has Colors':<12} {'Coverage':<12} {'Consistency':<12}")
    lines.append("-" * 100)
    for report in reports:
        name = report.file_path.name[:28] + ".." if len(report.file_path.name) > 30 else report.file_path.name
        has_colors = "Yes" if report.color.has_vertex_colors else "No"
        coverage = f"{report.color.vertex_color_coverage:.1f}%" if report.color.has_vertex_colors else "N/A"
        consistency = f"{report.color.color_consistency:.3f}" if report.color.has_vertex_colors else "N/A"
        lines.append(f"{name:<30} {has_colors:<12} {coverage:<12} {consistency:<12}")
    
    # Justification
    lines.append("\n" + "=" * 100)
    lines.append("RANKING JUSTIFICATION")
    lines.append("=" * 100)
    for rank, report in enumerate(reports[:3], 1):  # Top 3
        lines.append(f"\nRank {rank}: {report.file_path.name}")
        lines.append(f"  Quality Score: {report.quality_score:.4f}")
        
        strengths = []
        if report.geometric.is_manifold and report.geometric.is_watertight:
            strengths.append("manifold and watertight")
        if report.geometric.avg_aspect_ratio > 0.5:
            strengths.append("good triangle quality")
        if report.geometric.num_connected_components == 1:
            strengths.append("single connected component")
        if report.smoothness.normal_consistency > 0.8:
            strengths.append("smooth surface")
        if report.color.has_vertex_colors and report.color.color_consistency > 0.7:
            strengths.append("consistent colors")
        
        if strengths:
            lines.append(f"  Strengths: {', '.join(strengths)}")
        
        weaknesses = []
        if not report.geometric.is_manifold:
            weaknesses.append("non-manifold topology")
        if report.geometric.degenerate_triangles > 100:
            weaknesses.append(f"many degenerate triangles ({report.geometric.degenerate_triangles})")
        if report.geometric.num_boundary_edges > 1000:
            weaknesses.append("many boundary edges")
        if report.smoothness.surface_roughness > 0.01:
            weaknesses.append("high surface roughness")
        
        if weaknesses:
            lines.append(f"  Weaknesses: {', '.join(weaknesses)}")
    
    lines.append("\n" + "=" * 100)
    return "\n".join(lines)


def format_comparison_json(reports: List[EvaluationReport]) -> str:
    """Format comparison report as JSON."""
    data = {
        "comparison": {
            "num_meshes": len(reports),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meshes": []
        }
    }
    
    for rank, report in enumerate(reports, 1):
        mesh_data = {
            "rank": rank,
            "file_path": str(report.file_path),
            "file_name": report.file_path.name,
            "quality_score": report.quality_score,
            "geometric": {
                "vertex_count": report.geometric.vertex_count,
                "face_count": report.geometric.face_count,
                "avg_aspect_ratio": report.geometric.avg_aspect_ratio,
                "degenerate_triangles": report.geometric.degenerate_triangles,
                "is_manifold": report.geometric.is_manifold,
                "is_watertight": report.geometric.is_watertight,
                "num_non_manifold_edges": report.geometric.num_non_manifold_edges,
                "num_non_manifold_vertices": report.geometric.num_non_manifold_vertices,
                "num_boundary_edges": report.geometric.num_boundary_edges,
                "num_holes": report.geometric.num_holes,
                "num_connected_components": report.geometric.num_connected_components,
            },
            "smoothness": {
                "avg_normal_deviation": report.smoothness.avg_normal_deviation,
                "surface_roughness": report.smoothness.surface_roughness,
                "normal_consistency": report.smoothness.normal_consistency,
            },
            "color": {
                "has_vertex_colors": report.color.has_vertex_colors,
                "vertex_color_coverage": report.color.vertex_color_coverage,
                "color_consistency": report.color.color_consistency,
            },
            "file": {
                "file_size_mb": report.file.file_size_mb,
                "overall_density": report.file.overall_density,
            }
        }
        data["comparison"]["meshes"].append(mesh_data)
    
    return json.dumps(data, indent=2)


def format_comparison_markdown(reports: List[EvaluationReport]) -> str:
    """Format comparison report as Markdown."""
    lines = []
    lines.append("# Mesh Quality Comparison Report")
    lines.append(f"\nComparing {len(reports)} meshes\n")
    lines.append("## Ranking\n")
    lines.append("| Rank | Score | File | Vertices | Faces |")
    lines.append("|------|-------|------|----------|-------|")
    
    for rank, report in enumerate(reports, 1):
        lines.append(
            f"| {rank} | {report.quality_score:.4f} | {report.file_path.name} | "
            f"{report.geometric.vertex_count:,} | {report.geometric.face_count:,} |"
        )
    
    lines.append("\n## Detailed Metrics\n")
    # Add detailed tables...
    
    return "\n".join(lines)


def format_single_mesh_html(report: EvaluationReport) -> str:
    """Format single mesh report as interactive HTML with JavaScript."""
    # Helper function to convert numpy types to native Python types
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return tuple(convert_to_native(item) for item in obj)
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        return obj
    
    # Convert single report to JSON format for consistency
    mesh_data = {
        "file_path": str(report.file_path),
        "file_name": report.file_path.name,
        "quality_score": float(report.quality_score),
        "geometric": {
            "vertex_count": int(report.geometric.vertex_count),
            "face_count": int(report.geometric.face_count),
            "avg_aspect_ratio": float(report.geometric.avg_aspect_ratio),
            "min_aspect_ratio": float(report.geometric.min_aspect_ratio),
            "degenerate_triangles": int(report.geometric.degenerate_triangles),
            "degenerate_triangles_fraction": float(report.geometric.degenerate_triangles / max(1, report.geometric.face_count)),
            "is_manifold": bool(report.geometric.is_manifold),
            "is_watertight": bool(report.geometric.is_watertight),
            "num_non_manifold_edges": int(report.geometric.num_non_manifold_edges),
            "num_non_manifold_vertices": int(report.geometric.num_non_manifold_vertices),
            "num_boundary_edges": int(report.geometric.num_boundary_edges),
            "num_holes": int(report.geometric.num_holes),
            "num_connected_components": int(report.geometric.num_connected_components),
            "num_self_intersecting_triangles": int(report.geometric.num_self_intersecting_triangles),
            "surface_area": float(report.geometric.surface_area),
            "volume": float(report.geometric.volume),
            "bounding_box": [float(x) for x in report.geometric.bounding_box],
        },
        "smoothness": {
            "avg_normal_deviation": float(report.smoothness.avg_normal_deviation),
            "surface_roughness": float(report.smoothness.surface_roughness),
            "normal_consistency": float(report.smoothness.normal_consistency),
        },
        "color": {
            "has_vertex_colors": bool(report.color.has_vertex_colors),
            "vertex_color_coverage": float(report.color.vertex_color_coverage),
            "color_consistency": float(report.color.color_consistency),
            "mean_rgb": [float(x) for x in report.color.mean_rgb],
            "std_rgb": [float(x) for x in report.color.std_rgb],
        },
        "file": {
            "file_size_mb": float(report.file.file_size_mb),
            "overall_density": float(report.file.overall_density),
        },
        "warnings": list(report.warnings),
        "errors": list(report.errors),
    }
    
    # Load mesh geometry for 3D visualization
    # For large meshes, we'll sample or use a simplified version
    print("[Info] Loading mesh geometry for 3D visualization...")
    try:
        # Reload mesh for geometry data
        if report.file_path.suffix.lower() == '.fbx' and ASPOSE_AVAILABLE:
            import tempfile
            scene = a3d.Scene.from_file(str(report.file_path))
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
                tmp_ply_path = tmp_file.name
            scene.save(tmp_ply_path, a3d.FileFormat.PLY)  # type: ignore
            mesh = o3d.io.read_triangle_mesh(tmp_ply_path)
            Path(tmp_ply_path).unlink()
        else:
            mesh = o3d.io.read_triangle_mesh(str(report.file_path))
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        # For very large meshes, sample vertices/faces to keep JSON size manageable
        max_vertices = 50000
        max_faces = 100000
        
        if len(vertices) > max_vertices or len(faces) > max_faces:
            print(f"[Info] Mesh is large ({len(vertices):,} vertices, {len(faces):,} faces). Sampling for visualization...")
            # For large meshes, sample faces uniformly (simpler and preserves connectivity)
            if len(faces) > max_faces:
                # Sample faces uniformly
                face_indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int32)
                faces = faces[face_indices]
                # Find unique vertices used by sampled faces
                unique_vertices = np.unique(faces.flatten())
                # Create mapping from old vertex indices to new indices
                vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
                # Remap face indices
                faces = np.array([[vertex_map[v_idx] for v_idx in face] for face in faces])
                # Extract only used vertices
                vertices = vertices[unique_vertices]
                print(f"[Info] Sampled to {len(vertices):,} vertices, {len(faces):,} faces")
        
        # Get vertex colors if available
        vertex_colors = None
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            # Map to simplified mesh if we simplified
            if len(colors) != len(vertices):
                # Recompute colors for simplified mesh (approximate)
                if len(colors) > len(vertices):
                    # Sample colors
                    indices = np.linspace(0, len(colors) - 1, len(vertices), dtype=np.int32)
                    vertex_colors = colors[indices].tolist()
                else:
                    vertex_colors = colors.tolist()
            else:
                vertex_colors = colors.tolist()
        
        # Add geometry to mesh_data
        mesh_data["geometry"] = {
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "has_colors": vertex_colors is not None,
            "vertex_colors": vertex_colors if vertex_colors else None,
        }
    except Exception as e:
        print(f"[Warning] Failed to load mesh geometry for visualization: {e}")
        mesh_data["geometry"] = None
    
    json_data = json.dumps({"mesh": mesh_data}, indent=2)
    
    html = ["""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mesh Quality Evaluation Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .score-section {
            text-align: center;
            margin: 30px 0;
            padding: 30px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
        }
        
        .score-value {
            font-size: 4em;
            font-weight: 700;
            margin: 20px 0;
        }
        
        .score-badge {
            display: inline-block;
            padding: 10px 24px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 1.2em;
            margin-top: 10px;
        }
        
        .score-excellent { background: #4caf50; color: white; }
        .score-good { background: #8bc34a; color: white; }
        .score-fair { background: #ffc107; color: #333; }
        .score-poor { background: #ff9800; color: white; }
        .score-bad { background: #f44336; color: white; }
        
        .tabs {
            display: flex;
            border-bottom: 2px solid #eee;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .tab {
            padding: 12px 24px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1em;
            color: #666;
            transition: all 0.3s;
        }
        
        .tab:hover {
            color: #667eea;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            font-weight: 600;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .metric-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .metric-card h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .metric-value {
            font-size: 1.8em;
            font-weight: 600;
            color: #667eea;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .info-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .info-table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            width: 40%;
        }
        
        .info-table td {
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        
        .info-table tr:hover {
            background: #f5f7fa;
        }
        
        .warnings-section, .errors-section {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
        }
        
        .warnings-section {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }
        
        .errors-section {
            background: #f8d7da;
            border-left: 4px solid #f44336;
        }
        
        .warnings-section h3, .errors-section h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .warnings-section ul, .errors-section ul {
            list-style: none;
            padding-left: 0;
        }
        
        .warnings-section li, .errors-section li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }
        
        .warnings-section li:before {
            content: "⚠";
            position: absolute;
            left: 0;
        }
        
        .errors-section li:before {
            content: "✗";
            position: absolute;
            left: 0;
            color: #f44336;
        }
        
        .status-good { color: #4caf50; }
        .status-warning { color: #ff9800; }
        .status-error { color: #f44336; }
        
        #viewer-container {
            width: 100%;
            height: 600px;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin-top: 20px;
            position: relative;
            background: #1a1a1a;
        }
        
        .viewer-controls {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.95);
            padding: 8px 12px;
            border-radius: 6px;
            z-index: 100;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .viewer-controls button {
            margin: 0;
            padding: 6px 12px;
            border: 1px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            white-space: nowrap;
        }
        
        .viewer-controls button:hover {
            background: #667eea;
            color: white;
        }
        
        .viewer-info {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.85em;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Simple OrbitControls implementation (no external dependency)
        THREE.OrbitControls = function(camera, domElement) {
            this.camera = camera;
            this.domElement = domElement || document;
            this.target = new THREE.Vector3();
            this.enableDamping = true;
            this.dampingFactor = 0.05;
            
            let isMouseDown = false;
            let mouseX = 0, mouseY = 0;
            let spherical = new THREE.Spherical();
            let panOffset = new THREE.Vector3();
            
            // Initialize spherical coordinates
            const updateSpherical = () => {
                const offset = new THREE.Vector3().subVectors(this.camera.position, this.target);
                spherical.setFromVector3(offset);
            };
            updateSpherical();
            
            this.update = function() {
                if (this.enableDamping) {
                    spherical.theta *= (1 - this.dampingFactor);
                    spherical.phi *= (1 - this.dampingFactor);
                    panOffset.multiplyScalar(1 - this.dampingFactor);
                }
                
                spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                
                const offset = new THREE.Vector3();
                offset.setFromSpherical(spherical);
                offset.add(this.target).add(panOffset);
                this.camera.position.copy(offset);
                this.camera.lookAt(this.target);
            };
            
            const onMouseDown = (e) => {
                if (e.button === 0) { // Left mouse button
                    isMouseDown = true;
                    mouseX = e.clientX;
                    mouseY = e.clientY;
                    domElement.style.cursor = 'grabbing';
                }
            };
            
            const onMouseMove = (e) => {
                if (!isMouseDown) return;
                const deltaX = e.clientX - mouseX;
                const deltaY = e.clientY - mouseY;
                spherical.theta -= deltaX * 0.01;
                spherical.phi += deltaY * 0.01;
                mouseX = e.clientX;
                mouseY = e.clientY;
            };
            
            const onMouseUp = () => {
                isMouseDown = false;
                domElement.style.cursor = 'grab';
            };
            
            const onWheel = (e) => {
                e.preventDefault();
                spherical.radius *= (1 + e.deltaY * 0.001);
                spherical.radius = Math.max(0.1, Math.min(10000, spherical.radius));
            };
            
            const onContextMenu = (e) => {
                e.preventDefault();
            };
            
            domElement.addEventListener('mousedown', onMouseDown);
            domElement.addEventListener('mousemove', onMouseMove);
            domElement.addEventListener('mouseup', onMouseUp);
            domElement.addEventListener('wheel', onWheel);
            domElement.addEventListener('contextmenu', onContextMenu);
            domElement.style.cursor = 'grab';
        };
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Mesh Quality Evaluation</h1>
            <p id="mesh-name">Loading...</p>
        </div>
        <div class="content">
            <div id="loading" class="loading" style="text-align: center; padding: 40px; color: #666;">Loading...</div>
            <div id="content" style="display: none;">
                <div class="score-section">
                    <h2>Overall Quality Score</h2>
                    <div class="score-value" id="score-value">0.0000</div>
                    <div class="score-badge" id="score-badge">Evaluating...</div>
                </div>
                
                <div class="tabs">
                    <button class="tab active" onclick="showTab('geometric')">Geometric</button>
                    <button class="tab" onclick="showTab('smoothness')">Smoothness</button>
                    <button class="tab" onclick="showTab('color')">Color</button>
                    <button class="tab" onclick="showTab('file')">File Quality</button>
                    <button class="tab" onclick="showTab('visualization')">3D View</button>
                    <button class="tab" onclick="showTab('summary')">Summary</button>
                </div>
                
                <div id="geometric" class="tab-content active">
                    <h2>Geometric Quality Metrics</h2>
                    <div class="metric-grid" id="geometric-grid"></div>
                    <table class="info-table">
                        <tr>
                            <th>Topology</th>
                            <td id="topology-info"></td>
                        </tr>
                        <tr>
                            <th>Triangle Quality</th>
                            <td id="triangle-quality"></td>
                        </tr>
                        <tr>
                            <th>Edge Quality</th>
                            <td id="edge-quality"></td>
                        </tr>
                        <tr>
                            <th>Geometry</th>
                            <td id="geometry-info"></td>
                        </tr>
                    </table>
                </div>
                
                <div id="smoothness" class="tab-content">
                    <h2>Smoothness & Noise Metrics</h2>
                    <div class="metric-grid" id="smoothness-grid"></div>
                </div>
                
                <div id="color" class="tab-content">
                    <h2>Color Quality Metrics</h2>
                    <div class="metric-grid" id="color-grid"></div>
                </div>
                
                <div id="file" class="tab-content">
                    <h2>File Quality Metrics</h2>
                    <div class="metric-grid" id="file-grid"></div>
                </div>
                
                <div id="visualization" class="tab-content">
                    <h2>3D Mesh Visualization</h2>
                    <div id="viewer-container">
                        <div class="viewer-controls">
                            <button onclick="resetCamera()">Reset View</button>
                            <button onclick="toggleWireframe()">Toggle Wireframe</button>
                            <button onclick="toggleColors()">Toggle Colors</button>
                        </div>
                    </div>
                    <div class="viewer-info" id="viewer-info">Loading mesh...</div>
                </div>
                
                <div id="summary" class="tab-content">
                    <div id="warnings-container"></div>
                    <div id="errors-container"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const data = """ + json_data + """;
        const mesh = data.mesh;
        
        function getScoreClass(score) {
            if (score >= 0.8) return {class: 'score-excellent', label: 'Excellent'};
            if (score >= 0.6) return {class: 'score-good', label: 'Good'};
            if (score >= 0.4) return {class: 'score-fair', label: 'Fair'};
            if (score >= 0.2) return {class: 'score-poor', label: 'Poor'};
            return {class: 'score-bad', label: 'Bad'};
        }
        
        function formatNumber(num) {
            return new Intl.NumberFormat().format(num);
        }
        
        let viewerInitialized = false;
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
            
            // Initialize 3D viewer when visualization tab is shown
            if (tabName === 'visualization' && !viewerInitialized) {
                console.log('Visualization tab clicked, initializing viewer...');
                setTimeout(() => {
                    try {
                        init3DViewer();
                        viewerInitialized = true;
                    } catch (error) {
                        console.error('Failed to initialize viewer:', error);
                        const infoElement = document.getElementById('viewer-info');
                        if (infoElement) {
                            infoElement.textContent = 'Error: ' + error.message;
                        }
                    }
                }, 100); // Small delay to ensure container has dimensions
            }
        }
        
        function render() {
            // Header
            document.getElementById('mesh-name').textContent = mesh.file_name;
            
            // Score
            const scoreInfo = getScoreClass(mesh.quality_score);
            document.getElementById('score-value').textContent = mesh.quality_score.toFixed(4);
            const badge = document.getElementById('score-badge');
            badge.textContent = scoreInfo.label;
            badge.className = 'score-badge ' + scoreInfo.class;
            
            // Geometric metrics
            const geomGrid = document.getElementById('geometric-grid');
            geomGrid.innerHTML = `
                <div class="metric-card">
                    <h3>Vertices</h3>
                    <div class="metric-value">${formatNumber(mesh.geometric.vertex_count)}</div>
                </div>
                <div class="metric-card">
                    <h3>Faces</h3>
                    <div class="metric-value">${formatNumber(mesh.geometric.face_count)}</div>
                </div>
                <div class="metric-card">
                    <h3>Aspect Ratio</h3>
                    <div class="metric-value">${mesh.geometric.avg_aspect_ratio.toFixed(3)}</div>
                    <div class="metric-label">Average (1.0 = equilateral)</div>
                </div>
                <div class="metric-card">
                    <h3>Holes</h3>
                    <div class="metric-value">${mesh.geometric.num_holes}</div>
                    <div class="metric-label">Boundary loops</div>
                </div>
                <div class="metric-card">
                    <h3>Degenerate Triangles</h3>
                    <div class="metric-value">${mesh.geometric.degenerate_triangles}</div>
                    <div class="metric-label">${(mesh.geometric.degenerate_triangles_fraction * 100).toFixed(2)}% of total</div>
                </div>
                <div class="metric-card">
                    <h3>Connected Components</h3>
                    <div class="metric-value">${mesh.geometric.num_connected_components}</div>
                </div>
            `;
            
            document.getElementById('topology-info').innerHTML = `
                Manifold: <span class="${mesh.geometric.is_manifold ? 'status-good' : 'status-error'}">${mesh.geometric.is_manifold ? '✓' : '✗'}</span> | 
                Watertight: <span class="${mesh.geometric.is_watertight ? 'status-good' : 'status-warning'}">${mesh.geometric.is_watertight ? '✓' : '✗'}</span> | 
                Non-manifold edges: ${mesh.geometric.num_non_manifold_edges} | 
                Non-manifold vertices: ${mesh.geometric.num_non_manifold_vertices}
            `;
            
            document.getElementById('triangle-quality').textContent = 
                `Average: ${mesh.geometric.avg_aspect_ratio.toFixed(3)}, Min: ${mesh.geometric.min_aspect_ratio.toFixed(3)}, Degenerate: ${mesh.geometric.degenerate_triangles} (${(mesh.geometric.degenerate_triangles_fraction * 100).toFixed(2)}%)`;
            
            document.getElementById('edge-quality').textContent = 
                `Boundary edges: ${formatNumber(mesh.geometric.num_boundary_edges)}, Holes: ${mesh.geometric.num_holes}`;
            
            document.getElementById('geometry-info').innerHTML = 
                `Surface area: ${mesh.geometric.surface_area.toFixed(6)} | Volume: ${mesh.geometric.volume > 0 ? mesh.geometric.volume.toFixed(6) : 'N/A'} | ` +
                `Bounding box: ${mesh.geometric.bounding_box[0].toFixed(3)} × ${mesh.geometric.bounding_box[1].toFixed(3)} × ${mesh.geometric.bounding_box[2].toFixed(3)}`;
            
            // Smoothness metrics
            const smoothGrid = document.getElementById('smoothness-grid');
            smoothGrid.innerHTML = `
                <div class="metric-card">
                    <h3>Normal Deviation</h3>
                    <div class="metric-value">${mesh.smoothness.avg_normal_deviation.toFixed(6)}</div>
                    <div class="metric-label">radians (lower = smoother)</div>
                </div>
                <div class="metric-card">
                    <h3>Surface Roughness</h3>
                    <div class="metric-value">${mesh.smoothness.surface_roughness.toFixed(6)}</div>
                    <div class="metric-label">lower = smoother</div>
                </div>
                <div class="metric-card">
                    <h3>Normal Consistency</h3>
                    <div class="metric-value">${mesh.smoothness.normal_consistency.toFixed(3)}</div>
                    <div class="metric-label">higher = more consistent</div>
                </div>
            `;
            
            // Color metrics
            const colorGrid = document.getElementById('color-grid');
            if (mesh.color.has_vertex_colors) {
                colorGrid.innerHTML = `
                    <div class="metric-card">
                        <h3>Color Coverage</h3>
                        <div class="metric-value">${mesh.color.vertex_color_coverage.toFixed(1)}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Color Consistency</h3>
                        <div class="metric-value">${mesh.color.color_consistency.toFixed(3)}</div>
                        <div class="metric-label">higher = more consistent</div>
                    </div>
                    <div class="metric-card">
                        <h3>Mean RGB</h3>
                        <div class="metric-value">(${mesh.color.mean_rgb[0].toFixed(2)}, ${mesh.color.mean_rgb[1].toFixed(2)}, ${mesh.color.mean_rgb[2].toFixed(2)})</div>
                    </div>
                `;
            } else {
                colorGrid.innerHTML = '<div class="metric-card"><h3>No Vertex Colors</h3><p>This mesh does not have vertex colors.</p></div>';
            }
            
            // File metrics
            const fileGrid = document.getElementById('file-grid');
            fileGrid.innerHTML = `
                <div class="metric-card">
                    <h3>File Size</h3>
                    <div class="metric-value">${mesh.file.file_size_mb.toFixed(2)} MB</div>
                </div>
                <div class="metric-card">
                    <h3>Overall Density</h3>
                    <div class="metric-value">${mesh.file.overall_density.toFixed(1)}</div>
                    <div class="metric-label">vertices & faces per MB</div>
                </div>
            `;
            
            // Warnings and errors
            const warningsContainer = document.getElementById('warnings-container');
            if (mesh.warnings && mesh.warnings.length > 0) {
                warningsContainer.innerHTML = `
                    <div class="warnings-section">
                        <h3>⚠ Warnings (${mesh.warnings.length})</h3>
                        <ul>
                            ${mesh.warnings.map(w => `<li>${w}</li>`).join('')}
                        </ul>
                    </div>
                `;
            } else {
                warningsContainer.innerHTML = '<div class="warnings-section"><h3>✓ No Warnings</h3><p>No issues detected in this mesh.</p></div>';
            }
            
            const errorsContainer = document.getElementById('errors-container');
            if (mesh.errors && mesh.errors.length > 0) {
                errorsContainer.innerHTML = `
                    <div class="errors-section">
                        <h3>✗ Errors (${mesh.errors.length})</h3>
                        <ul>
                            ${mesh.errors.map(e => `<li>${e}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            // Show content
            document.getElementById('loading').style.display = 'none';
            document.getElementById('content').style.display = 'block';
            
            // Don't initialize 3D viewer yet - wait until tab is shown
        }
        
        // 3D Visualization
        let scene, camera, renderer, controls, meshObject;
        let wireframeMode = false;
        let showColors = true;
        
        function init3DViewer() {
            console.log('init3DViewer called');
            const container = document.getElementById('viewer-container');
            const infoElement = document.getElementById('viewer-info');
            
            if (!container) {
                console.error('Viewer container not found');
                infoElement.textContent = 'Error: Viewer container not found';
                return;
            }
            
            if (!mesh) {
                console.error('Mesh data not available');
                infoElement.textContent = 'Error: Mesh data not available';
                return;
            }
            
            console.log('Mesh data:', mesh);
            console.log('Mesh geometry:', mesh.geometry);
            
            if (!mesh.geometry) {
                console.error('Mesh geometry not available');
                infoElement.textContent = 'Mesh geometry not available for visualization. The mesh may be too large or failed to load.';
                return;
            }
            
            if (!mesh.geometry.vertices || !mesh.geometry.faces) {
                console.error('Mesh geometry missing vertices or faces:', mesh.geometry);
                infoElement.textContent = 'Error: Mesh geometry is incomplete (missing vertices or faces)';
                return;
            }
            
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            console.log('Container dimensions:', width, height);
            
            if (width === 0 || height === 0) {
                infoElement.textContent = 'Container has no dimensions. Please refresh the page.';
                console.error('Container dimensions are zero:', width, height);
                return;
            }
            
            if (typeof THREE === 'undefined') {
                console.error('Three.js library not loaded');
                infoElement.textContent = 'Error: Three.js library failed to load. Please check your internet connection.';
                return;
            }
            
            infoElement.textContent = 'Initializing 3D viewer...';
            console.log('Starting 3D viewer initialization...');
            
            try {
                // Scene
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);
                console.log('Scene created');
                
                // Camera
                camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 10000);
                console.log('Camera created');
                
                // Renderer
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(width, height);
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);
                console.log('Renderer created and added to container');
                
                // Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                console.log('Controls created');
                
                // Lighting
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                
                const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
                directionalLight1.position.set(1, 1, 1);
                scene.add(directionalLight1);
                
                const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
                directionalLight2.position.set(-1, -1, -1);
                scene.add(directionalLight2);
                console.log('Lighting added');
                
                // Load mesh geometry
                console.log('Loading mesh geometry...');
                const vertices = mesh.geometry.vertices;
                const faces = mesh.geometry.faces;
                const colors = mesh.geometry.vertex_colors;
                
                console.log('Vertices count:', vertices.length);
                console.log('Faces count:', faces.length);
                console.log('Has colors:', colors && colors.length > 0);
                
                if (!vertices || vertices.length === 0) {
                    throw new Error('No vertices found in mesh geometry');
                }
                
                if (!faces || faces.length === 0) {
                    throw new Error('No faces found in mesh geometry');
                }
                
                // Create geometry
                const geometry = new THREE.BufferGeometry();
                
                // Vertices
                const positions = new Float32Array(vertices.length * 3);
                for (let i = 0; i < vertices.length; i++) {
                    positions[i * 3] = vertices[i][0];
                    positions[i * 3 + 1] = vertices[i][1];
                    positions[i * 3 + 2] = vertices[i][2];
                }
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                console.log('Vertices added to geometry');
                
                // Colors (if available)
                if (colors && colors.length > 0) {
                    const colorArray = new Float32Array(vertices.length * 3);
                    for (let i = 0; i < Math.min(colors.length, vertices.length); i++) {
                        colorArray[i * 3] = colors[i][0];
                        colorArray[i * 3 + 1] = colors[i][1];
                        colorArray[i * 3 + 2] = colors[i][2];
                    }
                    geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
                    console.log('Colors added to geometry');
                }
                
                // Faces (indices)
                const indices = new Uint32Array(faces.length * 3);
                for (let i = 0; i < faces.length; i++) {
                    indices[i * 3] = faces[i][0];
                    indices[i * 3 + 1] = faces[i][1];
                    indices[i * 3 + 2] = faces[i][2];
                }
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                console.log('Faces added to geometry');
                
                geometry.computeVertexNormals();
                console.log('Vertex normals computed');
                
                // Material
                const material = new THREE.MeshStandardMaterial({
                    vertexColors: colors && colors.length > 0,
                    side: THREE.DoubleSide,
                    wireframe: wireframeMode
                });
                
                if (!colors || colors.length === 0) {
                    material.color = new THREE.Color(0x667eea);
                }
                console.log('Material created');
                
                // Mesh
                meshObject = new THREE.Mesh(geometry, material);
                scene.add(meshObject);
                console.log('Mesh added to scene');
                
                // Compute bounding box and center camera
                geometry.computeBoundingBox();
                const box = geometry.boundingBox;
                const center = new THREE.Vector3();
                box.getCenter(center);
                
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const distance = maxDim * 2;
                
                camera.position.set(center.x + distance, center.y + distance, center.z + distance);
                camera.lookAt(center);
                controls.target.copy(center);
                controls.update();
                console.log('Camera positioned');
            
                // Update info
                infoElement.textContent = 
                    `${formatNumber(vertices.length)} vertices, ${formatNumber(faces.length)} faces | ` +
                    `Drag to rotate, Scroll to zoom, Right-click to pan`;
                console.log('3D viewer initialized successfully');
                
                // Handle window resize
                const resizeHandler = () => {
                    const newWidth = container.clientWidth;
                    const newHeight = container.clientHeight;
                    if (newWidth > 0 && newHeight > 0) {
                        camera.aspect = newWidth / newHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(newWidth, newHeight);
                    }
                };
                window.addEventListener('resize', resizeHandler);
                
                // Animation loop
                function animate() {
                    requestAnimationFrame(animate);
                    if (controls) controls.update();
                    if (renderer && scene && camera) {
                        renderer.render(scene, camera);
                    }
                }
                animate();
            } catch (error) {
                console.error('Error initializing 3D viewer:', error);
                console.error('Error stack:', error.stack);
                infoElement.textContent = 'Error loading 3D viewer: ' + error.message + '. Check browser console for details.';
            }
        }
        
        function resetCamera() {
            if (!mesh.geometry || !controls) return;
            
            const vertices = mesh.geometry.vertices;
            const faces = mesh.geometry.faces;
            
            // Compute bounding box
            let minX = Infinity, minY = Infinity, minZ = Infinity;
            let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
            
            for (let v of vertices) {
                minX = Math.min(minX, v[0]);
                minY = Math.min(minY, v[1]);
                minZ = Math.min(minZ, v[2]);
                maxX = Math.max(maxX, v[0]);
                maxY = Math.max(maxY, v[1]);
                maxZ = Math.max(maxZ, v[2]);
            }
            
            const center = new THREE.Vector3(
                (minX + maxX) / 2,
                (minY + maxY) / 2,
                (minZ + maxZ) / 2
            );
            
            const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
            const distance = size * 2;
            
            camera.position.set(center.x + distance, center.y + distance, center.z + distance);
            camera.lookAt(center);
            controls.target.copy(center);
            controls.update();
        }
        
        function toggleWireframe() {
            if (!meshObject) return;
            wireframeMode = !wireframeMode;
            meshObject.material.wireframe = wireframeMode;
        }
        
        function toggleColors() {
            if (!meshObject || !mesh.geometry.vertex_colors) return;
            showColors = !showColors;
            meshObject.material.vertexColors = showColors;
            if (!showColors) {
                meshObject.material.color = new THREE.Color(0x667eea);
            }
        }
        
        render();
    </script>
</body>
</html>"""]
    
    return "\n".join(html)


def format_comparison_html(reports: List[EvaluationReport]) -> str:
    """Format comparison report as interactive HTML with JavaScript."""
    # Generate JSON data for JavaScript
    json_data = format_comparison_json(reports)
    
    html = ["""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mesh Quality Comparison Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .ranking-section {
            margin-bottom: 40px;
        }
        
        .ranking-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .ranking-table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        .ranking-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }
        
        .ranking-table tr:hover {
            background: #f5f7fa;
        }
        
        .ranking-table tr.rank-1 {
            background: #e8f5e9;
        }
        
        .ranking-table tr.rank-2 {
            background: #f1f8e9;
        }
        
        .ranking-table tr.rank-3 {
            background: #fff9e6;
        }
        
        .score-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .score-excellent { background: #4caf50; color: white; }
        .score-good { background: #8bc34a; color: white; }
        .score-fair { background: #ffc107; color: #333; }
        .score-poor { background: #ff9800; color: white; }
        .score-bad { background: #f44336; color: white; }
        
        .tabs {
            display: flex;
            border-bottom: 2px solid #eee;
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 12px 24px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1em;
            color: #666;
            transition: all 0.3s;
        }
        
        .tab:hover {
            color: #667eea;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            font-weight: 600;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .metric-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }
        
        .metric-card h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: 600;
            color: #667eea;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.9em;
        }
        
        .comparison-table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }
        
        .comparison-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }
        
        .comparison-table tr:hover {
            background: #f5f7fa;
        }
        
        .justification {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .justification h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .strength {
            color: #4caf50;
            margin-left: 20px;
        }
        
        .weakness {
            color: #f44336;
            margin-left: 20px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Mesh Quality Comparison</h1>
            <p id="mesh-count">Loading...</p>
        </div>
        <div class="content">
            <div id="loading" class="loading">Loading comparison data...</div>
            <div id="content" style="display: none;">
                <div class="ranking-section">
                    <h2>📊 Ranking</h2>
                    <table class="ranking-table" id="ranking-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Quality Score</th>
                                <th>File Name</th>
                                <th>Vertices</th>
                                <th>Faces</th>
                                <th>Holes</th>
                            </tr>
                        </thead>
                        <tbody id="ranking-body"></tbody>
                    </table>
                </div>
                
                <div class="tabs">
                    <button class="tab active" onclick="showTab('geometric')">Geometric</button>
                    <button class="tab" onclick="showTab('smoothness')">Smoothness</button>
                    <button class="tab" onclick="showTab('color')">Color</button>
                    <button class="tab" onclick="showTab('file')">File Quality</button>
                    <button class="tab" onclick="showTab('justification')">Justification</button>
                </div>
                
                <div id="geometric" class="tab-content active">
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th>Mesh</th>
                                <th>Aspect Ratio</th>
                                <th>Degenerate</th>
                                <th>Non-Manifold</th>
                                <th>Boundary Edges</th>
                                <th>Holes</th>
                                <th>Components</th>
                                <th>Manifold</th>
                                <th>Watertight</th>
                            </tr>
                        </thead>
                        <tbody id="geometric-body"></tbody>
                    </table>
                </div>
                
                <div id="smoothness" class="tab-content">
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th>Mesh</th>
                                <th>Normal Deviation</th>
                                <th>Roughness</th>
                                <th>Normal Consistency</th>
                            </tr>
                        </thead>
                        <tbody id="smoothness-body"></tbody>
                    </table>
                </div>
                
                <div id="color" class="tab-content">
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th>Mesh</th>
                                <th>Has Colors</th>
                                <th>Coverage</th>
                                <th>Consistency</th>
                            </tr>
                        </thead>
                        <tbody id="color-body"></tbody>
                    </table>
                </div>
                
                <div id="file" class="tab-content">
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th>Mesh</th>
                                <th>File Size (MB)</th>
                                <th>Vertices/MB</th>
                                <th>Faces/MB</th>
                                <th>Density</th>
                            </tr>
                        </thead>
                        <tbody id="file-body"></tbody>
                    </table>
                </div>
                
                <div id="justification" class="tab-content">
                    <div id="justification-content"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const data = """ + json_data + """;
        
        function getScoreClass(score) {
            if (score >= 0.8) return 'score-excellent';
            if (score >= 0.6) return 'score-good';
            if (score >= 0.4) return 'score-fair';
            if (score >= 0.2) return 'score-poor';
            return 'score-bad';
        }
        
        function formatNumber(num) {
            return new Intl.NumberFormat().format(num);
        }
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        function renderRanking() {
            const tbody = document.getElementById('ranking-body');
            tbody.innerHTML = '';
            
            data.comparison.meshes.forEach((mesh, idx) => {
                const row = document.createElement('tr');
                row.className = `rank-${mesh.rank}`;
                row.innerHTML = `
                    <td><strong>#${mesh.rank}</strong></td>
                    <td><span class="score-badge ${getScoreClass(mesh.quality_score)}">${mesh.quality_score.toFixed(4)}</span></td>
                    <td>${mesh.file_name}</td>
                    <td>${formatNumber(mesh.geometric.vertex_count)}</td>
                    <td>${formatNumber(mesh.geometric.face_count)}</td>
                    <td>${mesh.geometric.num_holes}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function renderGeometric() {
            const tbody = document.getElementById('geometric-body');
            tbody.innerHTML = '';
            
            data.comparison.meshes.forEach(mesh => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${mesh.file_name}</td>
                    <td>${mesh.geometric.avg_aspect_ratio.toFixed(3)}</td>
                    <td>${mesh.geometric.degenerate_triangles}</td>
                    <td>${mesh.geometric.num_non_manifold_edges + mesh.geometric.num_non_manifold_vertices}</td>
                    <td>${formatNumber(mesh.geometric.num_boundary_edges)}</td>
                    <td><strong>${mesh.geometric.num_holes}</strong></td>
                    <td>${mesh.geometric.num_connected_components}</td>
                    <td>${mesh.geometric.is_manifold ? '✓' : '✗'}</td>
                    <td>${mesh.geometric.is_watertight ? '✓' : '✗'}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function renderSmoothness() {
            const tbody = document.getElementById('smoothness-body');
            tbody.innerHTML = '';
            
            data.comparison.meshes.forEach(mesh => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${mesh.file_name}</td>
                    <td>${mesh.smoothness.avg_normal_deviation.toFixed(6)}</td>
                    <td>${mesh.smoothness.surface_roughness.toFixed(6)}</td>
                    <td>${mesh.smoothness.normal_consistency.toFixed(3)}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function renderColor() {
            const tbody = document.getElementById('color-body');
            tbody.innerHTML = '';
            
            data.comparison.meshes.forEach(mesh => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${mesh.file_name}</td>
                    <td>${mesh.color.has_vertex_colors ? '✓' : '✗'}</td>
                    <td>${mesh.color.has_vertex_colors ? mesh.color.vertex_color_coverage.toFixed(1) + '%' : 'N/A'}</td>
                    <td>${mesh.color.has_vertex_colors ? mesh.color.color_consistency.toFixed(3) : 'N/A'}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function renderFile() {
            const tbody = document.getElementById('file-body');
            tbody.innerHTML = '';
            
            data.comparison.meshes.forEach(mesh => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${mesh.file_name}</td>
                    <td>${mesh.file.file_size_mb.toFixed(2)}</td>
                    <td>${mesh.file.overall_density > 0 ? formatNumber(Math.round(mesh.file.overall_density / (mesh.file.file_size_mb || 1))) : 'N/A'}</td>
                    <td>${mesh.file.overall_density > 0 ? formatNumber(Math.round(mesh.file.overall_density / (mesh.file.file_size_mb || 1))) : 'N/A'}</td>
                    <td>${mesh.file.overall_density.toFixed(1)}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function renderJustification() {
            const container = document.getElementById('justification-content');
            container.innerHTML = '';
            
            const top3 = data.comparison.meshes.slice(0, 3);
            top3.forEach(mesh => {
                const div = document.createElement('div');
                div.className = 'justification';
                div.innerHTML = `
                    <h3>Rank ${mesh.rank}: ${mesh.file_name}</h3>
                    <p><strong>Quality Score:</strong> ${mesh.quality_score.toFixed(4)}</p>
                    <p><strong>Strengths:</strong></p>
                    <ul>
                        ${mesh.geometric.is_manifold ? '<li class="strength">✓ Manifold topology</li>' : ''}
                        ${mesh.geometric.is_watertight ? '<li class="strength">✓ Watertight mesh</li>' : ''}
                        ${mesh.geometric.avg_aspect_ratio > 0.5 ? '<li class="strength">✓ Good triangle quality</li>' : ''}
                        ${mesh.geometric.num_connected_components === 1 ? '<li class="strength">✓ Single connected component</li>' : ''}
                        ${mesh.geometric.num_holes === 0 ? '<li class="strength">✓ No holes</li>' : ''}
                        ${mesh.smoothness.normal_consistency > 0.8 ? '<li class="strength">✓ Smooth surface</li>' : ''}
                    </ul>
                    <p><strong>Weaknesses:</strong></p>
                    <ul>
                        ${!mesh.geometric.is_manifold ? '<li class="weakness">✗ Non-manifold topology</li>' : ''}
                        ${mesh.geometric.degenerate_triangles > 100 ? '<li class="weakness">✗ Many degenerate triangles (' + mesh.geometric.degenerate_triangles + ')</li>' : ''}
                        ${mesh.geometric.num_boundary_edges > 1000 ? '<li class="weakness">✗ Many boundary edges</li>' : ''}
                        ${mesh.geometric.num_holes > 0 ? '<li class="weakness">✗ Has ' + mesh.geometric.num_holes + ' hole(s)</li>' : ''}
                        ${mesh.smoothness.surface_roughness > 0.01 ? '<li class="weakness">✗ High surface roughness</li>' : ''}
                    </ul>
                `;
                container.appendChild(div);
            });
        }
        
        // Initialize
        document.getElementById('mesh-count').textContent = `Comparing ${data.comparison.num_meshes} meshes`;
        document.getElementById('loading').style.display = 'none';
        document.getElementById('content').style.display = 'block';
        
        renderRanking();
        renderGeometric();
        renderSmoothness();
        renderColor();
        renderFile();
        renderJustification();
    </script>
</body>
</html>"""]
    
    return "\n".join(html)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mesh files (FBX/PLY) using no-reference metrics. Supports single mesh and multi-mesh comparison modes."
    )
    parser.add_argument(
        "mesh_paths",
        nargs="+",
        type=str,
        help="Path(s) to mesh file(s) to evaluate. For single mesh mode, provide one path. For comparison mode, provide multiple paths."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Print report to console instead of showing GUI popup (single mesh mode only)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save report to file (optional). For multi-mesh mode, specify format: .txt, .json, .md, or .html"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "markdown", "html"],
        default="text",
        help="Output format for multi-mesh comparison (default: text)"
    )
    parser.add_argument(
        "--skip-detailed-non-manifold",
        action="store_true",
        help="Skip detailed non-manifold element detection (faster for large meshes, but less detailed warnings)"
    )
    
    args = parser.parse_args()
    
    mesh_paths = [Path(p).resolve() for p in args.mesh_paths]
    
    # Validate all paths exist
    for path in mesh_paths:
        if not path.exists():
            print(f"[Error] File not found: {path}", file=sys.stderr)
            sys.exit(1)
    
    try:
        # Determine mode: single mesh or multi-mesh comparison
        if len(mesh_paths) == 1:
            # Single mesh mode - generate HTML report by default
            mesh_path = mesh_paths[0]
            print(f"[Info] Loading and evaluating mesh file: {mesh_path}")
            report = evaluate_fbx_mesh(mesh_path, skip_detailed_non_manifold=args.skip_detailed_non_manifold)
            
            # Compute quality score (for single mesh, compare against itself)
            report.quality_score = compute_quality_score(report, [report])
            
            # Generate HTML report
            html_report = format_single_mesh_html(report)
            
            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                # Default: save next to mesh file with .html extension
                output_path = mesh_path.with_suffix('.html')
            
            # Save HTML report
            output_path.write_text(html_report, encoding="utf-8")
            print(f"[Info] HTML report saved to: {output_path}")
            
            # Open in browser (unless --no-gui flag is set)
            if not args.no_gui:
                import webbrowser
                import os
                file_url = f"file://{os.path.abspath(output_path)}"
                print(f"[Info] Opening report in browser...")
                webbrowser.open(file_url)
            else:
                print(f"[Info] Report saved. Open {output_path} in your browser to view.")
        else:
            # Multi-mesh comparison mode
            print(f"[Info] Comparing {len(mesh_paths)} meshes...")
            reports = []
            
            for i, mesh_path in enumerate(mesh_paths, 1):
                print(f"\n[{i}/{len(mesh_paths)}] Evaluating: {mesh_path.name}")
                try:
                    report = evaluate_fbx_mesh(mesh_path, skip_detailed_non_manifold=args.skip_detailed_non_manifold)
                    reports.append(report)
                except Exception as e:
                    print(f"[Error] Failed to evaluate {mesh_path.name}: {e}", file=sys.stderr)
                    continue
            
            if not reports:
                print("[Error] No meshes were successfully evaluated.", file=sys.stderr)
                sys.exit(1)
            
            # Generate comparison report
            comparison_text = compare_meshes(reports, output_format=args.format)
            
            if args.output:
                output_path = Path(args.output)
                output_path.write_text(comparison_text, encoding="utf-8")
                print(f"\n[Info] Comparison report saved to: {output_path}")
            
            print("\n" + comparison_text)
    
    except FileNotFoundError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

