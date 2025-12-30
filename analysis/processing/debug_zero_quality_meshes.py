#!/usr/bin/env python3
"""
Debug zero-quality mesh issues in 3D reconstruction pipeline.

This diagnostic script investigates why certain reconstructed meshes receive
a quality score of 0. It performs detailed analysis of mesh geometry, topology,
and quality metrics to identify the root causes of poor reconstruction quality.

Key features:
- Analyzes detailed geometric properties (vertices, triangles, components)
- Examines topological characteristics (manifold, watertight, boundary edges)
- Evaluates shape quality metrics (aspect ratio, skewness, degenerate triangles)
- Assesses surface smoothness and normal consistency
- Provides comprehensive debugging output for quality score calculation

Console Usage Examples:
    # Run debug analysis on hardcoded problematic meshes
    python analysis/processing/debug_zero_quality_meshes.py

    # To debug different meshes, modify the hardcoded mesh list in main():
    # meshes_to_debug = [
    #     {
    #         "name": "Your_Mesh_Name",
    #         "path": Path("/path/to/your/mesh.fbx"),
    #         "condition": "Fog"  # or "NoFog"
    #     }
    # ]

Note: This script currently analyzes a hardcoded set of problematic meshes.
Modify the mesh list in the main() function to debug different meshes.
"""

import sys
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(script_dir))

from analysis.computation.evaluate_fbx_quality import (
    compute_raw_metrics_for_mesh,
    compute_quality_scores,
    QualityScores,
    RawMeshMetrics
)


def print_detailed_metrics(raw: RawMeshMetrics, score: QualityScores):
    """Print detailed metrics for debugging."""
    print("\n" + "=" * 80)
    print(f"DETAILED METRICS FOR: {raw.name}")
    print("=" * 80)
    
    print("\n--- Basic Geometry ---")
    print(f"  Vertices: {raw.num_vertices:,}")
    print(f"  Triangles: {raw.num_triangles:,}")
    print(f"  Components: {raw.component_count}")
    print(f"  Is manifold: {raw.is_manifold}")
    print(f"  Is watertight: {raw.is_watertight}")
    print(f"  Is single component: {raw.is_single_component}")
    
    print("\n--- Shape Metrics ---")
    print(f"  Mean aspect ratio: {raw.mean_aspect_ratio:.4f}")
    print(f"  Mean skewness: {raw.mean_skewness:.4f}")
    print(f"  Degenerate triangles: {raw.degenerate_triangles:,}")
    
    print("\n--- Topology Metrics ---")
    print(f"  Total edges: {raw.total_edges:,}")
    print(f"  Non-manifold edges: {raw.non_manifold_edges:,}")
    print(f"  Boundary edge ratio: {raw.boundary_edge_ratio:.6f} ({raw.boundary_edge_ratio * 100:.2f}%)")
    
    print("\n--- Smoothness Metrics ---")
    print(f"  Normal deviation (avg): {raw.normal_deviation_avg_deg:.2f}°")
    print(f"  Dihedral min: {raw.dihedral_min_deg:.2f}°")
    print(f"  Dihedral max: {raw.dihedral_max_deg:.2f}°")
    print(f"  Dihedral penalty: {raw.dihedral_penalty:.4f}")
    print(f"  Surface roughness (stddev): {raw.surface_roughness:.4f}")
    
    print("\n--- Completeness Metrics ---")
    print(f"  Vertex density stddev: {raw.vertex_density_stddev:.4f}")
    
    print("\n--- Color Metrics ---")
    print(f"  Has color: {raw.has_color}")
    print(f"  Uncolored vertex ratio: {raw.uncolored_vertex_ratio:.4f}")
    print(f"  Color gradient stddev: {raw.color_gradient_stddev:.4f}")
    
    print("\n--- Quality Scores ---")
    print(f"  S_shape: {score.S_shape:.6f}")
    print(f"  S_topology: {score.S_topology:.6f}")
    print(f"  S_bonuses: {score.S_bonuses:.6f}")
    print(f"  S_geom: {score.S_geom:.6f}")
    print(f"  S_smooth: {score.S_smooth:.6f}")
    print(f"  S_complete: {score.S_complete:.6f}")
    print(f"  S_color: {score.S_color:.6f}")
    print(f"  Q_raw: {score.Q_raw:.6f}")
    print(f"  Q_norm: {score.Q_norm:.6f}")
    
    if score.Q_norm == 0.0:
        print("\n⚠️  WARNING: Q_norm is 0.0!")
        print("   This could mean:")
        print("   - Q_raw is the minimum in the batch (normalized to 0)")
        print("   - All meshes have identical Q_raw (normalized to 0.5, but if only one mesh, could be 0)")
        print("   - The mesh has severe quality issues")


def main():
    # Define the three problematic meshes based on session IDs
    base_path = Path("/Volumes/Intenso")
    
    meshes_to_debug = [
        {
            "name": "Niklas_Nofog_20251212_191309",
            "path": base_path / "NoFog" / "20251212_191309" / "reconstruction" / "color_mesh.fbx",
            "condition": "NoFog"
        },
        {
            "name": "Kilian_Kozerke_Fog_20251211_152157",
            "path": base_path / "Fog" / "20251211_152157" / "reconstruction" / "color_mesh.fbx",
            "condition": "Fog"
        },
        {
            "name": "Jan_Bienek_Nofog_20251209_163234",
            "path": base_path / "NoFog" / "20251209_163234" / "reconstruction" / "color_mesh.fbx",
            "condition": "NoFog"
        },
    ]
    
    print("=" * 80)
    print("DEBUGGING ZERO QUALITY SCORE MESHES")
    print("=" * 80)
    
    # Check if files exist
    print("\nChecking mesh file existence...")
    valid_meshes = []
    for mesh_info in meshes_to_debug:
        if mesh_info["path"].exists():
            print(f"  ✓ {mesh_info['name']}: {mesh_info['path']}")
            valid_meshes.append(mesh_info)
        else:
            # Try PLY instead
            ply_path = mesh_info["path"].with_suffix(".ply")
            if ply_path.exists():
                print(f"  ✓ {mesh_info['name']}: {ply_path} (PLY)")
                mesh_info["path"] = ply_path
                valid_meshes.append(mesh_info)
            else:
                print(f"  ✗ {mesh_info['name']}: NOT FOUND")
                print(f"     Tried: {mesh_info['path']}")
                print(f"     Tried: {ply_path}")
    
    if not valid_meshes:
        print("\n❌ No valid mesh files found!")
        return
    
    print(f"\nFound {len(valid_meshes)} valid mesh(es) to evaluate")
    
    # Compute raw metrics for all meshes
    print("\n" + "=" * 80)
    print("COMPUTING RAW METRICS")
    print("=" * 80)
    
    raw_metrics_list = []
    for mesh_info in valid_meshes:
        print(f"\nProcessing: {mesh_info['name']}")
        try:
            raw = compute_raw_metrics_for_mesh(mesh_info["path"], mesh_info["name"])
            raw_metrics_list.append(raw)
            print(f"  ✓ Successfully computed raw metrics")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not raw_metrics_list:
        print("\n❌ No raw metrics computed!")
        return
    
    # Compute quality scores (batch normalization)
    print("\n" + "=" * 80)
    print("COMPUTING QUALITY SCORES (with batch normalization)")
    print("=" * 80)
    
    scores = compute_quality_scores(raw_metrics_list)
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for raw, score in zip(raw_metrics_list, scores):
        print_detailed_metrics(raw, score)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nEvaluated {len(scores)} mesh(es):")
    for score in scores:
        status = "⚠️  ZERO" if score.Q_norm == 0.0 else "✓ OK"
        print(f"  {status} {score.name}: Q_norm = {score.Q_norm:.6f}, Q_raw = {score.Q_raw:.6f}")
    
    # If all have Q_norm = 0, explain why
    if all(s.Q_norm == 0.0 for s in scores):
        print("\n⚠️  All meshes have Q_norm = 0.0")
        print("   This happens when:")
        print("   - All meshes have identical Q_raw (normalized to 0.5)")
        print("   - Or when there's only one mesh in the batch")
        print("   - Check Q_raw values above to see actual quality")
    
    # If some have Q_norm = 0, explain
    zero_scores = [s for s in scores if s.Q_norm == 0.0]
    if zero_scores and len(zero_scores) < len(scores):
        print(f"\n⚠️  {len(zero_scores)} mesh(es) have Q_norm = 0.0")
        print("   These have the lowest Q_raw in the batch")
        q_raw_values = [s.Q_raw for s in scores]
        print(f"   Q_raw range: {min(q_raw_values):.6f} to {max(q_raw_values):.6f}")
        for s in zero_scores:
            print(f"   - {s.name}: Q_raw = {s.Q_raw:.6f} (lowest)")


if __name__ == "__main__":
    main()

