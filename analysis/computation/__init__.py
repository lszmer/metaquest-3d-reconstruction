"""
Evaluation module for benchmarking 3D mesh scans against ground truth.

This module provides tools for comparing reconstructed meshes with ground truth
models using various metrics and visualization techniques.
"""

# Import only when needed to avoid circular dependencies
__all__ = [
    "compare_meshes",
    "ComparisonMetrics",
    "ComparisonReport",
    "visualize_comparison",
]

