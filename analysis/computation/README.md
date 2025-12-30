# Mesh Evaluation and Benchmarking

This module provides tools for comparing reconstructed 3D mesh scans against ground truth models.

## Features

### Metrics Computed

1. **Chamfer Distance**: Bidirectional average distance between point clouds
2. **Hausdorff Distance**: Maximum distance between point clouds
3. **Point-to-Surface Distance**: Distance from scan points to nearest GT surface
   - Mean, median, standard deviation, and maximum
4. **F-score**: Precision, recall, and F-score at a distance threshold
5. **Volume Metrics**: Volume comparison and Intersection over Union (IoU)
6. **Scale and Alignment**: Bounding box ratios and center offset

### Visualization

- **Error Heatmap**: Color-coded point cloud showing per-point error
- **Side-by-Side Comparison**: Visual comparison of scan and ground truth
- **Overlay Visualization**: Semi-transparent overlay for alignment inspection
- **Export**: Save visualizations as PLY files and metrics as JSON

## Usage

### Command Line

```bash
# Basic comparison
python scripts/evaluation/compare_mesh_to_ground_truth.py \
    --scan path/to/scan.fbx \
    --gt path/to/ground_truth.fbx

# With custom threshold
python scripts/evaluation/compare_mesh_to_ground_truth.py \
    --scan scan.fbx \
    --gt gt.fbx \
    --threshold 0.01

# Specify output directory
python scripts/evaluation/compare_mesh_to_ground_truth.py \
    --scan scan.fbx \
    --gt gt.fbx \
    --output results/

# Non-interactive (no visualization windows)
python scripts/evaluation/compare_mesh_to_ground_truth.py \
    --scan scan.fbx \
    --gt gt.fbx \
    --no-interactive
```

### Python API

```python
from pathlib import Path
from evaluation.compare_mesh_to_ground_truth import compare_meshes, visualize_comparison

# Compare meshes
report = compare_meshes(
    scan_path=Path("scan.fbx"),
    ground_truth_path=Path("ground_truth.fbx"),
    threshold=0.01,  # Optional: distance threshold for F-score
    num_sample_points=50000  # Points to sample from each mesh
)

# Visualize results
visualize_comparison(
    report=report,
    output_dir=Path("results/"),
    interactive=True
)

# Access metrics
print(f"Chamfer Distance: {report.metrics.chamfer_distance}")
print(f"F-score: {report.metrics.f_score}")
```

## Output Files

The evaluation generates the following files in the output directory:

- `error_heatmap.ply`: Point cloud colored by error magnitude
- `comparison_metrics.json`: All computed metrics in JSON format

## Dependencies

- `numpy`: Numerical computations
- `open3d`: 3D mesh processing and visualization
- `aspose-3d`: FBX file support (optional, install with `pip install aspose-3d`)
- `matplotlib`: Colormap support for visualizations

## Supported Formats

- **Input**: FBX, PLY (and other formats supported by Open3D)
- **Output**: PLY (point clouds), JSON (metrics)

## Notes

- For large meshes, the script samples points for efficient computation
- Volume metrics require watertight meshes
- The automatic threshold is set to 1% of the ground truth bounding box diagonal
- Interactive visualizations can be disabled with `--no-interactive` for batch processing

