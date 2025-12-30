# Investigating Low Quality Score Meshes

## Meshes to Investigate

1. **Niklas Nofog** - Session: `20251212_191309`
2. **Kilian Kozerke Fog** - Session: `20251211_152157` (Q_norm = 0.0)
3. **Jan Bienek Nofog** - Session: `20251209_163234`

## How to Evaluate These Meshes

### Option 1: Evaluate individually (unpaired)

```bash
python scripts/evaluation/evaluate_fbx_quality.py \
    /Volumes/Intenso/NoFog/20251212_191309/reconstruction/color_mesh.fbx \
    /Volumes/Intenso/Fog/20251211_152157/reconstruction/color_mesh.fbx \
    /Volumes/Intenso/NoFog/20251209_163234/reconstruction/color_mesh.fbx \
    --csv analysis/mesh_quality_batch/debug_quality_scores.csv
```

### Option 2: Evaluate as a small batch (recommended for debugging)

This will show you the batch normalization and relative rankings:

```bash
python scripts/evaluation/evaluate_fbx_quality.py \
    /Volumes/Intenso/NoFog/20251212_191309/reconstruction/color_mesh.fbx \
    /Volumes/Intenso/Fog/20251211_152157/reconstruction/color_mesh.fbx \
    /Volumes/Intenso/NoFog/20251209_163234/reconstruction/color_mesh.fbx \
    --csv analysis/mesh_quality_batch/debug_three_meshes.csv \
    --out-dir analysis/mesh_quality_batch/debug_output
```

### Option 3: Use PLY files if FBX doesn't exist

If the `.fbx` files don't exist, the script will automatically try `.ply`:

```bash
python scripts/evaluation/evaluate_fbx_quality.py \
    /Volumes/Intenso/NoFog/20251212_191309/reconstruction/color_mesh.ply \
    /Volumes/Intenso/Fog/20251211_152157/reconstruction/color_mesh.ply \
    /Volumes/Intenso/NoFog/20251209_163234/reconstruction/color_mesh.ply \
    --csv analysis/mesh_quality_batch/debug_quality_scores.csv
```

## Why These Meshes Have Low Scores

### Kilian Kozerke Fog (20251211_152157) - Q_norm = 0.0

**Q_raw = 0.162** (lowest in batch, normalized to 0.0)

**Severe quality issues:**
- **Degenerate triangles: 6,448** (very high)
- **Component count: 59** (mesh is fragmented into many pieces)
- **Boundary edge ratio: 1.57%** (many holes/openings)
- **Normal deviation: 27.27°** (very rough surface)
- **Dihedral penalty: 25.91** (poor triangle angles)
- **Surface roughness: 41,696** (extremely high variation)
- **Not single component** (fragmented mesh)
- **Vertex density stddev: 41,696** (very uneven distribution)

**Sub-scores:**
- S_geom = 0.0926 (very low - poor geometry)
- S_smooth = 0.0043 (extremely low - very rough)
- S_complete = 0.4921 (moderate - fragmented)
- S_color = 0.4101 (moderate)

### Niklas Sindemann Nofog (20251212_191309) - Q_norm = 0.053

**Q_raw = 0.184** (second lowest)

**Quality issues:**
- **Degenerate triangles: 5,122** (high)
- **Component count: 82** (very fragmented)
- **Boundary edge ratio: 1.66%** (many holes)
- **Normal deviation: 24.16°** (rough surface)
- **Dihedral penalty: 24.69** (poor angles)
- **Surface roughness: 32,193** (very high)
- **Not single component** (fragmented)
- **Vertex density stddev: 32,193** (uneven)

**Sub-scores:**
- S_geom = 0.0925 (very low)
- S_smooth = 0.0875 (low)
- S_complete = 0.5600 (moderate)
- S_color = 0.3173 (low)

### Jan Bienek Nofog (20251209_163234) - Q_norm = 0.082

**Q_raw = 0.196** (third lowest)

**Quality issues:**
- **Degenerate triangles: 6,802** (very high)
- **Component count: 97** (extremely fragmented)
- **Boundary edge ratio: 1.53%** (many holes)
- **Normal deviation: 25.17°** (rough surface)
- **Dihedral penalty: 24.76** (poor angles)
- **Surface roughness: 31,883** (very high)
- **Not single component** (fragmented)
- **Vertex density stddev: 31,883** (uneven)

**Sub-scores:**
- S_geom = 0.1039 (very low)
- S_smooth = 0.0723 (very low)
- S_complete = 0.5629 (moderate)
- S_color = 0.4111 (moderate)

## Common Issues Across All Three

1. **High degenerate triangle count** (5,000-7,000)
2. **High component count** (59-97 separate pieces)
3. **High boundary edge ratio** (1.5-1.7% - many holes)
4. **High normal deviation** (24-27° - rough surfaces)
5. **High dihedral penalty** (24-26 - poor triangle quality)
6. **Very high surface roughness** (30,000-42,000)
7. **Fragmented meshes** (not single component)

These meshes appear to have reconstruction issues, possibly due to:
- Poor tracking/pose estimation
- Insufficient coverage
- Motion blur or tracking loss
- Reconstruction algorithm failures

