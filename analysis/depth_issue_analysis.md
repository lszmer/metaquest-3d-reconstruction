# Depth Map Conversion Issue Analysis

## Problem Description

For some recordings, depth maps exhibit "jumping" or visual artifacts in specific timestamp ranges. Example:
- Recording: `/Volumes/Intenso/NoFog/20251209_163234`
- Problematic range: `1765297957805` to `1765297964444` (approximately 6.6 seconds)

## Root Cause Analysis

### Primary Issue: Invalid `far` Parameter

**All frames in the recording have `far=inf` (infinity)** in the depth descriptors CSV. This causes the depth conversion formula to use a special case:

```python
# From depth_utils.py
if np.isinf(far) or far < near:
    x = -2.0 * near  # = -0.2 when near=0.1
    y = -1.0
```

When `far=inf`, the conversion formula becomes:
```
linear_depth = -0.2 / (ndc - 1.0)
where ndc = raw_depth * 2.0 - 1.0
```

This formula is **highly sensitive** to small changes in raw depth values, especially near 1.0.

### Secondary Issue: Raw Depth Value Anomalies

In the problematic range, the raw depth values show significant differences:

**Normal frames (before problematic range):**
- Raw min: ~0.858-0.860
- Raw max: ~0.9973-0.9974
- Linear depth range: ~0.70-38.10 meters

**Problematic frames:**
- Raw min: ~0.801-0.818 (much lower!)
- One frame has min=0.0000 (likely invalid/corrupted)
- Raw max: ~0.9973-0.9974 (similar)
- Linear depth range: ~0.50-38.10 meters

The lower minimum raw depth values, combined with the `far=inf` conversion formula, cause the linear depth to have different ranges. When these are clipped and normalized to 0-255 for PNG output, the visual "jumping" occurs.

## Why This Happens

1. **Capture device issue**: The depth sensor may have had temporary issues during capture, producing different raw depth ranges
2. **File corruption**: Some raw depth files may be partially corrupted
3. **Conversion sensitivity**: With `far=inf`, the conversion is extremely sensitive to raw depth value changes

## Solutions

### Option 1: Fix the Conversion Formula (Recommended)

Modify the depth conversion to handle `far=inf` more robustly by using a reasonable default far value:

```python
# In depth_utils.py, modify compute_ndc_to_linear_depth_params:
def compute_ndc_to_linear_depth_params(near, far):
    if np.isinf(far) or far < near:
        # Use a reasonable default far value (e.g., 100 meters)
        # This prevents extreme sensitivity to raw depth changes
        far = 100.0  # or derive from typical scene depth
        x = -2.0 * far * near / (far - near)
        y = -(far + near) / (far - near)
    else:
        x = -2.0 * far * near / (far - near)
        y = -(far + near) / (far - near)
    return x, y
```

### Option 2: Validate and Filter Raw Depth Values

Add validation to detect and handle anomalous raw depth values:

```python
# In depth_data_io.py, enhance is_depth_map_valid:
def is_depth_map_valid(self, depth_map: np.ndarray) -> bool:
    is_valid = (depth_map != 0).any() and (depth_map != 1).any()
    is_valid = is_valid and not np.isnan(depth_map).any()
    is_valid = is_valid and (depth_map >= 0).all()
    
    # Additional check: ensure depth values are in reasonable range
    # Most valid depth values should be between 0.5 and 1.0 for this sensor
    valid_pixels = (depth_map > 0.5) & (depth_map < 1.0)
    valid_ratio = np.sum(valid_pixels) / depth_map.size
    is_valid = is_valid and (valid_ratio > 0.1)  # At least 10% valid pixels
    
    return bool(is_valid)
```

### Option 3: Post-Process Linear Depth Maps

Add smoothing or filtering to the linear depth maps to reduce artifacts:

```python
# In convert_depth_to_linear.py, after conversion:
if depth_to_linear_config.smooth_depth:
    from scipy import ndimage
    linear_depth_map = ndimage.gaussian_filter(linear_depth_map, sigma=0.5)
```

### Option 4: Re-capture or Exclude Problematic Frames

For existing recordings:
1. Identify problematic frames using the diagnostic script
2. Exclude them from reconstruction
3. Or manually fix the `far` values in the descriptors CSV

## Diagnostic Tools

Use the `investigate_depth_issues.py` script to analyze problematic recordings:

```bash
python analysis/investigate_depth_issues.py <session_dir> \
    --side left \
    --start-timestamp <start> \
    --end-timestamp <end>
```

## Recommendations

1. **Immediate fix**: Implement Option 1 (fix conversion formula) to handle `far=inf` more robustly
2. **Long-term**: Investigate why the capture device is producing `far=inf` values and fix at the source
3. **Validation**: Add Option 2 validation to catch problematic frames early
4. **Monitoring**: Use the diagnostic script to identify recordings with this issue before reconstruction

