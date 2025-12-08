#!/usr/bin/env python3
"""
FBX Mesh Downsampling Script

Downsamples an FBX mesh file by reducing the number of vertices to a specified
percentage of the original count. Preserves vertex colors and mesh topology.

Uses Open3D for mesh simplification and Aspose.3D for FBX file handling.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

try:
    import aspose.threed as a3d
    ASPOSE_AVAILABLE = True
except ImportError:
    ASPOSE_AVAILABLE = False
    print("[Error] aspose-3d is not installed. Please install it using: pip install aspose-3d", file=sys.stderr)
    sys.exit(1)


def load_fbx_as_open3d_mesh(fbx_path: Path) -> o3d.geometry.TriangleMesh:
    """
    Load an FBX file and convert it to an Open3D TriangleMesh.
    
    Args:
        fbx_path: Path to the input FBX file
        
    Returns:
        Open3D TriangleMesh object
        
    Raises:
        FileNotFoundError: If the FBX file doesn't exist
        ValueError: If loading or conversion fails
    """
    if not fbx_path.exists():
        raise FileNotFoundError(f"FBX file not found: {fbx_path}")
    
    if not fbx_path.suffix.lower() == '.fbx':
        raise ValueError(f"Input file must be a .fbx file, got: {fbx_path.suffix}")
    
    print(f"[Info] Loading FBX file: {fbx_path}")
    
    # Load FBX scene with Aspose
    scene = a3d.Scene.from_file(str(fbx_path))
    
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
        
        return mesh
        
    except Exception as e:
        # Clean up temp file on error
        if Path(tmp_ply_path).exists():
            Path(tmp_ply_path).unlink()
        raise ValueError(f"Failed to convert/load FBX via PLY: {e}")


def save_open3d_mesh_as_fbx(mesh: o3d.geometry.TriangleMesh, fbx_path: Path) -> None:
    """
    Save an Open3D TriangleMesh to an FBX file.
    
    Args:
        mesh: Open3D TriangleMesh object
        fbx_path: Path to the output FBX file
        
    Raises:
        ValueError: If saving fails
    """
    print(f"[Info] Saving mesh to FBX: {fbx_path}")
    
    # Ensure output directory exists
    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to temporary PLY file first
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        tmp_ply_path = tmp_file.name
    
    try:
        # Save mesh as PLY with Open3D
        o3d.io.write_triangle_mesh(tmp_ply_path, mesh)
        
        # Load PLY with Aspose and save as FBX
        scene = a3d.Scene.from_file(tmp_ply_path)
        scene.save(str(fbx_path))
        
        # Clean up temp file
        Path(tmp_ply_path).unlink()
        
        if not fbx_path.exists():
            raise ValueError("FBX file was not created. Check for errors in the conversion process.")
        
        file_size = fbx_path.stat().st_size
        print(f"[Info] Successfully saved FBX file: {fbx_path.name}")
        print(f"[Info] Output file size: {file_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        # Clean up temp file on error
        if Path(tmp_ply_path).exists():
            Path(tmp_ply_path).unlink()
        raise ValueError(f"Failed to save mesh as FBX: {e}")


def downsample_mesh(mesh: o3d.geometry.TriangleMesh, target_vertex_percent: float) -> o3d.geometry.TriangleMesh:
    """
    Downsample a mesh to a specified percentage of original vertices.
    
    Args:
        mesh: Open3D TriangleMesh to downsample
        target_vertex_percent: Target percentage of vertices to keep (0-100)
        
    Returns:
        Downsampled Open3D TriangleMesh
        
    Raises:
        ValueError: If target percentage is invalid or downsampling fails
    """
    if target_vertex_percent <= 0 or target_vertex_percent > 100:
        raise ValueError(f"Target vertex percentage must be between 0 and 100, got: {target_vertex_percent}")
    
    original_vertex_count = len(mesh.vertices)
    target_vertex_count = int(original_vertex_count * (target_vertex_percent / 100.0))
    
    # Ensure we have at least 4 vertices (minimum for a valid mesh)
    target_vertex_count = max(4, target_vertex_count)
    
    # If target is very close to original, return original
    if target_vertex_count >= original_vertex_count:
        print(f"[Info] Target vertex count ({target_vertex_count}) >= original ({original_vertex_count}), returning original mesh")
        return mesh
    
    print(f"[Info] Downsampling mesh from {original_vertex_count} to {target_vertex_count} vertices ({target_vertex_percent:.1f}%)")
    
    # Check if mesh has triangles (is a proper mesh) or is just a point cloud
    has_triangles = len(mesh.triangles) > 0
    
    if not has_triangles:
        print(f"[Info] Mesh has no triangles (point cloud), using uniform downsampling")
        # Convert to point cloud for downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
        
        # Validate point cloud
        if len(pcd.points) == 0:
            raise ValueError("Point cloud has no points")
        
        # Use uniform downsampling for point clouds
        # Calculate voxel size based on bounding box and target vertex count
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_size = bbox.get_extent()
        diagonal = np.linalg.norm(bbox_size)
        
        print(f"[Info] Bounding box size: {bbox_size}, diagonal: {diagonal:.6f}")
        
        if diagonal <= 0 or np.any(np.isnan(bbox_size)) or np.any(np.isinf(bbox_size)):
            # Fallback: use simple uniform downsampling with every_k_points
            print(f"[Warning] Bounding box is invalid, using simple uniform downsampling")
            every_k = max(1, original_vertex_count // target_vertex_count)
            downsampled_pcd = pcd.uniform_down_sample(every_k_points=every_k)
        else:
            # Estimate voxel size: voxel_size = diagonal / (target_vertex_count^(1/3))
            # Add a safety factor to ensure we get close to target
            voxel_size = diagonal / (target_vertex_count ** (1/3)) * 1.2
            
            if voxel_size <= 0 or np.isnan(voxel_size) or np.isinf(voxel_size):
                print(f"[Warning] Calculated voxel size is invalid ({voxel_size}), using uniform downsampling")
                every_k = max(1, original_vertex_count // target_vertex_count)
                downsampled_pcd = pcd.uniform_down_sample(every_k_points=every_k)
            else:
                print(f"[Info] Using voxel size: {voxel_size:.6f} for uniform downsampling")
                # Try uniform downsampling first
                every_k = max(1, original_vertex_count // target_vertex_count)
                downsampled_pcd = pcd.uniform_down_sample(every_k_points=every_k)
                
                # If we still have too many points, use voxel downsampling
                if len(downsampled_pcd.points) > target_vertex_count * 1.5:
                    print(f"[Info] Uniform downsampling resulted in {len(downsampled_pcd.points)} points, using voxel downsampling")
                    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # Convert back to mesh (point cloud mesh)
        simplified_mesh = o3d.geometry.TriangleMesh()
        simplified_mesh.vertices = downsampled_pcd.points
        if downsampled_pcd.has_colors():
            simplified_mesh.vertex_colors = downsampled_pcd.colors
        
    else:
        # Use quadric decimation for meshes with triangles
        # This method preserves mesh features better than vertex clustering
        try:
            # Estimate target triangles (roughly 2x vertices for a typical mesh)
            target_triangles = max(1, int(target_vertex_count * 1.5))
            simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
            
            # The quadric decimation works with triangles, so we approximate
            # We'll use vertex clustering if we need more precise vertex count control
            actual_vertex_count = len(simplified_mesh.vertices)
            
            # If the result is not close enough to target, use vertex clustering for fine-tuning
            if abs(actual_vertex_count - target_vertex_count) > original_vertex_count * 0.1:
                print(f"[Info] Quadric decimation resulted in {actual_vertex_count} vertices, using vertex clustering for fine-tuning")
                
                # Calculate voxel size for vertex clustering to achieve target vertex count
                bbox = mesh.get_axis_aligned_bounding_box()
                bbox_size = bbox.get_extent()
                diagonal = np.linalg.norm(bbox_size)
                
                if diagonal <= 0:
                    raise ValueError("Mesh bounding box has zero size, cannot downsample")
                
                # Estimate voxel size (rough approximation)
                voxel_size = diagonal / (target_vertex_count ** (1/3)) * 1.2
                
                if voxel_size <= 0:
                    raise ValueError(f"Calculated voxel size is invalid: {voxel_size}")
                
                print(f"[Info] Using voxel size: {voxel_size:.6f} for vertex clustering")
                simplified_mesh = mesh.simplify_vertex_clustering(
                    voxel_size=voxel_size,
                    contraction=o3d.geometry.SimplificationContraction.Average
                )
                
                actual_vertex_count = len(simplified_mesh.vertices)
                print(f"[Info] Vertex clustering resulted in {actual_vertex_count} vertices")
        
        except Exception as e:
            # Fallback to vertex clustering if quadric decimation fails
            print(f"[Warning] Quadric decimation failed: {e}, trying vertex clustering")
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_extent()
            diagonal = np.linalg.norm(bbox_size)
            
            if diagonal <= 0:
                raise ValueError("Mesh bounding box has zero size, cannot downsample")
            
            voxel_size = diagonal / (target_vertex_count ** (1/3)) * 1.2
            
            if voxel_size <= 0:
                raise ValueError(f"Calculated voxel size is invalid: {voxel_size}")
            
            simplified_mesh = mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average
            )
    
    # Ensure mesh has normals for proper rendering
    if not simplified_mesh.has_vertex_normals() and len(simplified_mesh.vertices) > 0:
        simplified_mesh.compute_vertex_normals()
    
    # Preserve vertex colors if they exist
    if mesh.has_vertex_colors() and len(simplified_mesh.vertices) > 0:
        print(f"[Info] Vertex colors preserved: {len(simplified_mesh.vertex_colors) if simplified_mesh.has_vertex_colors() else 0} colors")
    
    print(f"[Info] Downsampling complete: {len(simplified_mesh.vertices)} vertices, {len(simplified_mesh.triangles)} faces")
    print(f"[Info] Reduction: {original_vertex_count - len(simplified_mesh.vertices)} vertices removed ({100 * (1 - len(simplified_mesh.vertices) / original_vertex_count):.1f}%)")
    
    return simplified_mesh


def downsample_fbx_mesh(input_fbx_path: Path, output_fbx_path: Path, vertex_percent: float) -> None:
    """
    Downsample an FBX mesh file to a specified percentage of vertices.
    
    Args:
        input_fbx_path: Path to the input FBX file
        output_fbx_path: Path to the output FBX file
        vertex_percent: Percentage of vertices to keep (0-100)
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If processing fails
    """
    # Load FBX mesh
    mesh = load_fbx_as_open3d_mesh(input_fbx_path)
    
    # Downsample mesh
    downsampled_mesh = downsample_mesh(mesh, vertex_percent)
    
    # Save downsampled mesh to FBX
    save_open3d_mesh_as_fbx(downsampled_mesh, output_fbx_path)
    
    print(f"[Info] Downsampling complete!")
    print(f"[Info] Input:  {input_fbx_path} ({len(mesh.vertices)} vertices)")
    print(f"[Info] Output: {output_fbx_path} ({len(downsampled_mesh.vertices)} vertices)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Downsample an FBX mesh file by reducing vertices to a specified percentage. "
                    "Preserves vertex colors and mesh topology."
    )
    parser.add_argument(
        'input_fbx',
        type=str,
        help="Path to the input FBX file"
    )
    parser.add_argument(
        'output_fbx',
        type=str,
        help="Path to the output FBX file"
    )
    parser.add_argument(
        '--vertex-percent',
        type=float,
        default=50.0,
        help="Percentage of vertices to keep (0-100). Default: 50.0 (keep 50%% of vertices)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        input_path = Path(args.input_fbx).resolve()
        output_path = Path(args.output_fbx).resolve()
        
        # Validate vertex percentage
        if args.vertex_percent <= 0 or args.vertex_percent > 100:
            print(f"[Error] Vertex percentage must be between 0 and 100, got: {args.vertex_percent}", file=sys.stderr)
            sys.exit(1)
        
        downsample_fbx_mesh(input_path, output_path, args.vertex_percent)
        
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

