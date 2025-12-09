"""
Mesh loading utilities for FBX and PLY files.

Provides a unified interface for loading meshes regardless of format.
"""

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


def load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    """
    Load a mesh from FBX or PLY file format.
    
    Args:
        mesh_path: Path to the mesh file (FBX or PLY)
        
    Returns:
        Open3D TriangleMesh object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If loading fails or format is unsupported
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    file_ext = mesh_path.suffix.lower()
    
    if file_ext == '.fbx':
        return _load_fbx(mesh_path)
    elif file_ext == '.ply':
        return _load_ply(mesh_path)
    else:
        # Try Open3D for other formats (OBJ, etc.)
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if len(mesh.vertices) == 0:
                raise ValueError(f"Loaded mesh has no vertices from {mesh_path}")
            return mesh
        except Exception as e:
            raise ValueError(f"Unsupported file format or failed to load {mesh_path}: {e}")


def _load_fbx(fbx_path: Path) -> o3d.geometry.TriangleMesh:
    """Load an FBX file using Aspose.3D (converts to PLY temporarily)."""
    if not ASPOSE_AVAILABLE:
        raise ValueError(
            "FBX loading requires aspose-3d. Install with: pip install aspose-3d"
        )
    
    print(f"[Info] Loading FBX file: {fbx_path}")
    
    # Load FBX scene with Aspose
    scene = a3d.Scene.from_file(str(fbx_path))
    
    # Save to temporary PLY file
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        tmp_ply_path = tmp_file.name
    
    try:
        # Save scene as PLY (preserves vertex colors)
        scene.save(tmp_ply_path, a3d.FileFormat.PLY)  # type: ignore
        
        # Load PLY with Open3D
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


def _load_ply(ply_path: Path) -> o3d.geometry.TriangleMesh:
    """Load a PLY file using Open3D."""
    print(f"[Info] Loading PLY file: {ply_path}")
    
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    
    if len(mesh.vertices) == 0:
        raise ValueError("Loaded mesh has no vertices")
    
    print(f"[Info] Successfully loaded PLY mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    if mesh.has_vertex_colors():
        print(f"[Info] Vertex colors detected: {len(mesh.vertex_colors)} colors")
    
    return mesh


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh, num_points: Optional[int] = None) -> o3d.geometry.PointCloud:
    """
    Convert a mesh to a point cloud by sampling points from the surface.
    
    Args:
        mesh: Input triangle mesh
        num_points: Number of points to sample. If None, uses all vertices.
        
    Returns:
        Point cloud sampled from the mesh surface
    """
    if num_points is None:
        # Use all vertices
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
        return pcd
    else:
        # Sample points uniformly from the mesh surface
        return mesh.sample_points_uniformly(number_of_points=num_points)

