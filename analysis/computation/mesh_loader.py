"""
Mesh loading utilities for 3D mesh processing and evaluation.

This module provides a unified interface for loading and processing 3D mesh files
in different formats (FBX, PLY). It handles format-specific loading requirements
and provides utilities for converting meshes to point clouds for analysis.

Key features:
- Unified mesh loading interface for FBX and PLY formats
- Automatic format detection based on file extension
- FBX support using Aspose.3D library (optional dependency)
- PLY support using Open3D library
- Mesh to point cloud conversion utilities
- Error handling for unsupported formats and missing files

Supported formats:
- FBX: Autodesk FBX format (requires aspose-3d package)
- PLY: Polygon File Format (supported by Open3D)

Dependencies:
- open3d: Core 3D processing library
- aspose-3d: FBX file support (optional, install with: pip install aspose-3d)

Usage:
    from analysis.computation.mesh_loader import load_mesh, mesh_to_point_cloud

    # Load a mesh from file
    mesh = load_mesh(Path("model.fbx"))

    # Convert mesh to point cloud for analysis
    point_cloud = mesh_to_point_cloud(mesh, num_points=50000)
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
    """Load an FBX file using Aspose.3D (converts to PLY temporarily or extracts directly)."""
    if not ASPOSE_AVAILABLE:
        raise ValueError(
            "FBX loading requires aspose-3d. Install with: pip install aspose-3d"
        )
    
    print(f"[Info] Loading FBX file: {fbx_path}")
    
    # Load FBX scene with Aspose
    scene = a3d.Scene.from_file(str(fbx_path))
    
    # Try to extract mesh data directly from Aspose scene
    mesh = o3d.geometry.TriangleMesh()
    vertices = []
    triangles = []
    vertex_colors = []
    
    # Traverse the scene to find mesh nodes
    for node in scene.root_node.child_nodes:
        try:
            # Get mesh from node
            if hasattr(node, 'entity') and node.entity is not None:
                entity = node.entity
                if hasattr(entity, 'mesh'):
                    mesh_data = entity.mesh  # type: ignore
                    if mesh_data is not None:
                        # Extract vertices
                        if hasattr(mesh_data, 'control_points'):
                            control_points = mesh_data.control_points
                            for cp in control_points:
                                vertices.append([cp.x, cp.y, cp.z])
                        
                        # Extract triangles/faces
                        if hasattr(mesh_data, 'polygon_count'):
                            # Try to get polygon data
                            pass  # Polygon extraction is complex, will try PLY fallback
        except Exception as e:
            print(f"[Debug] Error extracting from node: {e}")
            continue
    
    # If direct extraction didn't work, try PLY conversion
    if len(vertices) == 0:
        print(f"[Debug] Direct extraction failed, trying PLY conversion...")
        # Save to temporary PLY file
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
            tmp_ply_path = tmp_file.name
        
        try:
            # Save scene as PLY (preserves vertex colors)
            scene.save(tmp_ply_path, a3d.FileFormat.PLY)  # type: ignore
            
            # Check if file was created and has content
            if not Path(tmp_ply_path).exists():
                raise ValueError("PLY file was not created")
            
            file_size = Path(tmp_ply_path).stat().st_size
            if file_size == 0:
                raise ValueError("PLY file is empty")
            
            print(f"[Debug] PLY file created: {tmp_ply_path}, size: {file_size} bytes")
            
            # Try loading with Open3D
            mesh = o3d.io.read_triangle_mesh(tmp_ply_path)
            
            # Check if Open3D loaded correctly (has vertices, not all zeros, AND has triangles)
            vertices_valid = False
            if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                vertices_array = np.asarray(mesh.vertices)
                # Check if vertices are not all at origin
                if not (np.allclose(vertices_array[:, 0], 0.0) and 
                        np.allclose(vertices_array[:, 1], 0.0) and 
                        np.allclose(vertices_array[:, 2], 0.0)):
                    vertices_valid = True
            
            # If Open3D failed to load triangles (even if vertices loaded), use manual parser
            if not vertices_valid or len(mesh.triangles) == 0:
                print(f"[Debug] Open3D failed to load PLY correctly (empty or all zeros), trying manual PLY parsing...")
                # Don't delete temp file yet, we'll use it for manual parsing
                mesh = _load_ply_manual(Path(tmp_ply_path))
            
            # Clean up temp file
            Path(tmp_ply_path).unlink()
            
        except Exception as e:
            # Clean up temp file on error
            if Path(tmp_ply_path).exists():
                Path(tmp_ply_path).unlink()
            print(f"[Debug] PLY conversion failed: {e}")
            # Try OBJ format as fallback
            print(f"[Debug] Trying OBJ format as fallback...")
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
                tmp_obj_path = tmp_file.name
            try:
                scene.save(tmp_obj_path, a3d.FileFormat.WAVEFRONT_OBJ)  # type: ignore
                mesh = o3d.io.read_triangle_mesh(tmp_obj_path)
                Path(tmp_obj_path).unlink()
            except Exception as e2:
                if Path(tmp_obj_path).exists():
                    Path(tmp_obj_path).unlink()
                raise ValueError(f"Failed to convert/load FBX via PLY or OBJ: {e}, {e2}")
    
    if len(mesh.vertices) == 0:
        raise ValueError("Loaded mesh has no vertices")
    
    # Verify vertices are not all zeros
    vertices_array = np.asarray(mesh.vertices)
    if len(vertices_array) > 0:
        vertex_ranges = {
            'x': (np.min(vertices_array[:, 0]), np.max(vertices_array[:, 0])),
            'y': (np.min(vertices_array[:, 1]), np.max(vertices_array[:, 1])),
            'z': (np.min(vertices_array[:, 2]), np.max(vertices_array[:, 2]))
        }
        print(f"[Debug] Vertex ranges after loading: X={vertex_ranges['x']}, Y={vertex_ranges['y']}, Z={vertex_ranges['z']}")
        
        # Check if all vertices are at origin
        if (vertex_ranges['x'][0] == vertex_ranges['x'][1] == 0.0 and
            vertex_ranges['y'][0] == vertex_ranges['y'][1] == 0.0 and
            vertex_ranges['z'][0] == vertex_ranges['z'][1] == 0.0):
            raise ValueError("All vertices are at origin (0,0,0) - FBX file may be corrupted or incorrectly loaded")
    
    print(f"[Info] Successfully loaded FBX mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    if mesh.has_vertex_colors():
        print(f"[Info] Vertex colors detected: {len(mesh.vertex_colors)} colors")
    
    return mesh


def _load_ply_manual(ply_path: Path) -> o3d.geometry.TriangleMesh:
    """Manually parse PLY file if Open3D fails."""
    mesh = o3d.geometry.TriangleMesh()
    vertices = []
    faces = []
    colors = []
    
    try:
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        
        # Find vertex count and face count
        vertex_count = 0
        face_count = 0
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('element face'):
                face_count = int(line.split()[-1])
            elif line.strip() == 'end_header':
                header_end = i + 1
                break
        
        # Read vertices
        for i in range(header_end, header_end + vertex_count):
            if i < len(lines):
                # Replace commas with periods for decimal separator (locale issue)
                line = lines[i].strip().replace(',', '.')
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        vertices.append([x, y, z])
                        # Check for colors (r, g, b after x, y, z)
                        if len(parts) >= 6:
                            r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
                            colors.append([r, g, b])
                    except ValueError as e:
                        print(f"[Warning] Failed to parse vertex line {i}: {e}")
                        continue
        
        # Read faces
        face_start = header_end + vertex_count
        print(f"[Debug] Reading {face_count} faces starting at line {face_start+1}")
        faces_read = 0
        for i in range(face_start, min(face_start + face_count, len(lines))):
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    try:
                        n = int(parts[0])
                        if n == 3:  # Triangle
                            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
                            faces_read += 1
                        elif n > 3:
                            # Polygon with more than 3 vertices - triangulate by using first 3
                            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
                            faces_read += 1
                    except (ValueError, IndexError) as e:
                        if faces_read < 10:  # Only print first few errors
                            print(f"[Warning] Failed to parse face line {i+1}: {e}, parts={parts[:5]}")
                        continue
        
        print(f"[Debug] Successfully read {faces_read} faces out of {face_count} expected")
        
        # Create mesh
        if len(vertices) > 0:
            mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
            if len(faces) > 0:
                mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
            if len(colors) > 0:
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0 if np.max(colors) > 1.0 else np.array(colors))
    
    except Exception as e:
        print(f"[Warning] Manual PLY parsing failed: {e}")
    
    return mesh


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
    # Check if mesh has triangles
    has_triangles = len(mesh.triangles) > 0
    
    if num_points is None:
        # Use all vertices
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
        return pcd
    else:
        if has_triangles:
            # Sample points uniformly from the mesh surface
            return mesh.sample_points_uniformly(number_of_points=num_points)
        else:
            # Mesh has no triangles, so it's already a point cloud
            # Sample from vertices if we need fewer points
            pcd = o3d.geometry.PointCloud()
            vertices = np.asarray(mesh.vertices)
            
            if len(vertices) <= num_points:
                # Use all vertices
                pcd.points = mesh.vertices
            else:
                # Randomly sample vertices
                indices = np.random.choice(len(vertices), size=num_points, replace=False)
                pcd.points = o3d.utility.Vector3dVector(vertices[indices])
            
            # Copy colors if available
            if mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors)
                if len(vertices) <= num_points:
                    pcd.colors = mesh.vertex_colors
                else:
                    pcd.colors = o3d.utility.Vector3dVector(colors[indices])
            
            return pcd

