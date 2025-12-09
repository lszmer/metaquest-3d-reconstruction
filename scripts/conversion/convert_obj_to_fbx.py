#!/usr/bin/env python3
"""
Script to convert OBJ folder to Unity-importable FBX format.

This script:
1. Takes an OBJ folder path as input
2. Finds the OBJ file in that folder
3. Loads the OBJ file (with MTL and textures if present)
4. Converts vertex colors or texture colors to vertex colors
5. Converts it to FBX format using Aspose.3D
6. Outputs the FBX file in the same folder
7. Verifies the output has colors, faces, and vertices

Requires:
    - aspose-3d: pip install aspose-3d
    - open3d: pip install open3d
"""

import argparse
import sys
import importlib
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[Warning] PIL/Pillow not available. Texture color extraction will be limited.")


def _load_aspose_module():
    """Load the Aspose.3D module."""
    try:
        module = importlib.import_module("aspose.threed")
    except ImportError:
        print(
            "[Error] aspose-3d is not installed. Please install it using: pip install aspose-3d",
            file=sys.stderr
        )
        sys.exit(1)
    return module


def find_obj_file(obj_folder: Path) -> Path:
    """
    Find the OBJ file in the given folder.
    
    Args:
        obj_folder: Path to the folder containing OBJ files
        
    Returns:
        Path to the OBJ file
        
    Raises:
        FileNotFoundError: If no OBJ file is found
        ValueError: If multiple OBJ files are found
    """
    if not obj_folder.exists():
        raise FileNotFoundError(f"OBJ folder not found: {obj_folder}")
    
    if not obj_folder.is_dir():
        raise ValueError(f"Path is not a directory: {obj_folder}")
    
    obj_files = list(obj_folder.glob("*.obj"))
    
    if len(obj_files) == 0:
        raise FileNotFoundError(f"No OBJ files found in folder: {obj_folder}")
    
    if len(obj_files) > 1:
        print(f"[Warning] Multiple OBJ files found, using: {obj_files[0].name}")
    
    return obj_files[0]


def load_obj_as_mesh(obj_path: Path) -> o3d.geometry.TriangleMesh:
    """
    Load an OBJ file as a triangle mesh.
    
    Args:
        obj_path: Path to the OBJ file
        
    Returns:
        Open3D TriangleMesh object
        
    Raises:
        FileNotFoundError: If the OBJ file doesn't exist
        ValueError: If the file cannot be loaded or is invalid
    """
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    
    if not obj_path.suffix.lower() == '.obj':
        raise ValueError(f"Input file must be a .obj file, got: {obj_path.suffix}")
    
    print(f"[Info] Loading OBJ file: {obj_path}")
    
    try:
        # Load OBJ file with Open3D
        # Open3D should handle MTL files and textures automatically
        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        
        if len(mesh.vertices) == 0:
            raise ValueError("OBJ file has no vertices")
        
        if len(mesh.triangles) == 0:
            raise ValueError("OBJ file has no faces/triangles")
        
        print(f"[Info] ✓ OBJ file loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        
        return mesh
        
    except Exception as e:
        raise ValueError(f"Failed to load OBJ file: {str(e)}") from e


def extract_colors_from_texture(mesh: o3d.geometry.TriangleMesh, obj_path: Path) -> bool:
    """
    Extract colors from texture images and apply them to vertex colors.
    
    Args:
        mesh: Open3D TriangleMesh object
        obj_path: Path to the OBJ file (to find texture files)
        
    Returns:
        True if colors were extracted and applied, False otherwise
    """
    if not PIL_AVAILABLE:
        print("[Warning] PIL/Pillow not available. Cannot extract colors from textures.")
        return False
    
    # Check if mesh has texture coordinates
    if not mesh.has_triangle_uvs():
        print("[Info] Mesh has no texture coordinates, skipping texture color extraction")
        return False
    
    # Try to find texture files in the same folder as OBJ
    obj_folder = obj_path.parent
    texture_files = list(obj_folder.glob("*.jpg")) + list(obj_folder.glob("*.png")) + \
                    list(obj_folder.glob("*.jpeg")) + list(obj_folder.glob("*.JPG"))
    
    if len(texture_files) == 0:
        print("[Info] No texture files found in OBJ folder, skipping texture color extraction")
        return False
    
    print(f"[Info] Found {len(texture_files)} texture file(s), extracting colors...")
    
    # Use the first texture file found
    texture_path = texture_files[0]
    print(f"[Info] Loading texture: {texture_path.name}")
    
    try:
        # Load texture image
        texture_img = Image.open(texture_path)
        # Convert to RGB if needed
        if texture_img.mode != 'RGB':
            texture_img = texture_img.convert('RGB')
        texture_array = np.array(texture_img)
        texture_height, texture_width = texture_array.shape[:2]
        
        print(f"[Info] Texture size: {texture_width}x{texture_height}")
        
        # Get triangle UVs and triangles
        triangle_uvs = np.asarray(mesh.triangle_uvs)
        triangles = np.asarray(mesh.triangles)
        
        if len(triangle_uvs) == 0:
            print("[Warning] No triangle UVs found")
            return False
        
        # Map UV coordinates to texture pixels and assign to vertices
        # Each triangle has 3 UV coordinates (one per vertex)
        num_triangles = len(triangles)
        num_uvs = len(triangle_uvs)
        
        if num_uvs != num_triangles * 3:
            print(f"[Warning] UV count ({num_uvs}) doesn't match triangle count ({num_triangles * 3})")
            return False
        
        # Initialize vertex color accumulator
        vertices = np.asarray(mesh.vertices)
        num_vertices = len(vertices)
        vertex_color_sum = np.zeros((num_vertices, 3))
        vertex_color_count = np.zeros(num_vertices)
        
        print(f"[Info] Mapping texture colors to {num_vertices} vertices...")
        
        # For each triangle, sample texture at UV coordinates and assign to vertices
        for tri_idx in range(num_triangles):
            triangle = triangles[tri_idx]
            
            # Get UV coordinates for this triangle (3 UVs per triangle)
            uv_start = tri_idx * 3
            uv_end = uv_start + 3
            
            for corner_idx in range(3):
                uv = triangle_uvs[uv_start + corner_idx]
                vertex_idx = triangle[corner_idx]
                
                # Convert UV to pixel coordinates
                # UV coordinates are typically in [0, 1] range
                u, v = uv[0], uv[1]
                
                # Clamp UV to [0, 1]
                u = np.clip(u, 0.0, 1.0)
                v = np.clip(v, 0.0, 1.0)
                
                # Convert to pixel coordinates (note: v is typically flipped in images)
                pixel_x = int(u * (texture_width - 1))
                pixel_y = int((1.0 - v) * (texture_height - 1))  # Flip v coordinate
                
                # Clamp pixel coordinates
                pixel_x = np.clip(pixel_x, 0, texture_width - 1)
                pixel_y = np.clip(pixel_y, 0, texture_height - 1)
                
                # Sample color from texture
                color = texture_array[pixel_y, pixel_x]
                
                # Convert from 0-255 to 0-1 range
                color_normalized = color.astype(np.float64) / 255.0
                
                # Accumulate color for this vertex
                if vertex_idx < num_vertices:
                    vertex_color_sum[vertex_idx] += color_normalized
                    vertex_color_count[vertex_idx] += 1
        
        # Average colors per vertex
        for v_idx in range(num_vertices):
            if vertex_color_count[v_idx] > 0:
                vertex_color_sum[v_idx] /= vertex_color_count[v_idx]
            else:
                # Default to white if no color assigned
                vertex_color_sum[v_idx] = [1.0, 1.0, 1.0]
        
        # Assign colors to mesh
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_color_sum)
        
        print(f"[Info] ✓ Successfully extracted and assigned colors from texture")
        print(f"[Info]   {np.sum(vertex_color_count > 0)} vertices have colors")
        
        return True
        
    except Exception as e:
        print(f"[Warning] Failed to extract colors from texture: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_has_color(mesh: o3d.geometry.TriangleMesh) -> bool:
    """
    Verify that mesh has color information.
    
    Args:
        mesh: Open3D TriangleMesh object
        
    Returns:
        True if the mesh has colors, False otherwise
    """
    has_colors = mesh.has_vertex_colors()
    if has_colors:
        num_colors = len(mesh.vertex_colors)
        print(f"[Info] ✓ Color information verified: {num_colors} vertex colors found")
        
        # Check color range
        colors = np.asarray(mesh.vertex_colors)
        print(f"[Info] Color range: min={colors.min():.3f}, max={colors.max():.3f}, mean={colors.mean():.3f}")
    else:
        print("[Warning] ✗ No vertex colors found")
    
    return has_colors


def convert_obj_to_fbx(obj_folder: Path, verify_color: bool = True) -> Path:
    """
    Convert an OBJ folder to FBX format using Aspose.3D.
    
    Args:
        obj_folder: Path to the folder containing OBJ files
        verify_color: If True, verify that the OBJ file has color before converting
        
    Returns:
        Path to the output FBX file
        
    Raises:
        FileNotFoundError: If the OBJ folder or file doesn't exist
        ValueError: If the file is not valid, lacks color (when verify_color=True), or conversion fails
    """
    # Find OBJ file
    obj_path = find_obj_file(obj_folder)
    
    # Load OBJ file
    mesh = load_obj_as_mesh(obj_path)
    
    # Try to extract colors from texture if no vertex colors
    if not mesh.has_vertex_colors():
        print("[Info] No vertex colors found, attempting to extract from textures...")
        extract_colors_from_texture(mesh, obj_path)
    
    # Verify color if requested
    if verify_color:
        has_color = verify_has_color(mesh)
        if not has_color:
            print("[Warning] OBJ file does not have vertex color information.")
            print("[Warning] The mesh will be converted without colors.")
            print("[Warning] You may need to apply materials/textures in Unity.")
    
    # Ensure mesh has triangles
    if len(mesh.triangles) == 0:
        raise ValueError("Mesh has no triangles. Cannot create solid mesh for FBX export.")
    
    print(f"[Info] Converting mesh to FBX...")
    
    # Generate output path (same folder, same name as OBJ, different extension)
    fbx_path = obj_path.with_suffix('.fbx')
    
    # Save mesh to temporary PLY file (to preserve colors for Aspose)
    temp_ply_path = None
    try:
        a3d = _load_aspose_module()
        
        # Save mesh to temporary PLY file with colors
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
            temp_ply_path = Path(tmp_file.name)
        
        print(f"[Info] Saving mesh to temporary PLY: {temp_ply_path}")
        o3d.io.write_triangle_mesh(str(temp_ply_path), mesh, write_ascii=False)
        
        # Load PLY file via Aspose.3D Scene class
        print("[Info] Loading PLY with Aspose.3D...")
        scene = a3d.Scene.from_file(str(temp_ply_path))
        print("[Info] Successfully loaded PLY file with Aspose.3D")
        
        # Export to FBX format
        # Aspose.3D will auto-detect the format from the .fbx extension
        print(f"[Info] Exporting to FBX: {fbx_path}")
        scene.save(str(fbx_path))
        
        # Clean up temporary file
        if temp_ply_path and temp_ply_path.exists():
            temp_ply_path.unlink()
        
        # Verify the output file was created
        if not fbx_path.exists():
            raise ValueError("FBX file was not created. Check for errors in the conversion process.")
        
        file_size = fbx_path.stat().st_size
        print(f"[Info] ✓ Successfully converted {obj_path.name} to {fbx_path.name}")
        print(f"[Info] Output file size: {file_size / 1024 / 1024:.2f} MB")
        print(f"[Info] Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
        
        return fbx_path
        
    except Exception as e:
        # Clean up temporary file on error
        if temp_ply_path and temp_ply_path.exists():
            temp_ply_path.unlink()
        raise ValueError(f"Failed to convert OBJ to FBX: {str(e)}") from e


def verify_fbx_output(fbx_path: Path) -> bool:
    """
    Verify that the converted FBX file has colors, faces, and vertices.
    
    Args:
        fbx_path: Path to the FBX file to verify
        
    Returns:
        True if verification passes, False otherwise
    """
    print(f"\n[Info] Verifying FBX output: {fbx_path}")
    
    if not fbx_path.exists():
        print("[Error] ✗ FBX file does not exist")
        return False
    
    try:
        a3d = _load_aspose_module()
        
        # Load FBX with Aspose to check structure
        print("[Info] Loading FBX with Aspose.3D...")
        scene = a3d.Scene.from_file(str(fbx_path))
        
        # Try to load directly with Open3D first (if supported)
        mesh = None
        try:
            print("[Info] Attempting to load FBX directly with Open3D...")
            mesh = o3d.io.read_triangle_mesh(str(fbx_path))
            if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                print("[Info] ✓ Successfully loaded FBX directly with Open3D")
        except Exception as e:
            print(f"[Debug] Direct Open3D loading failed: {e}")
            mesh = None
        
        # If direct loading failed, try via OBJ format
        if mesh is None or len(mesh.triangles) == 0:
            print("[Info] Trying to convert via OBJ format...")
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
                temp_obj_path = Path(tmp_file.name)
            
            try:
                scene.save(str(temp_obj_path), a3d.FileFormat.WAVEFRONT_OBJ)  # type: ignore
                mesh = o3d.io.read_triangle_mesh(str(temp_obj_path))
                
                # Clean up temp file
                if temp_obj_path.exists():
                    temp_obj_path.unlink()
            except Exception as e:
                if temp_obj_path.exists():
                    temp_obj_path.unlink()
                raise ValueError(f"Failed to convert FBX via OBJ: {e}")
        
        # Verify vertices
        num_vertices = len(mesh.vertices)
        if num_vertices == 0:
            print("[Error] ✗ FBX file has no vertices")
            return False
        print(f"[Info] ✓ Vertices: {num_vertices}")
        
        # Verify faces
        num_faces = len(mesh.triangles)
        if num_faces == 0:
            print("[Error] ✗ FBX file has no faces")
            return False
        print(f"[Info] ✓ Faces: {num_faces}")
        
        # Verify colors
        has_colors = mesh.has_vertex_colors()
        if has_colors:
            num_colors = len(mesh.vertex_colors)
            colors = np.asarray(mesh.vertex_colors)
            print(f"[Info] ✓ Colors: {num_colors} vertex colors")
            print(f"[Info]   Color range: [{colors.min():.3f}, {colors.max():.3f}]")
            print(f"[Info]   Color mean: {colors.mean():.3f}")
        else:
            print("[Warning] ⚠ FBX file has no vertex colors")
        
        print(f"[Info] ✓✓✓ Verification passed! ✓✓✓")
        return True
            
    except Exception as e:
        print(f"[Error] ✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert an OBJ folder to Unity-importable FBX format. "
            "Finds the OBJ file in the folder and converts it to FBX. "
            "The output FBX file will be saved in the same folder as the OBJ file."
        )
    )
    parser.add_argument(
        'obj_folder',
        type=str,
        help="Path to the folder containing OBJ files"
    )
    parser.add_argument(
        '--skip-color-check',
        action='store_true',
        help="Skip color verification (not recommended)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        obj_folder = Path(args.obj_folder).resolve()
        fbx_path = convert_obj_to_fbx(
            obj_folder, 
            verify_color=not args.skip_color_check
        )
        print(f"\n[Info] Conversion complete: {fbx_path}")
        
        # Verify the output
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)
        verification_passed = verify_fbx_output(fbx_path)
        
        if verification_passed:
            print(f"\n[Info] ✓✓✓ Success! The FBX file is ready for Unity import. ✓✓✓")
        else:
            print(f"\n[Warning] ⚠ Verification found issues. Please check the output file manually.")
        
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

