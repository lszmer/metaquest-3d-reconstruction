#!/usr/bin/env python3
"""
Utility script to convert PLY files to FBX format using Aspose.3D for Python.
Takes a PLY file path as argument and saves the FBX file in the same folder.

Uses Aspose.3D library for reliable FBX export with vertex color support.
Requires aspose-3d to be installed: pip install aspose-3d
"""

import argparse
import sys
import importlib
from pathlib import Path
from typing import Optional


def _load_aspose_module():
    try:
        module = importlib.import_module("aspose.threed")
    except ImportError:
        print("[Error] aspose-3d is not installed. Please install it using: pip install aspose-3d", file=sys.stderr)
        sys.exit(1)
    return module


def convert_ply_to_fbx_aspose(ply_path: Path) -> Path:
    """
    Convert a PLY file to FBX format using Aspose.3D.
    
    Args:
        ply_path: Path to the input PLY file
        
    Returns:
        Path to the output FBX file
        
    Raises:
        FileNotFoundError: If the PLY file doesn't exist
        ValueError: If the file is not a valid PLY file or conversion fails
    """
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    if not ply_path.suffix.lower() == '.ply':
        raise ValueError(f"Input file must be a .ply file, got: {ply_path.suffix}")
    
    print(f"[Info] Loading PLY file: {ply_path}")
    
    # Generate output path (same folder, same name, different extension)
    fbx_path = ply_path.with_suffix('.fbx')
    
    try:
        a3d = _load_aspose_module()
        # Step 1: Load PLY file via the from_file of Scene class
        scene = a3d.Scene.from_file(str(ply_path))
        print("[Info] Successfully loaded PLY file")
        
        # Step 2: Create an instance of FbxSaveOptions
        # Step 3: Set FBX specific properties for advanced conversion
        # (For now, we'll save without explicit options - format auto-detected from extension)
        
        # Step 4: Call the Scene.save method
        # Step 5: Pass the output path with FBX file extension
        # Aspose.3D will auto-detect the format from the .fbx extension
        print(f"[Info] Exporting to FBX: {fbx_path}")
        scene.save(str(fbx_path))
        
        # Step 6: Check resultant FBX file at specified path
        if not fbx_path.exists():
            raise ValueError("FBX file was not created. Check for errors in the conversion process.")
        
        file_size = fbx_path.stat().st_size
        print(f"[Info] Successfully converted {ply_path.name} to {fbx_path.name}")
        print(f"[Info] Output file size: {file_size / 1024 / 1024:.2f} MB")
        
        return fbx_path
        
    except Exception as e:
        raise ValueError(f"Failed to convert PLY to FBX: {str(e)}") from e


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a PLY file to FBX format using Aspose.3D. "
                    "The output FBX file will be saved in the same folder as the input PLY file."
    )
    parser.add_argument(
        'ply_path',
        type=str,
        help="Path to the input PLY file"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        ply_path = Path(args.ply_path).resolve()
        fbx_path = convert_ply_to_fbx_aspose(ply_path)
        print(f"[Info] Conversion complete: {fbx_path}")
        
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

