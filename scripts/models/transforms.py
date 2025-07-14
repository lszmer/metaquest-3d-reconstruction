from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R


class CoordinateSystem(Enum):
    """
    Enum representing different coordinate systems used in 3D graphics and computer vision.
    
    - UNITY:
        - World: Y-up, left-handed
        - Camera: X-right, Y-up, Z-forward
        - Used in Unity3D engine
    - OPEN3D:
        - World: Y-up, right-handed
        - Camera: X-right, Y-down, Z-forward
        - Used in Open3D
    - NERFSTUDIO:
        - World: Z-up, right-handed
        - Camera: X-right, Y-up, Z-backward
        - Used in NerfStudio
    - COLMAP:
        - World: Y-down, right-handed
        - Camera: X-right, Y-down, Z-forward
        - Used in COLMAP
    """
    UNITY = "Unity"
    OPEN3D = "Open3D"
    NERFSTUDIO = "NerfStudio"
    COLMAP = "COLMAP"


class ExtrinsicMode(Enum):
    CameraToWorld = "camera_to_world"
    WorldToCamera = "world_to_camera"



@dataclass
class Transforms:
    coordinate_system: CoordinateSystem

    positions: np.ndarray # shape=(N, 3), axis1=(x, y, z)
    rotations: np.ndarray # shape=(N, 4), axis1=(x, y, z, w)


    @property
    def extrinsics_wc(self) -> np.ndarray:
        return self.to_extrinsic_matrices(mode=ExtrinsicMode.WorldToCamera)


    @property
    def extrinsics_cw(self) -> np.ndarray:
        return self.to_extrinsic_matrices(mode=ExtrinsicMode.CameraToWorld)


    def get_coordinate_transform_matrix(self, source: CoordinateSystem, target: CoordinateSystem) -> np.ndarray:
        def basis(cs: CoordinateSystem) -> np.ndarray:
            if cs == CoordinateSystem.UNITY:        # X-right, Y-up, Z-forward (L-handed)
                return np.eye(3)
            elif cs == CoordinateSystem.OPEN3D:     # X-right, Y-up, Z-backward (R-handed)
                return np.diag((1, 1, -1))
            elif cs == CoordinateSystem.NERFSTUDIO: # X-right, Y-forward, Z-up (R-handed)
                return np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]) 
            elif cs == CoordinateSystem.COLMAP:     # X-right, Y-down, Z-forward (R-handed)
                return np.diag((1, -1, 1))
            else:
                raise ValueError(f"Unknown coordinate system: {cs}")
            
        R_source = basis(source)
        R_target = basis(target)

        return R_target @ R_source.T


    def get_camera_basis_matrix(self, cs: CoordinateSystem) -> np.ndarray:
        if cs == CoordinateSystem.UNITY:        # World:  X-right, Y-up, Z-forward
            return np.eye(3)                    # Camera: X-right, Y-up, Z-forward
        elif cs == CoordinateSystem.OPEN3D:     # World:  X-right, Y-up, Z-backward
            return np.diag((1, -1, -1))         # Camera: X-right, Y-down, Z-forward
        elif cs == CoordinateSystem.NERFSTUDIO:                 # World:  X-right, Y-forward, Z-up
            return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) # Camera: X-right, Y-up, Z-backward
        elif cs == CoordinateSystem.COLMAP:     # World:  X-right, Y-down, Z-forward
            return np.eye(3)                    # Camera: X-right, Y-down, Z-forward
        else:
            raise ValueError(f"Unknown coordinate system: {cs}")


    def convert_coordinate_system(
        self,
        target_coordinate_system: CoordinateSystem,
        is_camera: bool = False
    ) -> 'Transforms':
        if self.coordinate_system == target_coordinate_system:
            return self

        R_conv = self.get_coordinate_transform_matrix(self.coordinate_system, target_coordinate_system)  # shape (3,3)

        # Apply to positions (world transformation)
        converted_positions = (R_conv @ self.positions.T).T  # (N, 3)

        # Apply to rotations
        rotation_matrices = R.from_quat(self.rotations).as_matrix()  # (N, 3, 3)

        if is_camera:
            source_basis_matrix = self.get_camera_basis_matrix(self.coordinate_system)
            rotation_matrices = rotation_matrices @ source_basis_matrix.T

        converted_rotations = R_conv @ rotation_matrices @ R_conv.T  # (N, 3, 3)

        if is_camera:
            target_basis_matrix = self.get_camera_basis_matrix(target_coordinate_system)
            converted_rotations = converted_rotations @ target_basis_matrix

        return Transforms(
            coordinate_system=target_coordinate_system,
            positions=converted_positions,
            rotations=R.from_matrix(converted_rotations).as_quat()
        )
        

    def to_extrinsic_matrices(self, mode: ExtrinsicMode = ExtrinsicMode.WorldToCamera) -> np.ndarray:
        N = len(self.positions)

        R_cw = R.from_quat(self.rotations).as_matrix()  # (N, 3, 3)

        extrinsic_matrices = np.zeros((N, 4, 4), dtype=np.float32)
        extrinsic_matrices[:, :3, :3] = R_cw
        extrinsic_matrices[:, :3, 3] = self.positions
        extrinsic_matrices[:, 3, 3] = 1.0

        if mode == ExtrinsicMode.WorldToCamera:
            return np.linalg.inv(extrinsic_matrices)
        elif mode == ExtrinsicMode.CameraToWorld:
            return extrinsic_matrices
        else:
            raise ValueError(f"Unsupported extrinsic mode: {mode}")


    def compose_transform(
        self,
        local_position: np.ndarray, # shape=(3), axis1=(x, y, z)
        local_rotation: np.ndarray, # shape=(4), axis1=(x, y, z, w)
    ) -> 'Transforms':
        parent_rotations = R.from_quat(self.rotations)
        rotated_local_positions = parent_rotations.apply(local_position)

        world_positions = self.positions + rotated_local_positions
        world_rotations = parent_rotations * R.from_quat(local_rotation)

        return Transforms(
            coordinate_system=self.coordinate_system,
            positions=world_positions,
            rotations=world_rotations.as_quat()
        )
    

    def to_dict(self) -> dict:
        d = {
            "coordinate_system": self.coordinate_system,
            "positions": self.positions,
            "rotations": self.rotations,
        }

        return d
    

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            **self.to_dict()
        )


    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    

    @classmethod
    def load(cls, path: Path):
        data = dict(np.load(path, allow_pickle=False))
        return cls.from_dict(data=data)