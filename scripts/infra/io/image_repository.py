from pathlib import Path
import numpy as np
import cv2


class ImageRepository:
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir


    @property
    def paths(self) -> list[Path]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist.")
        
        return sorted(self.image_dir.glob("*.png"))


    def get_relaive_path(self, file_stem: str) -> str:
        depth_map_path = self.depth_dir / f"{file_stem}.raw"
        return depth_map_path.relative_to(self.project_root)
        

    def save(self, file_stem: str, image: np.ndarray):
        self.image_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.image_dir / f"{file_stem}.png"
        cv2.imwrite(str(output_path), image)