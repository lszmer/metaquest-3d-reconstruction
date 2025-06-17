from pathlib import Path
import numpy as np
import cv2


class ImageRepository:
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir


    @property
    def paths(self) -> list[Path]:
        return sorted(self.image_dir.glob("*.png"))
    

    def save(self, file_stem: str, image: np.ndarray):
        self.image_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.image_dir / f"{file_stem}.png"
        cv2.imwrite(str(output_path), image)