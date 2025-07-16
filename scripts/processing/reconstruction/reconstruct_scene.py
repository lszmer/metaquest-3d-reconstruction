import open3d as o3d

from dataio.depth_data_io import DepthDataIO
from processing.reconstruction.make_fragments import make_fragment_datasets
from processing.reconstruction.o3d_utils import integrate


def log_step(title: str):
    print("\n" + "="*40)
    print(f">>> [Step] {title}")
    print("="*40)


def reconstruct_scene(depth_data_io: DepthDataIO):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    log_step("Make Fragments")
    frag_dataset_map = make_fragment_datasets(depth_data_io=depth_data_io, use_cache=True)

    print("[Info] Visualizing the generated point cloud...")
    fragments = []
    for side, frag_datasets in frag_dataset_map.items():
        for frag_dataset in frag_datasets:
            vgb = integrate(frag_dataset, depth_data_io, side, 0.01, 16, 50_000, 1.5, 8.0, o3d.core.Device("CUDA:0"))
            fragments.append(vgb.extract_point_cloud())

    legacy_fragments = [f.to_legacy() for f in fragments]

    o3d.visualization.draw_geometries(legacy_fragments + [axis], window_name="Generated Point Cloud")