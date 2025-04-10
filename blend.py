import time
from dataclasses import dataclass, field
from pathlib import Path

import scipy.sparse as sp
import tyro
import numpy as np
import torch

import blend
from flame_model.flame import FlameHead


@dataclass
class Config:
    blend_model_paths: list[Path] = field(default_factory=lambda : [Path("./media/253/flame_param/00000.npz"),Path("./media/460/flame_param/00000.npz")])
    main_model_path: Path = Path("media/teeth+eyes/306/flame_param.npz")
    MT_matrix_path: Path = Path("./media/flame_laplace_MT.npz")


def load_mesh_from_flame(flame_model: FlameHead, path: Path):
    flame_param = np.load(str(path))
    flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}
    return flame_model(
        flame_param['shape'][None, ...],
        flame_param['expr'][[0]],
        flame_param['rotation'][[0]],
        flame_param['neck_pose'][[0]],
        flame_param['jaw_pose'][[0]],
        flame_param['eyes_pose'][[0]],
        flame_param['translation'][[0]],
        zero_centered_at_root_node=False,
        return_landmarks=False,
        # static_offset=flame_param['static_offset'],
    )

def main(cfg: Config, n_shape=300, n_expr=100):

    x = time.time()

    flame_model = FlameHead(
        n_shape,
        n_expr,
        add_teeth=False,
    )

    tris = flame_model.faces.numpy()
    flame_model = flame_model.cuda()

    meshes = np.ndarray((len(cfg.blend_model_paths)+1, flame_model.v_template.shape[0], 3), dtype=np.float32)
    meshes[0] = load_mesh_from_flame(flame_model, cfg.main_model_path).cpu().numpy()
    for i, p in enumerate(cfg.blend_model_paths):
        meshes[i+1] = load_mesh_from_flame(flame_model, p).cpu().numpy()

    model = blend.Model(tris, meshes, cfg.MT_matrix_path)

    y = time.time()
    print(f"setup took {y-x}s\n")

    cfs = np.array([0.2,0.4,0.4], dtype=np.float32)
    model.calc_mesh(cfs)

    print("\n=======================\n")

    cfs = np.array([0.2,0.4,0.4], dtype=np.float32)
    model.calc_mesh(cfs)

    print("\n=======================\n")
    cfs = np.array([0.2,0.4,0.4], dtype=np.float32)
    model.calc_mesh(cfs)

    print("\n=======================\n")
    cfs = np.array([0.2,0.4,0.4], dtype=np.float32)
    model.calc_mesh(cfs)

    print("\n=======================\n")
    cfs = np.array([0.2,0.4,0.4], dtype=np.float32)
    model.calc_mesh(cfs)

    print("\n=======================\n")

    cfs = np.array([0.9,0.,0.1], dtype=np.float32)
    model.calc_mesh(cfs)

    print("\n=======================\n")

    cfs = np.array([1.,0.,0.], dtype=np.float32)
    model.calc_mesh(cfs)



if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
