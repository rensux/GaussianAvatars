
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData

import blend
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz


class BlendGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.faces = self.flame_model.faces
        self.verts = None
        self.blend_model: blend.Model = None


        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()


    def update_mesh_by_blending(self, blend_param):
        coefficients = blend_param['coefficients']

        v = self.blend_model.calc_mesh(coefficients)
        print(v.shape)
        verts = torch.from_numpy(v).cuda()
        cano = torch.from_numpy(self.blend_model.cano_verts).cuda()
        self.update_mesh_properties(verts, cano)

    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}


            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers


        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]

        self.verts, canno = self.flame_model(
            self.flame_param['shape'][None, ...],
            self.flame_param['expr'][[0]],
            self.flame_param['rotation'][[0]],
            self.flame_param['neck_pose'][[0]],
            self.flame_param['jaw_pose'][[0]],
            self.flame_param['eyes_pose'][[0]],
            self.flame_param['translation'][[0]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=self.flame_param['static_offset'],
            dynamic_offset=self.flame_param['dynamic_offset'][[0]],
        )

        self.update_mesh_properties(self.verts, canno)

    def load_meshes(self, main:str, blends:list[str]):
        if blends is None:
            blends = []
        plydata = PlyData.read(main)

        tris = np.concatenate(np.asarray(plydata["face"]["vertex_indices"]))
        m = len(blends)+1
        vs = plydata["vertex"].count
        mesh_verts = np.ndarray(shape=(m, vs, 3), dtype=np.float32)
        mesh_verts[0] = np.stack((
            np.asarray(plydata["vertex"]["x"]),
            np.asarray(plydata["vertex"]["y"]),
            np.asarray(plydata["vertex"]["z"])), axis=1)
        for i in range(1,m):
            plydata = PlyData.read(blends[i-1])
            mesh_verts[i] = np.stack((
                np.asarray(plydata["vertex"]["x"]),
                np.asarray(plydata["vertex"]["y"]),
                np.asarray(plydata["vertex"]["z"])), axis=1)
        self.blend_model = blend.Model(tris, mesh_verts)

