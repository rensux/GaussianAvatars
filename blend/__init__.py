import time
from pathlib import Path

import scipy.sparse as sp
import numpy as np
from numpy.typing import NDArray
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import SuperLU

from blend.mask import Mask


class Model:
    MT: sp.csc_matrix # V x E
    MTM: sp.csc_matrix # V x V
    cano_edges: NDArray[np.float32] # M * E * 3
    vert_edge_binding: sp.csc_matrix # V x E
    LU: SuperLU
    mask: Mask


    def __init__(self, tris: NDArray[np.int32], mesh_verts: NDArray[np.float32], mask:Mask, MT_matrix_path: Path|None = None):
        self.mask = mask
        self.load_MT_MTM(tris, mesh_verts.shape[1], MT_matrix_path)

        m = mesh_verts.shape[0]
        cano_edges = np.ndarray(shape=(m, tris.size, 3), dtype=np.float32)
        for i in range(mesh_verts.shape[0]): # for each mesh
            cano_edges[i]=calc_edges(tris, mesh_verts[i])
        self.cano_edges = cano_edges

# M x E x 3
# M x 1

# M x E x 3
# M x E

    def calc_weighted_edges(self, vert_coeffs: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        vert_coeffs: MxV
        returns  MxV * VxE = sum(MxE element_mult MxEx3, axis=0) = Ex3
        """
        edge_weights = vert_coeffs @ self.vert_edge_binding # MxE
        return (edge_weights[:,:,None] * self.cano_edges).sum(axis=0)

    def calc_mesh(self, coefficients: NDArray[np.float32]):
        """
        coeffs: Mx1
        returns 1xM * MxEx3 = Ex3
        """
        d = self.calc_weighted_edges(coefficients) # calc blended edges
        MTd = self.MT.dot(d) # calc MTd
        # Run each column in parallel using cuda streams
        return self.LU.solve(MTd)

    def load_MT_MTM(self, tris: NDArray[np.int32], vs: int, MT_matrix_path: Path | None = None):
        if MT_matrix_path is not None and MT_matrix_path.exists():
            self.MT = sp.load_npz(str(MT_matrix_path))
        else:
            self.MT = incidence_matrix(tris, vs)
        self.vert_edge_binding = self.MT.copy()
        self.vert_edge_binding.data = (self.vert_edge_binding.data+1) / 2.
        self.vert_edge_binding.eliminate_zeros()
        self.LU = spla.spilu(self.MT.dot(self.MT.T))
        if MT_matrix_path is not None and not MT_matrix_path.exists():
            sp.save_npz(str(MT_matrix_path), self.MT)

def calc_edges(tris: NDArray[np.int32], verts: NDArray[np.int32]):
    edges = np.zeros((tris.size, 3), dtype=np.float32)
    for i in range(tris.shape[0]):
        v0, v1, v2 = verts[tris[i, 0]], verts[tris[i, 1]], verts[tris[i, 2]]
        e = i * 3
        edges[e] = v1 - v0
        edges[e + 1] = v2 - v1
        edges[e + 2] = v0 - v2
    return edges


def incidence_matrix(ts: NDArray[np.int32], vs:int) -> sp.csc_matrix:
    """
    VxE matrix where each column represents the incidence of a vector
    (v,e) = -1 means edge e leaves vertex v
    (u,e) = 1 means edge e enters vertex u
    """
    MT = sp.dok_matrix((vs, ts.size), dtype=np.float32)

    for i in range(ts.shape[0]): # 0 - 9976
        for j in range(ts.shape[1]):
            e = i*3+j
            v = ts[i,j]
            u = ts[i,(j+1)%3]
            MT[v, e] = -1
            MT[u, e] = 1

    return MT.tocsc()
