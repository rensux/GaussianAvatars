import time
from pathlib import Path

import scipy.sparse as sp
import numpy as np
from numpy.typing import NDArray
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csla
import cupy as cp
import scipy.sparse.linalg as spla


class Model:
    MT: csp.csc_matrix
    MTM: csp.csc_matrix
    cano_edges: NDArray[np.float32] # M * E * 3
    cano_verts: NDArray[np.float32] # V * 3
    verts: NDArray[np.float32] # V * 3
    P: csp.spmatrix


    def __init__(self, tris: NDArray[np.int32], mesh_verts: NDArray[np.float32], MT_matrix_path: Path|None = None):
        self.cano_verts = cp.asarray(mesh_verts[0])
        self.verts = self.cano_verts.copy()
        self.load_MT_MTM(tris, self.verts.shape[0], MT_matrix_path)

        m = mesh_verts.shape[0]
        cano_edges = np.ndarray(shape=(m, tris.size, 3), dtype=np.float32)
        for i in range(mesh_verts.shape[0]): # for each mesh
            cano_edges[i]=calc_edges(tris, mesh_verts[i])
        self.cano_edges = cp.asarray(cano_edges)

    def calc_cpu(self, coefficients: NDArray[np.float32]):
        d = cp.tensordot(coefficients, self.cano_edges, axes=1) # calc blended edges
        MTd = self.MT.dot(d).get()
        MTM = self.MTM.get()
        verts = self.cano_verts.get()

        x = time.time()
        v = np.column_stack([spla.cg(MTM, MTd[:, i], x0=verts[:,i])[0] for i in range(MTd.shape[1])])
        y = time.time()
        print(f"cpu calc took {y - x}s")


    def calc_mesh(self, coefficients: NDArray[np.float32]):
        print(f"calcing mesh with coefficients: {coefficients}")
        d = cp.tensordot(coefficients, self.cano_edges, axes=1) # calc blended edges
        MTd = self.MT.dot(d) # calc MTd
        # Run each column in parallel using cuda streams
        streams = [cp.cuda.Stream() for _ in range(3)]
        x = time.time()
        for i in range(3):
            with streams[i]:
               csla.cg(self.MTM, MTd[:,i], x0=self.cano_verts[:,i])
        # Wait for all streams to complete
        for stream in streams:
            stream.synchronize()
        y = time.time()
        print(f"gpu calc took {y-x}s")

        self.calc_cpu(coefficients)

    def load_MT_MTM(self, tris: NDArray[np.int32], vs: int, MT_matrix_path: Path | None = None):
        if MT_matrix_path is not None and MT_matrix_path.exists():
            MT = sp.load_npz(str(MT_matrix_path))
        else:
            MT = incidence_matrix(tris, vs)
        self.MT = csp.csr_matrix(MT)
        self.MTM = self.MT.dot(self.MT.T)
        self.P = csp.diags(1./self.MTM.diagonal())
        if MT_matrix_path is not None and not MT_matrix_path.exists():
            sp.save_npz(str(MT_matrix_path), MT)

def calc_edges(tris: NDArray[np.int32], verts: NDArray[np.int32]):
    edges = np.zeros((tris.size, 3), dtype=np.float32)
    for i in range(tris.shape[0]):
        v0, v1, v2 = verts[tris[i, 0]], verts[tris[i, 1]], verts[tris[i, 2]]
        e = i * 3
        edges[e] = v1 - v0
        edges[e + 1] = v2 - v1
        edges[e + 2] = v0 - v2
    return edges


def incidence_matrix(ts: NDArray[np.float32], vs:int) -> sp.csc_matrix:
    """
    VxE matrix where each column represents the incidence of a vector
    (v,e) = -1 means edge e leaves vertex v
    (u,e) = 1 means edge e enters vertex u
    """
    M = sp.dok_matrix((vs, ts.size), dtype=np.float32)

    for i in range(ts.shape[0]): # 0 - 9976
        for j in range(ts.shape[1]):
            e = i*3+j
            v = ts[i,j]
            u = ts[i,(j+1)%3]
            M[v, e] = -1
            M[u, e] = 1

    return M.tocsc()
