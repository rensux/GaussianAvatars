import time

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
from numpy.typing import NDArray
from tensorboard.compat.tensorflow_stub.dtypes import float32


class Model:
    MTM: sp.csr_matrix
    MT: sp.csr_matrix
    cano_edges: NDArray[np.float32] # M * E * 3
    cano_verts: NDArray[np.float32] # V * 3
    P: spla.LinearOperator


    def __init__(self, tris: NDArray[np.int32], mesh_verts: NDArray[np.float32]):
        m = mesh_verts.shape[0]
        self.cano_verts = mesh_verts[0]
        self.cano_edges = np.ndarray(shape=(m, len(tris), 3), dtype=np.float32)
        for i in range(mesh_verts.shape[0]): # for each mesh
            self.cano_edges[i]=calc_edges(tris, mesh_verts[i])

        x = time.time()
        self.MT, self.MTM = setup_MT_MTM(tris, self.cano_edges.shape[1], mesh_verts.shape[1])
        y = time.time()

        print(f"setting up MT and MTM took {y-x}s")
        # Compute preconditioner (Incomplete Cholesky)


        M_ichol = spla.spilu(self.MTM)
        self.P = spla.LinearOperator(self.MTM.shape, M_ichol.solve, dtype=np.float32)

        cfs = np.zeros(m, np.float32)
        cfs[0] = 1.

        self.calc_mesh(cfs)

    def calc_mesh(self, coefficients: NDArray[np.float32]) -> NDArray[np.float32]:
        x = time.time()
        # calc blended edges
        d = np.tensordot(coefficients, self.cano_edges, axes=1)
        # calc MTd
        MTd = self.MT.dot(d)
        # solve for vertices according to MTM * v = MTd
        v = np.column_stack([spla.cg(self.MTM, MTd[:, i])[0] for i in range(MTd.shape[1])])
        y = time.time()
        print(f"calcingmesh took {y-x}s")

        return v


def calc_edges(tris: NDArray[np.int32], verts: NDArray[np.int32]):
    edges = np.zeros((len(tris), 3), dtype=np.float32)
    for i in range(0, len(tris), 3):
        v0,v1,v2= verts[tris[i]], verts[tris[i + 1]],verts[tris[i + 2]]

        edges[i] = v1 - v0
        edges[i + 1] = v2 - v1
        edges[i + 2] = v0 - v2
    return edges


def setup_MT(ts: NDArray[np.int32], es:int, vs:int):
    mt_row = np.zeros(vs+1, dtype=np.int32)
    mt_val = np.zeros(es*2, dtype=np.float32)
    mt_col = np.zeros(es*2, dtype=np.int32)

    edges = np.zeros((es, 2), dtype=np.int32)
    mt_count = np.zeros(vs, dtype=np.int32)

    for i in range(0, len(ts), 3):
        # store for each edge which vector is outgoing / incoming (implicitly encoded as -1 and 1 respectively)
        # store for each vector how many adjacent vectors they have

        #  e: a -> b (edge from vector a to vector b)
        #  edges[e, 0] = a (outgoing)
        #  edges[e, 1] = b (incoming)
        #  counts[a] (# outgoing/incoming edges)

        #  e0: v0 -> v1
        edges[i, 0] = ts[i]
        edges[i, 1] = ts[i + 1]
        #  e1: v1 -> v2
        edges[i + 1, 0] = ts[i + 1]
        edges[i + 1, 1] = ts[i + 2]
        #  e2: v2 -> v0
        edges[i + 2, 0] = ts[i + 2]
        edges[i + 2, 1] = ts[i]

        mt_count[ts[i]] += 2
        mt_count[ts[i + 1]] += 2
        mt_count[ts[i + 2]] += 2

    for i in range(vs):
        mt_row[i + 1] = mt_row[i] + mt_count[i]

    for i in range(es):
        # e: v0 -> v1
        v0 = edges[i, 0] # -1
        v1 = edges[i, 1] # 1

        mt_idx0 = mt_row[v0 + 1] - mt_count[v0] # row offset + index in row = end - #entries left
        mt_idx1 = mt_row[v1 + 1] - mt_count[v1] # row offset + index in row = end - #entries left=

        mt_count[v0] -= 1
        mt_count[v1] -= 1

        mt_val[mt_idx0] = -1
        mt_col[mt_idx0] = i
        mt_val[mt_idx1] = 1
        mt_col[mt_idx1] = i
    return mt_val,mt_col, mt_row

def setup_MTM(ts, vs, mt_row):
    adjacency = [None for _ in range(vs)]  # directed edge adjacency list
    max_row = 0

    def init_adj_list(v, max_row):
        if adjacency[v] is not None:
            return max_row
        cap = mt_row[v + 1] - mt_row[v]
        adjacency[v] = []
        return max(max_row, cap)

    for i in range(0, len(ts), 3):  # 3 edges per triangle
        max_row = init_adj_list(ts[i],max_row)
        max_row = init_adj_list(ts[i + 1],max_row)
        max_row = init_adj_list(ts[i + 2],max_row)

        # e0: v0 -> v1
        adjacency[ts[i]].append(ts[i + 1])
        adjacency[ts[i + 1]].append(ts[i])
        # e1: v1 -> v2
        adjacency[ts[i + 1]].append(ts[i + 2])
        adjacency[ts[i + 2]].append(ts[i + 1])
        # e2: v2 -> v0
        adjacency[ts[i + 2]].append(ts[i])
        adjacency[ts[i]].append(ts[i + 2])

    max_row //= 2  # unique edges are at most half of the directed edges

    nnz = 0  # number of nonzero entries
    nnz_row = np.zeros(vs, dtype=int)
    col_buf = np.zeros((vs, max_row), dtype=int)
    val_buf = np.zeros((vs, max_row), dtype=float)

    for i in range(len(adjacency)):
        adjacency[i].sort()
        j = 0
        while j < len(adjacency[i]):
            v = nnz_row[i]
            col_buf[i, v] = adjacency[i][j]
            while j + 1 < len(adjacency[i]) and col_buf[i, v] == adjacency[i][j + 1]:
                val_buf[i, v] -= 1  # count duplicates
                j += 1
            nnz_row[i] += 1
            j += 1

        nnz += nnz_row[i]

    mtm_row = np.zeros(vs + 1, dtype=int)
    mtm_col = np.zeros(nnz, dtype=int)
    mtm_val = np.zeros(nnz, dtype=float)

    for i in range(vs):
        mtm_row[i + 1] = mtm_row[i] + nnz_row[i]
        mtm_col[mtm_row[i]:mtm_row[i + 1]] = col_buf[i, :nnz_row[i]]
        mtm_val[mtm_row[i]:mtm_row[i + 1]] = val_buf[i, :nnz_row[i]]


    return mtm_val,mtm_col, mtm_row

def setup_MT_MTM(ts: NDArray[np.int32], es:int, vs:int):
    mt_val, mt_col, mt_row = setup_MT(ts, es,vs)
    MT = sp.csr_matrix((mt_val, mt_col, mt_row))
    # MTM = sp.csr_matrix(setup_MTM(ts, vs, mt_row))
    # MTM_ =
    # assert MTM == MTM_
    return MT,MT.dot(MT.T)