from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flame_model.flame import FlameMask

FLAME_PARTS=[
    "left_eyeball",
    "right_eyeball",
    "left_eye_region",
    "right_eye_region",
    "lips",
    "left_ear",
    "right_ear",
    "forehead",
    "neck",
    "scalp",
    "nose",
    "face",
    "boundary",
             ]

@dataclass
class Mask:
    parts: list[str]
    v_count: int
    verts: list[NDArray[np.int32]]
    f_count: int
    faces: list[NDArray[np.int32]]

    def search(self):
        for i, l in enumerate(self.verts):
            for j, r in enumerate(self.verts):
                if j <= i:
                    continue
                intersect = np.intersect1d(l,r)
                intr = len(intersect)
                if intr > 0:
                    print(f"intersection for `{self.parts[i]}`[{intr}/{len(l)}] and `{self.parts[j]}`[{intr}/{len(r)}]")

    def validate(self) -> bool:
        valid = True
        all = np.concatenate([f for f in self.verts])
        unique = np.unique(all)
        if all.size != unique.size:
            print("overlapping vertices")
            valid =  False
        if unique.size != self.v_count:
            print("missing vertices")
            valid = False
        return valid

def _remove_dupes(parts: list[str], verts: list[NDArray[np.int32]], faces: list[NDArray[np.int32]], keep: str, cut: str):
    try:
        keep_i = parts.index(keep)
        cut_i = parts.index(cut)
    except ValueError:
        return

    verts[cut_i] = np.setdiff1d(verts[cut_i], verts[keep_i], assume_unique=True)
    faces[cut_i] = np.setdiff1d(faces[cut_i], faces[keep_i], assume_unique=True)

def _fix_flame_parts(parts: list[str], verts: list[NDArray[np.int32]], faces: list[NDArray[np.int32]]):
    for f in parts:
        if f == "face":
            continue
        _remove_dupes(parts, verts, faces, f, cut="face")

    _remove_dupes(parts, verts, faces,"forehead", "scalp")
    _remove_dupes(parts, verts, faces,"left_eye_region", "forehead")
    _remove_dupes(parts, verts, faces,"right_eye_region", "forehead")
    _remove_dupes(parts, verts, faces,"boundary", "scalp")
    _remove_dupes(parts, verts, faces,"boundary", "neck")
    _remove_dupes(parts, verts, faces,"neck", "scalp")

def from_flame(flame_mask: FlameMask, regions:list[str]=FLAME_PARTS) -> Mask:
    verts = []
    faces = []
    for region in regions:
        verts.append(flame_mask.get_vid_by_region(region).cpu().numpy())
        faces.append(flame_mask.get_fid_by_region(region).cpu().numpy())

    _fix_flame_parts(regions, verts, faces)

    verts.append(np.setdiff1d(np.arange(flame_mask.num_verts), np.concatenate(verts)))
    faces.append(np.setdiff1d(np.arange(flame_mask.num_faces), np.concatenate(faces)))
    m = Mask(parts=regions + ["remainder"], verts=verts, faces=faces, v_count=flame_mask.num_verts, f_count=flame_mask.num_faces)
    m.search()

    if not m.validate():
        print("error in mask. overlap")
        exit(1)

    return m
