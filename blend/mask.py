from dataclasses import dataclass, fields
from typing import Literal

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
    parts: dict[str, NDArray[np.int32]]
    count: int
    kind: Literal["vertex", "face"]

    def search(self):
        for i, f in enumerate(self.parts):
            for j, g in enumerate(self.parts):
                if j <= i:
                    continue
                l = self.parts[f]
                r = self.parts[g]
                intersect = np.intersect1d(l,r)
                intr = len(intersect)
                if intr > 0:
                    print(f"intersection for `{f}`[{intr}/{len(l)}] and `{g}`[{intr}/{len(r)}]")

    def validate(self) -> bool:
        valid = True
        all = np.concatenate([self.parts[f] for f in self.parts])
        unique = np.unique(all)
        if all.size != unique.size:
            print("overlapping vertices")
            valid =  False
        if unique.size != self.count:
            print("missing vertices")
            valid = False
        return valid

def _remove_dupes(parts: dict[str, NDArray[np.int32]], keep: str, cut: str):
    if keep not in parts or cut not in parts:
        return
    parts[cut] = np.setdiff1d(parts[cut], parts[keep], assume_unique=True)

def _fix_flame_parts(parts: dict[str, NDArray[np.int32]]):
    for f in parts:
        if f == "face":
            continue
        _remove_dupes(parts, f, cut="face")

    _remove_dupes(parts,"forehead", "scalp")
    _remove_dupes(parts,"left_eye_region", "forehead")
    _remove_dupes(parts,"right_eye_region", "forehead")
    _remove_dupes(parts,"boundary", "scalp")
    _remove_dupes(parts,"boundary", "neck")
    _remove_dupes(parts,"neck", "scalp")

def _from_flame(getter, count:int, kind:Literal["vertex","face"], regions:list[str]) -> Mask:
    parts = {}
    for region in regions:
        parts[region] = getter(region).cpu().numpy()

    _fix_flame_parts(parts)

    parts["remainder"] = np.setdiff1d(np.arange(count), np.concatenate(list(parts.values())))
    m = Mask(parts=parts, count=count, kind=kind)
    m.search()

    if not m.validate():
        print("error in mask. overlap")
        exit(1)

    return m

def fmask_from_flame(flame_mask: FlameMask, regions: list[str]=FLAME_PARTS):
    return _from_flame(flame_mask.get_fid_by_region, flame_mask.num_faces, "face", regions)

def vmask_from_flame(flame_mask: FlameMask, regions: list[str]=FLAME_PARTS):
    return _from_flame(flame_mask.get_vid_by_region, flame_mask.num_verts, "vertex", regions)
