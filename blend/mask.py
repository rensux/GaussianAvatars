from dataclasses import dataclass, fields
import numpy as np
from numpy.typing import NDArray

from flame_model.flame import FlameMask


@dataclass
class Mask:
    left_eyeball: NDArray[np.int32]
    right_eyeball: NDArray[np.int32]
    left_eye_region: NDArray[np.int32]
    right_eye_region: NDArray[np.int32]
    lips: NDArray[np.int32]
    nose: NDArray[np.int32]
    left_ear: NDArray[np.int32]
    right_ear: NDArray[np.int32]
    forehead: NDArray[np.int32]
    neck: NDArray[np.int32]
    boundary: NDArray[np.int32]
    # left_half: NDArray[np.int32] # remainder
    # right_half: NDArray[np.int32] # remainder

    def search(self):
        flds = fields(self)
        for i in range(len(flds)):
            for j in range(len(flds)):
                if j <= i:
                    continue
                f = flds[i]
                g = flds[j]
                if f.name=="left_half" or g.name=="left_half" or f.name=="right_half" or g.name=="right_half" :
                    continue
                l = getattr(self, f.name)
                r = getattr(self, g.name)
                intersect = np.intersect1d(l,r)
                intr = len(intersect)
                if intr > 0:
                    print(f"intersection for `{f.name}`[{intr}/{len(l)}] and `{g.name}`[{intr}/{len(r)}]")

    def validate(self, vs) -> bool:
        valid = True
        all = np.concatenate([getattr(self, f.name) for f in fields(self) ])
        unique = np.unique(all)
        print(all.size, unique.size)
        if all.size != unique.size:
            print("overlapping vertices")
            valid =  False
        if unique.size != vs:
            print("missing vertices")
            valid = False
        return valid

def from_flame(flame_mask: FlameMask) -> Mask:
    m = Mask(
        left_eyeball=flame_mask.get_vid_by_region("left_eyeball").cpu().numpy(),
        right_eyeball=flame_mask.get_vid_by_region("right_eyeball").cpu().numpy(),
        left_eye_region=flame_mask.get_vid_by_region("left_eye_region").cpu().numpy(),
        right_eye_region=flame_mask.get_vid_by_region("right_eye_region").cpu().numpy(),
        lips=flame_mask.get_vid_by_region("lips").cpu().numpy(),
        nose=flame_mask.get_vid_by_region("nose").cpu().numpy(),
        left_ear=flame_mask.get_vid_by_region("left_ear").cpu().numpy(),
        right_ear=flame_mask.get_vid_by_region("right_ear").cpu().numpy(),
        forehead=flame_mask.get_vid_by_region("forehead").cpu().numpy(),
        neck=flame_mask.get_vid_by_region("neck").cpu().numpy(),
        boundary=flame_mask.get_vid_by_region("boundary").cpu().numpy(),
        # left_half=flame_mask.get_vid_by_region("left_half").cpu().numpy(),
        # right_half=flame_mask.get_vid_by_region("right_half").cpu().numpy(),
    )
    vs=flame_mask.num_verts

    print(f"total verts: {vs}")
    m.search()

    if not m.validate(vs):
        print("error in mask. overlap")

    return m
