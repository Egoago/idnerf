from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List

import jaxlie

import jax.numpy as jnp


@dataclass
class Frame:
    T_rel: jaxlie.SE3
    id: int
    rgbdm_img: Optional[jnp.ndarray] = None
    pixel_coords_yx: Optional[jnp.ndarray] = None

    def to_dict(self):
        _dict = dict()
        _dict['id'] = self.id
        _dict['T_rel'] = se3_to_dict(self.T_rel)
        return _dict


@dataclass
class CameraParameters:
    focal: float
    width: int
    height: int


@dataclass
class History:
    loss: List[float] = field(default_factory=list)
    epsilon: List[float] = field(default_factory=list)
    grads: List[list] = field(default_factory=list)
    t_error: List[list] = field(default_factory=list)
    R_error: List[list] = field(default_factory=list)


@dataclass
class Data:
    cam_params: CameraParameters
    frames: List[Frame]
    T_true: jaxlie.SE3
    T_init: jaxlie.SE3
    T_final: Optional[jaxlie.SE3] = None
    history: Optional[History] = None

    def to_dict(self):
        _dict = dict()
        _dict['cam_params'] = asdict(self.cam_params)
        _dict['frames'] = [frame.to_dict() for frame in self.frames]
        _dict['T_true'] = se3_to_dict(self.T_true)
        _dict['T_init'] = se3_to_dict(self.T_init)
        _dict['T_final'] = se3_to_dict(self.T_final)
        _dict['history'] = asdict(self.history)
        return _dict


def se3_to_dict(T: jaxlie.SE3):
    return {'R': T.rotation().log().tolist(),
            't': T.translation().tolist()} if T is not None else None

