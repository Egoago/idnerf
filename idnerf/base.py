from dataclasses import dataclass, field, asdict
from typing import Optional, List

import jaxlie
from jaxnerf.nerf import utils

import jax.numpy as jnp


@dataclass
class Frame:
    T_cam2base: jaxlie.SE3
    id: int
    rgbdm_img: Optional[jnp.ndarray] = None

    def to_dict(self):
        _dict = dict()
        _dict['id'] = self.id
        _dict['T_cam2last'] = se3_to_dict(self.T_cam2base)
        return _dict


@dataclass
class CameraParameters:
    focal: float
    width: int
    height: int


@dataclass
class History:
    loss: List[float] = field(default_factory=list)
    log_T_pred: List[float] = field(default_factory=list)
    grads: List[list] = field(default_factory=list)
    t_error: List[list] = field(default_factory=list)
    R_error: List[list] = field(default_factory=list)
    sample_count: List[list] = field(default_factory=list)


@dataclass
class Data:
    cam_params: CameraParameters
    frames: List[Frame]
    T_true: Optional[jaxlie.SE3] = None
    T_init: Optional[jaxlie.SE3] = None
    T_final: Optional[jaxlie.SE3] = None
    history: Optional[History] = None

    def to_dict(self):
        _dict = dict()
        _dict['cam_params'] = asdict(self.cam_params) if self.cam_params is not None else None
        _dict['frames'] = [frame.to_dict() for frame in self.frames]
        _dict['T_true'] = se3_to_dict(self.T_true)
        _dict['T_init'] = se3_to_dict(self.T_init)
        _dict['T_final'] = se3_to_dict(self.T_final)
        _dict['history'] = asdict(self.history) if self.history is not None else None
        return _dict


def se3_to_dict(T: jaxlie.SE3):
    return {'R': T.rotation().log().tolist(),
            't': T.translation().tolist()} if T is not None else None


class Flags(dict):
    def __init__(self, flags: dict):
        super().__init__()
        self.load(flags)

    def load(self, flags: dict):
        for key, value in flags.items():
            self[key] = value

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


utils.define_flags()

FLAGS = None


def load_flags():
    global FLAGS
    import warnings
    import jax.numpy as jnp
    from absl import flags

    warnings.filterwarnings("ignore")

    jnp.set_printoptions(precision=4)

    flags.DEFINE_string("result_dir", "results/", "")
    flags.DEFINE_string("test_name", None, "")
    flags.DEFINE_enum("pixel_sampling", "total", ["total", "random", "patch", "fast", "fast_random"], "")
    flags.DEFINE_enum("subset", "lego", ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], "",
                      allow_override=True)
    flags.DEFINE_enum("optimizer", "adam", ["adam", "sgd"], "")
    flags.DEFINE_float("perturbation_R", 1.0, "", 0., 1e8)
    flags.DEFINE_float("perturbation_t", 1.0, "", 0., 1e8)
    flags.DEFINE_float("decay_rate", 0.6, "", 0., 1.)
    flags.DEFINE_float("huber_delta", 1.0, "", 0., 1.)
    flags.DEFINE_float("clip_grad", 0., "", 0., 10.)
    flags.DEFINE_float("depth_param", 0., "", 0., 10.)
    flags.DEFINE_float("rgb_param", 1., "", 0., 10.)
    flags.DEFINE_integer("pixel_count", 128, "", 1, 16384)
    flags.DEFINE_integer("decay_steps", 100, "", 1, 16384)
    flags.DEFINE_integer("frame_idx", 0, "", 0, 16384)
    flags.DEFINE_bool("distributed_render", False, "")
    flags.DEFINE_bool("per_sample_gradient", False, "")
    flags.DEFINE_bool("use_original_img", True, "")
    flags.DEFINE_bool("resample_rays", True, "")
    flags.DEFINE_bool("coarse_opt", False, "")
    flags.DEFINE_list("frame_ids", [0], "")

    if flags.FLAGS.config is not None:
        from jaxnerf.nerf import utils
        utils.update_flags(flags.FLAGS)
    if flags.FLAGS.result_dir is None:
        raise ValueError("data_dir must be set. None set now.")

    def getattribute(key):
        try:
            return flags.FLAGS.__getattribute__(key)
        except:
            return flags.FLAGS.__getattr__(key)

    flags_dict = {key:getattribute(key) for key in flags.FLAGS.flag_values_dict().keys()}
    FLAGS = Flags(flags_dict)
