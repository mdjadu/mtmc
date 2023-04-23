import torch


from models import TRTModule
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list

device = torch.device(0)
Engine = TRTModule(1, device)
H, W = Engine.inp_info[0].shape[-2:]
camera_id = 0
confidence_threshold = 0.6