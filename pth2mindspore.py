import torch

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

from models import resnext

import numpy as np


def torch_to_mindspore(torch_file_path, mindspore_file_path, sample_duration):
    ckpt = torch.load(torch_file_path, map_location=torch.device('cpu'))

    new_params_list = []
    for _, v in ckpt['state_dict'].items():
        # if hasattr(v, 'numpy'):
        new_params_list.append(v.numpy())
        # else:
        #     new_params_list.append(v)

    mindspore_params_list = []
    network = resnext.get_model(101,
            num_classes=27, # jester dataset
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=sample_duration)
    for v, k in zip(new_params_list, network.parameters_dict().keys()):
        if 'fc' in k:
            continue
        # if isinstance(v, np.ndarray):
        mindspore_params_list.append({'name': k, 'data': Tensor.from_numpy(v)})
        # else:
        #     print(type(v))
        #     mindspore_params_list.append({'name': k, 'data': v})

    save_checkpoint(mindspore_params_list, mindspore_file_path)
    print('convert pytorch ckpt file to mindspore ckpt file ok !')


if __name__ == '__main__':
    torch_ckpt_file_path = 'pretrained_models/jester_resnext_101_RGB_16_best.pth'#sys.argv[1]
    mindspore_ckpt_file_path = 'pretrained_models/jester_resnext_101_RGB_16_ms.ckpt'#sys.argv[2]
    sample_duration = 16
    context.set_context(device_target='GPU')
    torch_to_mindspore(torch_ckpt_file_path, mindspore_ckpt_file_path, sample_duration)
