import mindspore
from mindspore import nn
import mindspore.ops.operations as P
from mindspore import context
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from models import resnext, mmtnet
import pdb

def generate_model(opt):
    assert opt.model in ['resnext', 'mmtnet']


    if opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]
        from models.resnext import get_fine_tuning_parameters
        model = resnext.get_model(opt.model_depth,
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    elif opt.model == 'mmtnet':
        from models.mmtnet import get_fine_tuning_parameters
        opt.model = 'resnext'
        opt.modality = 'RGB'
        opt.no_cuda = True
        pretrain_dataset = opt.pretrain_dataset
        if opt.pretrain_dataset == opt.dataset:
            opt.pretrain_dataset = ''
        rgb = generate_model(opt)[0]
        opt.modality = 'Depth'
        depth = generate_model(opt)[0]
        opt.model = 'mmtnet'
        opt.modality = 'RGB-D'
        opt.pretrain_dataset = pretrain_dataset
        opt.no_cuda = False
        model = mmtnet.MMTNet(opt)
        model.set_return_both(True)
        model.set_rgb_depth_nets(rgb, depth)

    if not opt.no_cuda:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = load_checkpoint(opt.pretrain_path)
        if opt.pretrain_dataset == 'jester':
            if 'fc.weight' in pretrain:
                del pretrain['fc.weight']
                del pretrain['fc.bias']
            load_param_into_net(model, pretrain)

        if opt.model=='mmtnet':
            if opt.pretrain_dataset == 'hcigesture_allbutnone':
                del pretrain['rgb.fc.weight']
                del pretrain['rgb.fc.bias']
                del pretrain['depth.fc.weight']
                del pretrain['depth.fc.bias']
                load_param_into_net(model, pretrain)
            elif opt.pretrain_dataset == opt.dataset:
                load_param_into_net(model, pretrain)
        else:
            model = modify_kernels(opt, model, opt.modality)
            if opt.pretrain_dataset == opt.dataset:
                load_param_into_net(model, pretrain)
            elif opt.pretrain_dataset in ['egogesture', 'nvgesture', 'denso', 'hcigesture_allbutnone']:
                if 'fc.weight' in pretrain:
                    del pretrain['fc.weight']
                    del pretrain['fc.bias']
                load_param_into_net(model, pretrain)

        parameters = get_fine_tuning_parameters(model, opt.ft_portion)

    return model, parameters
    # else:
    #     if opt.pretrain_path:
    #         print('loading pretrained model {}'.format(opt.pretrain_path))
    #         pretrain = torch.load(opt.pretrain_path, map_location='cpu')
    #         from collections import OrderedDict
    #         pretrain_new = OrderedDict()
    #         for k, v in pretrain['state_dict'].items():
    #             name = k[7:]  # delete `module.`
    #             pretrain_new[name] = v

    #         # if opt.model not in ['mmtnet']:
    #         #     model = modify_kernels(opt, model, opt.pretrain_modality)
    #         # model.load_state_dict(pretrain['state_dict'])
    #         # model.load_state_dict(pretrain_new, strict=False)
    #         if opt.pretrain_dataset == opt.dataset:
    #             model.load_state_dict(pretrain_new)
    #         elif opt.pretrain_dataset in ['jester', 'egogesture', 'nvgesture', 'denso']:
    #             del pretrain_new['fc.weight']
    #             del pretrain_new['fc.bias']
    #             model.load_state_dict(pretrain_new,strict=False)

    #         if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
    #             model.classifier = nn.Sequential(
    #                             nn.Dropout(0.9),
    #                             nn.Linear(model.classifier[1].in_features, opt.n_finetune_classes)
    #                             )
    #         elif opt.model == 'squeezenet':
    #             model.classifier = nn.Sequential(
    #                             nn.Dropout(p=0.5),
    #                             nn.Conv3d(model.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
    #                             nn.ReLU(inplace=True),
    #                             nn.AvgPool3d((1,4,4), stride=1))
    #         else:
    #             model.fc = nn.Linear(model.fc.in_features, opt.n_finetune_classes)
    #         if opt.model not in ['mmtnet']:
    #             model = modify_kernels(opt, model, opt.modality)
    #         parameters = get_fine_tuning_parameters(model, opt.ft_portion)
    #         return model, parameters
    #     else:
    #         if opt.model not in ['mmtnet']:
    #             model = modify_kernels(opt, model, opt.modality)

    # return model, model.parameters()


def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    cells = list(map(lambda x:x[1], base_model.cells_and_names()))
    first_conv_idx = list(filter(lambda x: isinstance(cells[x], nn.Conv3d),
                                               list(range(len(cells)))))[0]

    conv_layer = cells[first_conv_idx]
    container = cells[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.get_parameters()]
    kernel_size = params[0].shape
    new_kernel_size = kernel_size[:1] + (1*motion_length,  ) + kernel_size[2:]
    new_kernels = P.ReduceMean(keep_dims=True)(params[0].data, axis=1)

    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, stride=conv_layer.stride,
                         pad_mode='pad', padding=conv_layer.padding, has_bias=True if len(params) == 2 else False)
    new_conv.weight.set_data(new_kernels)
    if len(params) == 2:
        new_conv.bias.set_data(params[1].data)  # add bias if neccessary
    layer_name = list(container.parameters_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)

    return base_model

def _construct_rgbdepth_model(base_model):
    # modify the first convolution kernels for RGB-D input
    cells = list(map(lambda x:x[1], base_model.cells_and_names()))
    first_conv_idx = list(filter(lambda x: isinstance(cells[x], nn.Conv3d),
                                               list(range(len(cells)))))[0]

    conv_layer = cells[first_conv_idx]
    container = modules[first_conv_idx - 1]
    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.get_parameters()]
    kernel_size = params[0].shape
    new_kernel_size = kernel_size[:1] + (1 * motion_length,) + kernel_size[2:]
    new_kernels = P.Mul()(P.Concat(1)((params[0].data, P.ReduceMean(keep_dims=True)(params[0].data, axis=1))), 0.6)
    new_kernel_size = kernel_size[:1] + (3 + 1 * motion_length,) + kernel_size[2:]
    new_conv = nn.Conv3d(4, conv_layer.out_channels, conv_layer.kernel_size, stride=conv_layer.stride,
                         pad_mode='pad', padding=conv_layer.padding, has_bias=True if len(params) == 2 else False)
    new_conv.weight.set_data(new_kernels)
    if len(params) == 2:
        new_conv.bias.set_data(params[1].data)  # add bias if neccessary
    layer_name = list(container.parameters_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

    # replace the first convolution layer
    setattr(container, layer_name, new_conv)
    return base_model

def _modify_first_conv_layer(base_model, new_kernel_size1, new_filter_num):
    cells = list(map(lambda x:x[1], base_model.cells_and_names()))
    first_conv_idx = list(filter(lambda x: isinstance(cells[x], nn.Conv3d),
                                               list(range(len(cells)))))[0]
    conv_layer = cells[first_conv_idx]
    container = cells[first_conv_idx - 1]
 
    new_conv = nn.Conv3d(new_filter_num, conv_layer.out_channels, kernel_size=(new_kernel_size1,7,7),
                         stride=(1,2,2), pad_mode='pad', padding=(1,1,3,3,3,3), has_bias=False)
    layer_name = list(container.parameters_dict().keys())[0][:-7]

    setattr(container, layer_name, new_conv)
    return base_model

def modify_kernels(opt, model, modality):
    if modality == 'RGB' and opt.model not in ['c3d', 'squeezenet', 'mobilenet','shufflenet', 'mobilenetv2', 'shufflenetv2', 'magnifynet']:
        print("[INFO]: RGB model is used for init model")
        if opt.dataset != 'jester' and not opt.no_change_firstlayer:
            model = _modify_first_conv_layer(model,3,3) ##### Check models trained (3,7,7) or (7,7,7)
    elif modality == 'Depth':
        print("[INFO]: Converting the pretrained model to Depth init model")
        model = _construct_depth_model(model)
        print("[INFO]: Done. Flow model ready.")
    elif modality == 'RGB-D':
        print("[INFO]: Converting the pretrained model to RGB+D init model")
        model = _construct_rgbdepth_model(model)
        if opt.no_change_firstlayer:
            model = _modify_first_conv_layer(model,3,4)
        print("[INFO]: Done. RGB-D model ready.")
    
    cells = list(map(lambda x:x[1], model.cells_and_names()))
    first_conv_idx = list(filter(lambda x: isinstance(cells[x], nn.Conv3d),
                                               list(range(len(cells)))))[0]
    conv_layer = cells[first_conv_idx]
    if conv_layer.kernel_size[0]> opt.sample_duration:
       model = _modify_first_conv_layer(model,int(opt.sample_duration/2),1)
    return model
