import os
import sys
import json
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from opts import parse_opts
from model import generate_model
from models.mmtnet import TrainStepWrap, NetWithLoss
from evalcallback import EvalCallBack

from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set


def get_ms_model(opt):
    network, parameters = generate_model(opt)
    # print(model)
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        param_dict = load_checkpoint(opt.resume_path)
        load_param_into_net(model, param_dict)

    milestone = [1]+opt.lr_steps
    lr_rates = [opt.learning_rate]
    for i in range(len(opt.lr_steps)):
        lr_rates.append(opt.learning_rate * (0.1 ** (i+1)))
    dynamic_lr = nn.piecewise_constant_lr(milestone, lr_rates)

    criterion = nn.SoftmaxCrossEntropyWithLogits()
    if opt.model=='resnext':
        optimizer = nn.SGD(
            filter(lambda p: p.requires_grad, network.get_parameters()), #parameters,  # 
            learning_rate=dynamic_lr,
            momentum=opt.momentum,
            dampening=opt.dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        if opt.resume_path:
            load_param_into_net(optimizer, param_dict)
        model = ms.Model(network, criterion, optimizer, metrics={"Accuracy": Accuracy()})
    elif opt.model=='mmtnet':
        optimizer_rgb = nn.SGD(network.get_rgb_params(), learning_rate=dynamic_lr, momentum=opt.momentum, 
          dampening=opt.dampening, weight_decay=opt.weight_decay, nesterov=opt.nesterov)
        optimizer_depth = nn.SGD(network.get_depth_params(), learning_rate=dynamic_lr, momentum=opt.momentum, 
          dampening=opt.dampening, weight_decay=opt.weight_decay, nesterov=opt.nesterov)
        if opt.resume_path:
            load_param_into_net(optimizer_rgb, param_dict)
            load_param_into_net(optimizer_depth, param_dict)

        loss_net = NetWithLoss(WideDeep_net, criterion)
        train_net = TrainStepWrap(loss_net, optimizer_rgb, optimizer_depth)
        train_net.set_train()
        model = ms.Model(train_net)

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            resume_paths = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)#'{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x',
                               opt.modality, str(opt.sample_duration)])
    if opt.nesterov:
        opt.dampening = 0

    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    model = get_ms_model(opt)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = C.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    elif not opt.std_norm:
        norm_method = C.Normalize(mean=opt.mean, std=[1, 1, 1])
    else:
        norm_method = C.Normalize(mean=opt.mean, std=opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        if opt.resize:
            spatial_transform = Compose([
                Scale(opt.resize_size),
                crop_method,
                SpatialElasticDisplacement(),
                norm_method,
                C.HWC2CHW()])
                # ToTensor(opt.norm_value), norm_method])
        else:
            spatial_transform = Compose([
                #RandomHorizontalFlip(),
                #RandomRotate(),
                #RandomResize(),
                crop_method,
                #MultiplyValues(),
                #Dropout(),
                #SaltImage(),
                #Gaussian_blur(),
                SpatialElasticDisplacement(),
                norm_method,
                C.HWC2CHW()
                # ToTensor(opt.norm_value), norm_method
            ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = ds.GeneratorDataset(training_data, ['clip', 'target', 'info'], shuffle=True)
        train_loader = train_loader.batch(opt.batch_size, drop_remainder=True)

 
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            norm_method,
            C.HWC2CHW()
            # ToTensor(opt.norm_value), norm_method
        ])
        #temporal_transform = LoopPadding(opt.sample_duration)
        temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        
        val_loader = ds.GeneratorDataset(validation_data, ['clip', 'target', 'info'], shuffle=False)
        val_loader = val_loader.batch(opt.batch_size, drop_remainder=True)


    best_info = {'epoch':0, 'acc':0, 'precision':0, 'recall':0, 'confusion_matrix': np.zeros((opt.n_classes, opt.n_classes), dtype=np.int).tolist()}

    print('run')
    
    time_cb = TimeMonitor(data_size=len(train_loader))
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=len(train_loader),
                                    keep_checkpoint_max=20)
    ckpt_cb = ModelCheckpoint(prefix=opt.store_name, directory=opt.result_path, config=config_ck)
    cb.append(ckpt_cb)

    def apply_eval(eval_param):
        res = eval_param['model'].eval(eval_param['dataset'])
        return res[eval_param['metrics_name']]

    eval_param_dict = {'model':model, 'dataset':val_loader,'metrics_name':'Accuracy'}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, ckpt_directory=opt.result_path, eval_start_epoch=10, interval=1)
    cb.append(eval_cb)

    model.train(opt.n_epochs, train_loader, callbacks=cb, dataset_sink_mode=True, sink_size=-1)



        






