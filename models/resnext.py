import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as P
# from torch.autograd import Variable
import math
from functools import partial
from mindspore.common import initializer as init
# from model_utils import KaimingNormal
from models import model_utils

__all__ = ['ResNeXt', 'resnext50', 'resnext101']


# def conv3x3x3(in_planes, out_planes, stride=1):
#     # 3x3x3 convolution with padding
#     return nn.Conv3d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=1,
#         has_bias=False)


def downsample_basic_block(x, planes, stride):
    out = P.AvgPool3D(x, kernel_size=1, strides=stride)
    zero_pads = P.Zeros()((
        out.shape(0), planes - out.shape(1), out.shape(2), out.shape(3),
        out.shape(4)), ms.float32)
    # if isinstance(out.data, torch.cuda.FloatTensor):
    #     zero_pads = zero_pads.cuda()

    out = P.Concat(axis=1)((out.data, zero_pads)) # out.data

    return out


class ResNeXtBottleneck(nn.Cell):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = model_utils.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            pad_mode='pad',
            padding=1,
            groups=cardinality,
            has_bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Cell):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            pad_mode='pad',
            padding=3,
            has_bias=False)
        #self.conv1 = nn.Conv3d(
        #    3,
        #    64,
        #    kernel_size=(3,7,7),
        #    stride=(1, 2, 2),
        #    padding=(1, 3, 3),
        #    has_bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = P.MaxPool3D(kernel_size=(3, 3, 3), strides=2, pad_mode='pad', pad_list=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = P.AvgPool3D(
            (last_duration, last_size, last_size), strides=1)
        self.fc = nn.Dense(in_channels=cardinality * 32 * block.expansion, out_channels=num_classes)

        for _, cell in self.cells_and_names():
            # if isinstance(m, nn.Conv3d):
            #     m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            # elif isinstance(m, nn.BatchNorm3d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

            if isinstance(cell, nn.Conv3d):
                cell.weight.set_data(init.initializer(
                    # KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    'HeNormal',
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm3d):
                cell.bn2d.gamma.set_data(init.initializer(
                    'ones', cell.bn2d.gamma.shape, cell.bn2d.gamma.dtype))
                cell.bn2d.beta.set_data(init.initializer(
                    'zeros', cell.bn2d.beta.shape, cell.bn2d.beta.dtype))
            # elif isinstance(cell, nn.Dense):
            #     cell.weight.set_data(init.initializer(
            #         init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
            #     if cell.bias is not None:
            #         cell.bias.set_data(init.initializer(
            #             'zeros', cell.bias.shape, cell.bias.dtype))




    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.SequentialCell([
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        has_bias=False), nn.BatchNorm3d(planes * block.expansion)])

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.SequentialCell([*layers])

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = P.Reshape()(x, (P.Shape()(x)[0], -1,))
        # print("size before fc", x.size())
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.trainable_params()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for param in model.get_parameters():
            for ft_module in ft_module_names:
                if ft_module in param.name:
                    parameters.append(param)
                    break
                else:
                    # parameters.append({'params': param, 'lr': 0.0})
                    pass
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")

def get_model(model_depth, **kwargs):
    if model_depth == 50:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    elif model_depth == 101:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    elif model_depth == 152:
        model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
# def resnext50(**kwargs):
#     """Constructs a ResNet-50 model.
#     """
#     model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
#     return model


# def resnext101(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
#     return model


# def resnext152(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
#     return model
