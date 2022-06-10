import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
# from mindspore.communication.management import get_group_size


from models import resnext


class MMTM(nn.Cell):
  def __init__(self, dim_rgb, dim_depth, ratio):
    super(MMTM, self).__init__()
    dim = dim_rgb + dim_depth
    dim_out = int(2*dim/ratio)
    self.fc_squeeze = nn.Dense(in_channels=dim, out_channels=dim_out)

    self.fc_rgb = nn.Dense(in_channels=dim_out, out_channels=dim_rgb)
    self.fc_depth = nn.Dense(in_channels=dim_out, out_channels=dim_depth)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def construct(self, rgb, depth):
    squeeze_array = []
    for tensor in [rgb, depth]:
      tview = P.Reshape()(tensor, tensor.shape[:2] + (-1,))
      squeeze_array.append(P.ReduceMean()(tview, -1)) #dim=
    squeeze = P.Concat(1)(squeeze_array)

    excitation = self.fc_squeeze(squeeze)
    excitation = self.relu(excitation)

    vis_out = self.fc_rgb(excitation)
    sk_out = self.fc_depth(excitation)

    vis_out = self.sigmoid(vis_out)
    sk_out = self.sigmoid(sk_out)

    dim_diff = len(rgb.shape) - len(vis_out.shape)
    vis_out = P.Reshape()(vis_out, vis_out.shape + (1,) * dim_diff)

    dim_diff = len(depth.shape) - len(sk_out.shape)
    sk_out = P.Reshape()(sk_out, sk_out.shape + (1,) * dim_diff)

    return rgb * vis_out, depth * sk_out


class MMTNet(nn.Cell):
  def __init__(self, args):
    super(MMTNet, self).__init__()
    self.rgb = None
    self.depth = None
    self.final_pred = None

    self.mmtm1 = MMTM(256, 256, 4)
    self.mmtm2 = MMTM(512, 512, 4)
    self.mmtm3 = MMTM(1024, 1024, 4)
    self.mmtm4 = MMTM(2048, 2048, 4)

    self.return_interm_feas = False
    self.return_both = False
    if hasattr(args, 'fc_final_preds') and args.fc_final_preds:
      self.final_pred = nn.Dense(in_channels=args.num_classes * 2, out_channels=args.num_classes)

  def get_mmtm_params(self):
    parameters = [
                {'params': self.mmtm1.get_parameters()},
                {'params': self.mmtm2.get_parameters()},
                {'params': self.mmtm3.get_parameters()},
                {'params': self.mmtm4.get_parameters()}
                         ]
    return parameters

  def get_rgb_params(self):
    parameters = [{'params': self.rgb.get_parameters()}]+self.get_mmtm_params()
    return parameters

  def get_depth_params(self):
    parameters = [{'params': self.depth.get_parameters()}]+self.get_mmtm_params()
    return parameters

  def set_rgb_depth_nets(self, rgb, depth, return_interm_feas=False):
    self.rgb = rgb
    self.depth = depth
    self.return_interm_feas = return_interm_feas

  def set_return_both(self, p):
    self.return_both = p

  def construct(self, x):
    rgb_x = x[:, :-1, :, :, :]
    depth_x = ops.ExpandDims()(x[:, -1, :, :, :], 1)

    # rgb INIT BLOCK
    rgb_x = self.rgb.conv1(rgb_x)
    rgb_x = self.rgb.bn1(rgb_x)
    rgb_x = self.rgb.relu(rgb_x)
    rgb_x = self.rgb.maxpool(rgb_x)

    # depth INIT BLOCK
    depth_x = self.depth.conv1(depth_x)
    depth_x = self.depth.bn1(depth_x)
    depth_x = self.depth.relu(depth_x)
    depth_x = self.depth.maxpool(depth_x)

    # MMTM
    rgb_features, depth_features = [], []

    rgb_x = self.rgb.layer1(rgb_x)
    depth_x = self.depth.layer1(depth_x)
    rgb_x, depth_x = self.mmtm1(rgb_x, depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)
    
    rgb_x = self.rgb.layer2(rgb_x)
    depth_x = self.depth.layer2(depth_x)
    rgb_x, depth_x = self.mmtm2(rgb_x, depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)

    rgb_x = self.rgb.layer3(rgb_x)
    depth_x = self.depth.layer3(depth_x)
    rgb_x, depth_x = self.mmtm3(rgb_x, depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)

    rgb_x = self.rgb.layer4(rgb_x)
    depth_x = self.depth.layer4(depth_x)
    rgb_x, depth_x = self.mmtm4(rgb_x, depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)

    rgb_x = self.rgb.avgpool(rgb_x)
    rgb_x = P.Reshape()(rgb_x, (P.Shape()(rgb_x)[0], -1,))
    rgb_x = self.rgb.fc(rgb_x)
    depth_x = self.depth.avgpool(depth_x)
    depth_x = P.Reshape()(depth_x, (P.Shape()(depth_x)[0], -1,))
    depth_x = self.depth.fc(depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)

    if self.return_interm_feas:
      return rgb_features, depth_features

    ### LATE FUSION
    if self.final_pred is None:
      pred = (rgb_x + depth_x)/2
    else:
      pred = self.final_pred(P.Concat(-1)([rgb_x, depth_x]))

    if self.return_both:
      return rgb_x, depth_x

    return pred


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


class NetWithLoss(nn.Cell):

    """"
    Provide training loss through network.
    Args:
        network (Cell): The training network
        criterion : loss function
    """

    def __init__(self, network, criterion, host_device_mix=False, parameter_server=False,
                 sparse=False, cache_enable=False):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = criterion

    def construct(self, inputs, label):
        '''
        Construct NetWithLoss
        '''
        rgb_out, depth_out = self.network(inputs)
        rgb_loss = self.loss(rgb_out, label)
        depth_loss = self.loss(depth_out, label)

        return rgb_loss, depth_loss



class IthOutputCell(nn.Cell):
    def __init__(self, network, output_index):
        super(IthOutputCell, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, *inputs):
        predict = self.network(*inputs)[self.output_index]
        return predict

# reference https://gitee.com/mindspore/models/blob/r1.5/official/recommend/wide_and_deep/src/wide_and_deep.py
class TrainStepWrap(nn.Cell):
    """
    Encapsulation class of WideDeep network training.
    Append Adam and FTRL optimizers to the training network after that construct
    function can be called to create the backward graph.
    Args:
        network (Cell): The training network. Note that loss function should have been added.
        sens (Number): The adjust parameter. Default: 1024.0
        host_device_mix (Bool): Whether run in host and device mix mode. Default: False
        parameter_server (Bool): Whether run in parameter server mode. Default: False
    """

    def __init__(self, network, optimizer_rgb, optimizer_depth, host_device_mix=False, parameter_server=False,
                 sparse=False, cache_enable=False, sens=1.0):
        super(TrainStepWrap, self).__init__()
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        self.network = network
        self.network.set_grad()

        self.optimizer_rgb = optimizer_rgb
        self.optimizer_depth = optimizer_depth
        
        self.weights_rgb = optimizer_rgb.parameters
        self.weights_depth = optimizer_depth.parameters
        # self.trainable_params = network.trainable_params()
        # weights_w = []
        # weights_d = []
        # for params in self.trainable_params:
        #     if 'wide' in params.name:
        #         weights_w.append(params)
        #     else:
        #         weights_d.append(params)
        # self.weights_w = ParameterTuple(weights_w)
        # self.weights_d = ParameterTuple(weights_d)
        
        # if (sparse and is_auto_parallel) or (sparse and parameter_server):
        #   if host_device_mix or (parameter_server and not cache_enable):
        #     self.optimizer_rgb.target = 'CPU'
        #     self.optimizer_depth.target = 'CPU'
        # if (sparse and is_auto_parallel) or (sparse and parameter_server):
        #     self.optimizer_depth = LazyAdam(
        #         self.weights_d, learning_rate=3.5e-4, eps=1e-8, loss_scale=sens)
        #     self.optimizer_rgb = FTRL(learning_rate=5e-2, params=self.weights_w,
        #                             l1=1e-8, l2=1e-8, initial_accum=1.0, loss_scale=sens)
        #     if host_device_mix or (parameter_server and not cache_enable):
        #         self.optimizer_rgb.target = "CPU"
        #         self.optimizer_depth.target = "CPU"
        # else:
        #     self.optimizer_depth = Adam(
        #         self.weights_d, learning_rate=3.5e-4, eps=1e-8, loss_scale=sens)
        #     self.optimizer_rgb = FTRL(learning_rate=5e-2, params=self.weights_w,
        #                             l1=1e-8, l2=1e-8, initial_accum=1.0, loss_scale=sens)
        # self.hyper_map = ops.HyperMap()
        self.grad_rgb = ops.GradOperation(get_by_list=True, sens_param=True)
        self.grad_depth = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.loss_net_rgb = IthOutputCell(network, output_index=0)
        self.loss_net_depth = IthOutputCell(network, output_index=1)
        self.loss_net_rgb.set_grad()
        self.loss_net_depth.set_grad()

        self.reducer_flag = False
        self.grad_reducer_rgb = None
        self.grad_reducer_depth = None
        self.reducer_flag = parallel_mode in (ParallelMode.DATA_PARALLEL,
                                              ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer_rgb = DistributedGradReducer(self.optimizer_rgb.parameters, mean, degree)
            self.grad_reducer_depth = DistributedGradReducer(self.optimizer_depth.parameters, mean, degree)

    def construct(self, *inputs):
        '''
        Construct wide and deep model
        '''

        loss_rgb, loss_depth = self.network(*inputs)
        sens_rgb = ops.Fill()(loss_rgb.dtype, loss_rgb.shape, self.sens)
        sens_depth = ops.Fill()(loss_depth.dtype, loss_depth.shape, self.sens)
        grad_rgb = self.grad_rgb(self.loss_net_rgb, self.weights_rgb)(*inputs, sens_rgb)
        grad_depth = self.grad_depth(self.loss_net_depth, self.weights_depth)(*inputs, sens_depth)
        if self.reducer_flag:
            grad_rgb = self.grad_reducer_rgb(grad_rgb)
            grad_depth = self.grad_reducer_depth(grad_depth)

        return ops.depend(loss_rgb, self.optimizer_rgb(grad_rgb)), ops.depend(loss_depth, self.optimizer_depth(grad_depth))


class CustomWithEvalCell(nn.Cell):
    """
    Predict definition
    """
    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__()
        self.network = network

    def construct(self, inputs, labels):
        rgb_out, depth_out = self.network(inputs)
        return _, (rgb_out+depth_out)/2, labels  # loss, outputs, label
