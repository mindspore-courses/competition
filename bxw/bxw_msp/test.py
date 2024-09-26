

# 1、Normal参数 放外面，Normal直接产生 standard normal
# 2、

import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor, Parameter, context, ParameterTuple, ops
from mindspore.nn.probability.distribution import Normal, Bernoulli

shape = (3, 4)
stdnormal = ops.StandardNormal(seed=2)
output = stdnormal(shape)
print(output)





context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mean_param = Parameter(Tensor([0,0,0,0,0],mindspore.float32))
        self.var_param = Parameter(Tensor([1,1,1,1,1],mindspore.float32))
        self.dist = Normal(self.mean_param, self.var_param)
    def construct(self, input_tensor):
        # snm = ops.StandardNormal()
        # out1 = snm(self.mean_param.shape)
        # out = out1 * self.var_param + self.mean_param
        shape = (3, 4)
        stdnormal = ops.StandardNormal(seed=2)
        output = stdnormal(shape)
        # print(self.dist._mean_value)
        # out = self.dist.sample()
        # loss = input_tensor*out
        # loss = loss.sum()
        # dist = self.dist
        return output

class GradNetWithWrtParams(nn.Cell):
    def __init__(self, net):
        super(GradNetWithWrtParams, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)
    def construct(self, x):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x)

net = Net()
print(net)
# params = ParameterTuple(net.trainable_params())
# optimizer = nn.Adam(params=net.trainable_params(), learning_rate=1e-3)
# def train_step(batch):
#     grad_op = ops.GradOperation(get_by_list=True)
#     grad = grad_op(net, params)(batch)
#     optimizer(grad)
# print(net.mean_param.data.asnumpy(),net.var_param.data.asnumpy())
# input_tensor  = Tensor(np.random.randn(5), mindspore.float32)
# train_step(input_tensor)
# print(net.mean_param.data.asnumpy(),net.var_param.data.asnumpy())


# mask_prior_prob = 0.4
# p_mask_probs = Tensor(mask_prior_prob * ops.ones((722, 100)), mindspore.float32)
# # print(p_mask_probs)
# x = ops.sigmoid(ops.log(0.01*ops.ones((722,100),mindspore.float32)))
# dist = Bernoulli(x)
# dist = Bernoulli(probs=ops.sigmoid(ops.log(p_mask_probs)))
# print(dist.sample((5,)))

