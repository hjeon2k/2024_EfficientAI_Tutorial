import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# STE: Straight-Through Estimator
class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        return grad_output


# STE can be implemented in various ways
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


# LSQ Function
class LSQ(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel=False):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1).transpose(0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp)) * alpha
            w_q = torch.transpose(w_q, 0, 1).contiguous().view(sizes)
        else:
            # TODO set w_q for layer-wise quantization
            # hint: there will be a single-sized tensor for quant scale factor, alpha
            # w_q =
            pass
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1).transpose(0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = torch.div(weight, alpha).transpose(0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            # TODO set q_w for layer-wise quantization
            # hint: there will be a single-sized tensor for quant scale factor, alpha
            # q_w =
            pass

        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger

        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp +
                           between * Round.apply(q_w) - between * q_w)*grad_weight * g)
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp +
                           between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None


class LSQWeightQuantizer(nn.Module):
    def __init__(self, w_bits, per_channel=False, channel_num = 128, batch_init = 20):
        super(LSQWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.batch_init = batch_init
        self.Qn = - 2 ** (w_bits - 1)
        self.Qp = 2 ** (w_bits - 1) - 1
        self.per_channel = per_channel
        if not self.per_channel:
            self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            # TODO define the size of the quantization scale parameter
            # self.s = torch.nn.Parameter(torch.ones(), requires_grad=True)
            pass
        self.init_state = 0
        self.g = 1.0

    def forward(self, weight):
        if self.init_state==0:
            if self.per_channel:
                self.s.data = torch.max(torch.abs(weight.detach().contiguous().view(weight.size()[0], -1)), dim=-1)[0]/self.Qp
            else:
                self.s.data = torch.max(torch.abs(weight.detach()))/self.Qp
            self.init_state += 1
            return weight
        elif self.init_state<self.batch_init:
            if self.per_channel:
                self.s.data = 0.9*self.s.data + 0.1*torch.max(torch.abs(weight.detach().contiguous().view(weight.size()[0], -1)), dim=-1)[0]/self.Qp
            else:
                self.s.data = 0.9*self.s.data + 0.1*torch.max(torch.abs(weight.detach()))/self.Qp
            self.init_state += 1
            return weight
        else:
            w_q = LSQ.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)
            # alpha = grad_scale(self.s, g)
            # w_q = Round.apply((weight/alpha).clamp(Qn, Qp)) * alpha
        return w_q

class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 w_bits=8,
                 quant_inference=False,
                 per_channel=False,
                 batch_init = 20):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, per_channel=per_channel, channel_num=out_features, batch_init = batch_init)

    def forward(self, input):
        if self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(input, quant_weight, self.bias)
        return output

def add_quant_op(module, w_bits=8, quant_inference=False, per_channel=False, batch_init = 20):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if child.bias is not None:
                quant_linear = QuantLinear(child.in_features, child.out_features, bias=True,
                                           w_bits=w_bits, quant_inference=quant_inference,
                                           per_channel=per_channel, batch_init = batch_init)
                quant_linear.bias.data = child.bias
            else:
                quant_linear = QuantLinear(child.in_features, child.out_features, bias=False,
                                           w_bits=w_bits, quant_inference=quant_inference,
                                           per_channel=per_channel, batch_init = batch_init)
            quant_linear.weight.data = child.weight
            module._modules[name] = quant_linear

        else:
            add_quant_op(child, w_bits=w_bits, quant_inference=quant_inference,
                         per_channel=per_channel, batch_init = batch_init)


def lsq_prepare(model, w_bits=8, quant_inference=False, per_channel=False, batch_init = 20):
    add_quant_op(model, w_bits=w_bits, quant_inference=quant_inference,
                 per_channel=per_channel, batch_init = batch_init)
    return model
