from torch.autograd import Function


class GradReverse(Function):
    lambd = 0

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * GradReverse.lambd
