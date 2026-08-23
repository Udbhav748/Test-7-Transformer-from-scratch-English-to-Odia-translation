from torch.optim.lr_scheduler import LambdaLR

from configs.base import D_MODEL, WARMUP_STEPS


def noam_lr_lambda(step: int) -> float:
    step = max(step, 1)
    return (D_MODEL ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))


def build_noam_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=noam_lr_lambda)
