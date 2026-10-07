from torch.optim.lr_scheduler import LambdaLR

from configs.base import D_MODEL, WARMUP_STEPS


def noam_lr_lambda(last_epoch: int) -> float:
    # LambdaLR calls this with its own internal, 0-indexed step counter
    # (last_epoch): 0 before the first optimizer.step(), 1 after the first
    # scheduler.step(), and so on. The published Noam formula is 1-indexed
    # (step_num starts at 1), so add 1 to convert. Without this, clamping
    # last_epoch=0 up to step=1 (to avoid 0**-0.5 = inf) would make the
    # first two optimizer steps use the identical rate instead of a
    # strictly increasing one during warmup.
    step = last_epoch + 1
    return (D_MODEL ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))


def build_noam_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=noam_lr_lambda)
