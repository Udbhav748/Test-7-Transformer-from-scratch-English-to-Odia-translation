from pathlib import Path

import torch

from configs.base import CHECKPOINT_DIR


def save_checkpoint(model, optimizer, scheduler, step: int, epoch: int, val_loss: float, name: str):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(CHECKPOINT_DIR) / f"{name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt
