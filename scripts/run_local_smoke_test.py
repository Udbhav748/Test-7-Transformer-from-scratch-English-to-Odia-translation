import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.base import (
    BATCH_SIZE_LOCAL_SMOKE,
    DATA_PROCESSED_DIR,
    NUM_EPOCHS_LOCAL_SMOKE,
)
from src.training.train import train


def main():
    model, history = train(
        train_path=DATA_PROCESSED_DIR / "train.parquet",
        val_path=DATA_PROCESSED_DIR / "val.parquet",
        batch_size=BATCH_SIZE_LOCAL_SMOKE,
        num_epochs=NUM_EPOCHS_LOCAL_SMOKE,
        checkpoint_name="smoke",
        device="cpu",
        max_train_examples=256,
        max_val_examples=64,
    )

    losses = [h["train_loss"] for h in history]
    assert losses[-1] < losses[0], (
        f"train loss did not decrease over the smoke run: {losses}"
    )
    print("smoke test passed: train loss decreased", losses)


if __name__ == "__main__":
    main()
