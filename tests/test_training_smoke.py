import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.base import CHECKPOINT_DIR, DATA_PROCESSED_DIR
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model, train


def test_loss_decreases_and_checkpoint_round_trips():
    model, history = train(
        train_path=DATA_PROCESSED_DIR / "train.parquet",
        val_path=DATA_PROCESSED_DIR / "val.parquet",
        batch_size=8,
        num_epochs=2,
        checkpoint_name="pytest_smoke",
        device="cpu",
        max_train_examples=64,
        max_val_examples=16,
    )

    losses = [h["train_loss"] for h in history]
    assert losses[-1] < losses[0]

    reloaded = build_model()
    ckpt_path = Path(CHECKPOINT_DIR) / "pytest_smoke_last.pt"
    ckpt = load_checkpoint(ckpt_path, reloaded)
    assert ckpt["epoch"] == 2

    original_param = next(model.parameters())
    reloaded_param = next(reloaded.parameters())
    assert (original_param == reloaded_param).all()
