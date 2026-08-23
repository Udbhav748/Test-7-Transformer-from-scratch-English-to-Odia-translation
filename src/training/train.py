import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from configs.base import ADAM_BETAS, ADAM_EPS, GRAD_CLIP_NORM, PAD_ID
from src.data.dataset import TranslationDataset, collate_fn
from src.model.transformer import Seq2SeqTransformer
from src.tokenization.tokenizer_utils import load_tokenizer
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH
from src.training.checkpoint import save_checkpoint
from src.training.lr_schedule import build_noam_scheduler


def build_model():
    en_vocab_size = load_tokenizer(EN_TOKENIZER_PATH).get_vocab_size()
    or_vocab_size = load_tokenizer(OR_TOKENIZER_PATH).get_vocab_size()
    return Seq2SeqTransformer(en_vocab_size, or_vocab_size)


def run_epoch(model, loader, loss_fn, optimizer=None, scheduler=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_tokens = 0.0, 0

    for src_ids, _src_pad_mask, tgt_ids, _tgt_pad_mask in loader:
        src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
        decoder_input = tgt_ids[:, :-1]
        decoder_target = tgt_ids[:, 1:]

        with torch.set_grad_enabled(is_train):
            logits = model(src_ids, decoder_input)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

        n_tokens = (decoder_target != PAD_ID).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def train(
    train_path,
    val_path,
    batch_size,
    num_epochs,
    checkpoint_name,
    device="cpu",
    max_train_examples=None,
    max_val_examples=None,
):
    model = build_model().to(device)

    train_ds = TranslationDataset(train_path)
    val_ds = TranslationDataset(val_path)
    if max_train_examples is not None:
        train_ds = Subset(train_ds, range(min(max_train_examples, len(train_ds))))
    if max_val_examples is not None:
        val_ds = Subset(val_ds, range(min(max_val_examples, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # base lr=1.0: the Noam schedule computes the actual lr as a multiplier, applied by LambdaLR
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=ADAM_BETAS, eps=ADAM_EPS)
    scheduler = build_noam_scheduler(optimizer)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_val_loss = float("inf")
    history = []
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        val_loss = run_epoch(model, val_loader, loss_fn, None, None, device)
        global_step += len(train_loader)
        elapsed = time.time() - start
        print(
            f"epoch {epoch}/{num_epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} ({elapsed:.1f}s)"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        save_checkpoint(model, optimizer, scheduler, global_step, epoch, val_loss, f"{checkpoint_name}_last")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, global_step, epoch, val_loss, f"{checkpoint_name}_best")

    return model, history
