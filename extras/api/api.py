"""Local inference API for the English -> Odia model.

Loads the trained checkpoint once and serves translations. Inference only;
nothing here trains or fits anything.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configs.base import CHECKPOINT_DIR, MAX_LEN
from extras.inference.beam_search import beam_search_decode
from src.inference.greedy_decode import greedy_decode
from src.tokenization.tokenizer_utils import decode, encode, load_tokenizer
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH
from src.training.checkpoint import load_checkpoint
from src.training.train import build_model

CHECKPOINT_PATH = CHECKPOINT_DIR / "kaggle_run_best.pt"

en_tok = load_tokenizer(EN_TOKENIZER_PATH)
or_tok = load_tokenizer(OR_TOKENIZER_PATH)
model = build_model()
load_checkpoint(CHECKPOINT_PATH, model)
model.eval()
torch.set_num_threads(4)

app = FastAPI(title="En-Or translator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    strategy: str = Field(default="greedy", pattern="^(greedy|beam)$")


class TranslateResponse(BaseModel):
    translation: str
    strategy: str
    source_tokens: int
    output_tokens: int


@app.get("/api/health")
def health():
    return {"status": "ok", "checkpoint": CHECKPOINT_PATH.name, "max_len": MAX_LEN}


@app.post("/api/translate", response_model=TranslateResponse)
@torch.no_grad()
def translate(req: TranslateRequest):
    src_ids = encode(en_tok, req.text.strip())
    if len(src_ids) > MAX_LEN:
        raise HTTPException(status_code=422, detail=f"input is {len(src_ids)} subword tokens; max is {MAX_LEN}")
    src = torch.tensor([src_ids], dtype=torch.long)
    if req.strategy == "beam":
        out_ids = beam_search_decode(model, src)
    else:
        out_ids = greedy_decode(model, src)
    return TranslateResponse(
        translation=decode(or_tok, out_ids),
        strategy=req.strategy,
        source_tokens=len(src_ids),
        output_tokens=len(out_ids),
    )
