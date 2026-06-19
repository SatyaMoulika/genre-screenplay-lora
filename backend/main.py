"""
main.py — Genre Screenplay Generation API
Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import gc
import time
import torch
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────
HF_USERNAME = "SatyaMoulika"
BASE_MODEL  = "meta-llama/Llama-3.2-3B-Instruct"

ADAPTERS = {
    "drama":  f"{HF_USERNAME}/llama-3.2-drama-lora",
    "horror": f"{HF_USERNAME}/llama-3.2-horror-lora",
    "scifi":  f"{HF_USERNAME}/llama-3.2-scifi-lora",
    "comedy": f"{HF_USERNAME}/llama-3.2-comedy-lora",
}

GENERATION_DEFAULTS = {
    "max_new_tokens": 400,
    "temperature":    0.8,
    "top_p":          0.9,
    "do_sample":      True,
}

# ── Model state (on-demand loader) ───────────────────────────────
class ModelState:
    def __init__(self):
        self.current_genre: Optional[str] = None
        self.model = None
        self.tokenizer = None

    def load(self, genre: str):
        """Load adapter for the requested genre, unloading any previous one."""
        if self.current_genre == genre and self.model is not None:
            return  # already loaded

        self._unload()

        print(f"[ModelState] Loading {genre} adapter...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base, ADAPTERS[genre])
        model.eval()

        self.model         = model
        self.tokenizer     = tok
        self.current_genre = genre
        print(f"[ModelState] {genre} adapter ready.")

    def _unload(self):
        if self.model is not None:
            del self.model
            self.model         = None
            self.tokenizer     = None
            self.current_genre = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[ModelState] Previous model unloaded.")

    def generate(self, genre: str, prompt: str, max_new_tokens: int,
                 temperature: float, top_p: float) -> tuple[str, float]:
        self.load(genre)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        full   = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        scene  = full.split("### Screenplay:")[-1].strip()
        return scene, latency_ms


model_state = ModelState()


# ── Lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("API starting up...")
    yield
    model_state._unload()
    print("API shut down.")


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Genre Screenplay Generator",
    description="Generate screenplay scenes using genre-specific LoRA adapters.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    genre:          str   = Field(..., description="drama | horror | scifi | comedy")
    scene_heading:  str   = Field(..., description="e.g. INT. KITCHEN - NIGHT")
    story_context:  str   = Field(..., description="Brief premise or setup")
    characters:     Optional[str] = Field(None, description="Character names / descriptions")
    tone:           Optional[str] = Field(None, description="e.g. tense, melancholic, absurd")
    max_new_tokens: int   = Field(400,  ge=50,  le=800)
    temperature:    float = Field(0.8,  ge=0.1, le=1.5)
    top_p:          float = Field(0.9,  ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    genre:      str
    screenplay: str
    latency_ms: float
    prompt_used: str


# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "SceneForge API is running.",
        "docs":    "/docs",
        "genres":  list(ADAPTERS.keys()),
    }


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "loaded_genre":  model_state.current_genre,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/genres")
def list_genres():
    return {"genres": list(ADAPTERS.keys())}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.genre not in ADAPTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown genre '{req.genre}'. Choose from: {list(ADAPTERS.keys())}",
        )

    # Build enriched prompt
    extra_lines = []
    if req.characters:
        extra_lines.append(f"Characters: {req.characters}")
    if req.tone:
        extra_lines.append(f"Tone: {req.tone}")

    extras = ("\n" + "\n".join(extra_lines)) if extra_lines else ""

    prompt = (
        f"### Instruction:\nWrite a {req.genre} screenplay scene.\n"
        f"Scene heading: {req.scene_heading}\n"
        f"Story context: {req.story_context}"
        f"{extras}\n\n"
        f"### Screenplay:\n"
    )

    try:
        screenplay, latency_ms = model_state.generate(
            genre          = req.genre,
            prompt         = prompt,
            max_new_tokens = req.max_new_tokens,
            temperature    = req.temperature,
            top_p          = req.top_p,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        genre       = req.genre,
        screenplay  = screenplay,
        latency_ms  = latency_ms,
        prompt_used = prompt,
    )
