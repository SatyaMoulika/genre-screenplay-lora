# Genre-Specific Screenplay Generation with LoRA

Fine-tunes separate LoRA adapters on `meta-llama/Llama-3.2-3B-Instruct` — one per genre (Drama, Horror, Sci-Fi, Comedy) — to generate genre-conditioned screenplay scenes. Includes a style-transfer evaluation suite and a deployable FastAPI + HTML app.

## Why

Most fine-tuning demos show a single before/after comparison. This project instead trains **four independent adapters on the same base model** and proves measurable style separation between them — via perplexity, embedding clustering, and distinctiveness scoring — rather than just eyeballing outputs.

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| 1. Data Prep | `notebooks/01_data_prep.ipynb` | Filters [`mocboch/movie_scripts`](https://huggingface.co/datasets/mocboch/movie_scripts) by genre, cleans and chunks screenplays into scene-level instruction samples |
| 2. Training | `notebooks/02_lora_training.ipynb` | QLoRA fine-tuning (via Unsloth) of 4 separate adapters on Kaggle 2x T4 |
| 3. Evaluation | `notebooks/03_style_transfer_eval.ipynb` | Perplexity, cross-genre confusion matrix, embedding distinctiveness, UMAP clustering |
| 4. Serving | `notebooks/04_kaggle_serve_backend.ipynb` | Run the FastAPI backend on a Kaggle GPU instance, exposed via ngrok |

## App

| Folder | Description |
|---|---|
| `backend/` | FastAPI service — loads adapters on-demand, generates screenplay scenes |
| `frontend/` | Single-file HTML/CSS/JS UI — genre picker, scene inputs, live generation |

See `backend/README.md` and `frontend/README.md` for setup.

## Models & Datasets (HuggingFace Hub)

**Datasets**
- [`SatyaMoulika/imsdb-drama-screenplay`](https://huggingface.co/datasets/SatyaMoulika/imsdb-drama-screenplay)
- [`SatyaMoulika/imsdb-horror-screenplay`](https://huggingface.co/datasets/SatyaMoulika/imsdb-horror-screenplay)
- [`SatyaMoulika/imsdb-scifi-screenplay`](https://huggingface.co/datasets/SatyaMoulika/imsdb-scifi-screenplay)
- [`SatyaMoulika/imsdb-comedy-screenplay`](https://huggingface.co/datasets/SatyaMoulika/imsdb-comedy-screenplay)

**LoRA Adapters**
- [`SatyaMoulika/llama-3.2-drama-lora`](https://huggingface.co/SatyaMoulika/llama-3.2-drama-lora)
- [`SatyaMoulika/llama-3.2-horror-lora`](https://huggingface.co/SatyaMoulika/llama-3.2-horror-lora)
- [`SatyaMoulika/llama-3.2-scifi-lora`](https://huggingface.co/SatyaMoulika/llama-3.2-scifi-lora)
- [`SatyaMoulika/llama-3.2-comedy-lora`](https://huggingface.co/SatyaMoulika/llama-3.2-comedy-lora)

## Key Results

| Genre  | Adapter PPL (own genre) | Base PPL (own genre) | PPL Delta | Distinctiveness |
|--------|--------------------------|------------------------|-----------|------------------|
| Drama  | 3.68 | 6.02 | +2.34 | 0.535 |
| Horror | 3.28 | 5.33 | +2.05 | 0.528 |
| Sci-Fi | 3.72 | 5.54 | +1.82 | 0.521 |
| Comedy | 2.98 | 4.60 | +1.62 | 0.525 |

Every adapter is meaningfully more fluent (lower perplexity) on its own genre than the base model, with consistent stylistic distinctiveness across all four — confirming the adapters learned genre-specific style rather than generic instruction-following.

## Tech Stack

`transformers` · `peft` · `bitsandbytes` · `Unsloth` · `trl` · `sentence-transformers` · `FastAPI` · `Llama-3.2-3B-Instruct`

## Setup

```bash
git clone https://github.com/SatyaMoulika/genre-screenplay-lora.git
cd genre-screenplay-lora

# Backend
cd backend && pip install -r requirements.txt
export HF_TOKEN=your_token
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd ../frontend && python -m http.server 3000
```

Open `http://localhost:3000`.

> Requires a CUDA GPU (8GB+ VRAM). See `notebooks/04_kaggle_serve_backend.ipynb` to run on a free Kaggle GPU instead.

## License

MIT — see `LICENSE`
