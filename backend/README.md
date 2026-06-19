# SceneForge — Backend

FastAPI service that loads genre-specific LoRA adapters on-demand and generates screenplay scenes.

## Setup

```bash
cd backend
pip install -r requirements.txt
export HF_TOKEN=your_token_here
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`

## Endpoints

| Method | Endpoint  | Description                        |
|--------|-----------|-------------------------------------|
| GET    | /         | API info                            |
| GET    | /health   | Health check + currently loaded genre |
| GET    | /genres   | Lists available genres              |
| POST   | /generate | Generate a screenplay scene         |

### POST /generate

```json
{
  "genre": "horror",
  "scene_heading": "EXT. ABANDONED FARMHOUSE - NIGHT",
  "story_context": "A group of friends realise they are not alone.",
  "characters": "JAKE (20s, reckless), SARA (20s, cautious)",
  "tone": "tense",
  "max_new_tokens": 400,
  "temperature": 0.8,
  "top_p": 0.9
}
```

## Notes

- Adapters load on-demand (memory efficient, ~5–10s swap time per genre change)
- Requires a CUDA GPU with at least 8GB VRAM
- CORS is open (`*`) by default — restrict `allow_origins` in `main.py` before deploying publicly
