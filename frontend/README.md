# SceneForge — Frontend

Single-file HTML/CSS/JS frontend for the screenplay generator. No build step, no framework.

## Run

Just open `index.html` in a browser, or serve it statically:

```bash
cd frontend
python -m http.server 3000
```

Then visit `http://localhost:3000`

## Configuration

The frontend talks to the backend API at a hardcoded base URL. Update this line near the bottom of `index.html` before deploying:

```js
const API_BASE = 'http://localhost:8000';
```

Change it to your deployed backend's URL (e.g. `https://your-api.onrender.com`).

## Requirements

The backend (`../backend`) must be running and reachable at `API_BASE` for generation to work.
