# Aftertrace

A web-first camera lab that turns user videos into glitch-art trace visuals and shows how trackable you are.

## Stack

- **Backend**: Python 3.11+, FastAPI, OpenCV, Librosa, MoviePy
- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS
- **Design**: Minimal, dark theme, one accent color (#FF6B35)

## Local Development

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000

---

## Deployment

### Suggested Hosting

| Component | Recommended | Alternatives |
|-----------|-------------|--------------|
| Backend   | Render, Fly.io | Railway, DigitalOcean App Platform |
| Frontend  | Vercel | Netlify, Cloudflare Pages |

### Backend Deployment

**Requirements:**
- Python 3.11+
- FFmpeg (usually pre-installed on Render/Fly; if not, `imageio-ffmpeg` provides a bundled binary)

**Environment Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `CORS_ORIGINS` | Comma-separated allowed origins | `https://aftertrace.vercel.app` |
| `DEBUG` | Enable debug mode (optional) | `false` |

**Build & Run:**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

On Render, set the start command to:
```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Frontend Deployment

**Environment Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `https://aftertrace-api.onrender.com` |

**Build & Run:**

```bash
cd web
npm install
npm run build
npm run start
```

On Vercel, these commands run automatically. Just set the env var in the project settings.

### Quick Checklist

1. Deploy backend first, note its URL
2. Set `NEXT_PUBLIC_API_URL` in frontend to backend URL
3. Set `CORS_ORIGINS` in backend to frontend URL
4. Deploy frontend

---

## Architecture Notes

- Videos are processed server-side and deleted after download
- No accounts, no persistent storage, no telemetry
- Max upload: 20 seconds, 1080p resolution
- Processed videos are stored in `/tmp` and cleaned up via `/cleanup/{job_id}`

## Project Structure

```
aftertrace/
├── backend/
│   ├── app/
│   │   ├── api/routes.py      # API endpoints
│   │   ├── core/config.py     # Settings
│   │   ├── services/          # Video processing
│   │   │   ├── process_video.py
│   │   │   ├── tracking.py
│   │   │   ├── effects.py
│   │   │   ├── audio.py
│   │   │   └── presets.py
│   │   └── main.py            # FastAPI app
│   └── requirements.txt
├── web/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── components/
│   ├── lib/
│   └── public/
└── PROJECT_OVERVIEW.md
```



