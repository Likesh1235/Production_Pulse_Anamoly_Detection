# Deployment (not yet deployed — run these yourself)

Status: **nothing is deployed.** The assistant's cloud workspace had no access to Vercel or any backend host and no credentials. What was tested locally (cloud, Linux): the backend started from a Docker-like minimal file tree, real model predictions, CORS, admin-token protection, history persistence across a restart, and the production frontend build talking to that backend in a headless browser.

## Backend (FastAPI + TensorFlow) — not suitable for Vercel
TensorFlow plus the model needs about **700 MB RAM** (measured) and a long-running process, so use a container host with **1 GB or more** (Fly.io, Render Docker web service, Railway, a VPS). A 512 MB free tier will likely run out of memory.

The repo's `.gitignore` excludes `/artifacts/` and `*.keras`, so a host that builds from Git will **not** have the model. Build from your local folder instead:

```powershell
# Fly.io example (needs flyctl and an account)
fly launch --no-deploy --dockerfile Dockerfile --internal-port 8000
fly secrets set PP_API_CORS_ORIGINS="https://YOUR-APP.vercel.app" PP_API_ADMIN_TOKEN="<16+ random chars>"
fly volume create ppdata --size 1          # then mount it at /app/data/api so history survives restarts
fly deploy
```
`Dockerfile` copies only `src`, `configs`, the saved model folder `artifacts/models/autoencoder_v1` and `reports/evaluation_metrics.json` (see `.dockerignore`). It listens on `$PORT` (default 8000) and sets `PRODUCTION_PULSE_ROOT=/app`.

Environment variables: `PP_API_CORS_ORIGINS` (comma list, exact origins, never `*`), `PP_API_ADMIN_TOKEN` (optional, 16+ chars, enables `POST /model/reload`), `PP_API_DATABASE_PATH` (must stay inside `data/`), `PP_API_STORE_INPUTS` (default false). Without a persistent volume, prediction history is lost on each restart.
The API has no login on `/predict`, `/history`, `/metrics`; anyone who can reach the URL can use it. Put it behind a gateway or keep the URL private if that matters.

## Frontend (Vercel)
`frontend/vercel.json` is included (SPA rewrite, basic security headers).
```powershell
cd frontend
npm i -g vercel
vercel login
vercel link
vercel env add VITE_API_BASE_URL production    # e.g. https://YOUR-API.fly.dev  (https, no trailing slash)
vercel --prod
```
Then set `PP_API_CORS_ORIGINS` on the backend to the exact Vercel URL (add any preview/custom domain too) and redeploy/restart the backend. `VITE_API_BASE_URL` is baked in at build time, so rebuild after changing it.

## Verify after deploying
```powershell
curl https://YOUR-API/health
curl -i -X OPTIONS https://YOUR-API/predict -H "Origin: https://YOUR-APP.vercel.app" -H "Access-Control-Request-Method: POST"
python scripts\demo_prediction.py --url https://YOUR-API
```
Then open the Vercel URL: Overview should show API Online / Model Ready; submit a sample on Predictions.
