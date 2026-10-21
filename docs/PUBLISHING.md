# What is safe to publish (e.g. GitHub) and what stays private

Checked 2026-10-21 against the actual project folder. The folder is **not** a Git repository yet; nothing has been committed, pushed or published. `.gitignore` already excludes everything in the "keep private / separate" column, so a normal `git add .` would pick up only the "safe" column.

| Safe to publish | Keep private or distribute separately | Why |
|---|---|---|
| `src/`, `tests/`, `scripts/`, `configs/`, `notebooks/*.ipynb` (sources, no outputs) | `.venv/`, `.venv313/`, `frontend/node_modules/` | environments are machine-specific and huge; re-create them from `pyproject.toml`, `requirements-lock.txt`, `package-lock.json` |
| `frontend/` sources, `package.json`, `package-lock.json`, `frontend/.env.example`, `frontend/vercel.json` | `frontend/dist/` | build output |
| `README.md`, `PROJECT_STATUS.md`, `docs/*.md` | `TEP_*.RData` (1.4 GB) | too large for Git; point to the public Tennessee Eastman dataset and keep `docs/DATA_CHECKSUMS.md` so others can verify their copy |
| `pyproject.toml`, `requirements-lock.txt`, `Dockerfile`, `.dockerignore`, `.gitignore` | `Mark Split 21CSE732P rubrics .pdf`, `Applied AI Industry Worklets Student Template.xlsx` | course material from the faculty (now in `.gitignore`); ask before sharing |
| `reports/figures/`, `reports/demo/0*.json`, `reports/demo/demo_samples.csv` (small, derived from the public dataset) | `artifacts/` (saved model, 5 experiment models) | binaries; publish them as a release asset or share privately if needed; `manifest.json` lets the receiver verify them |
| | `reports/executed/`, `reports/*.csv`, `reports/evaluation_*.json`, `reports/experiments/` | generated outputs; attach to the report or a release instead of the code history |
| | `data/` (includes `data/api/predictions.sqlite3`, your prediction history) | personal run history |
| | `logs/` | contain your Windows user name and paths |
| | `backups/` | local safety copies |
| | `reports/demo/holdout_*` | derived from the locked holdout runs; keep them for the live demo |

Secrets: no API keys, tokens, passwords or `.env` files were found (only `frontend/.env.example`). The admin token for `/model/reload` is read from the `PP_API_ADMIN_TOKEN` environment variable only; never write it into a file you publish.

Personal information to review before publishing:
* `docs/DATA_CHECKSUMS.md` names your local folder (`C:\Users\likes\...`). It is the baseline checksum record and was left unchanged on purpose; edit that one line yourself if you publish.
* The README and other docs use `<project folder>` instead of your path.
* Executed notebooks and logs (both ignored) contain local paths.

Before the first push (your decision, not done for you): `git init`, `git status` to review the file list, then commit. Check `git status` shows no `.RData`, `.keras`, `.sqlite3`, `logs/` or `artifacts/` entries.
