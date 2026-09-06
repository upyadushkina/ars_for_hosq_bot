# Ars Electronica 2026 — hosq Meet Bot

Telegram bot for finding people at Ars Electronica Festival 2026 (Negotiating Humanity, 8–13 Sep).

## Why Containers (not plain Workers)

This app is **FastAPI + pandas + uvicorn**. Cloudflare Workers cannot run that stack as a static/`wrangler deploy` site — you need **Cloudflare Containers** (Docker + a tiny Worker that proxies requests).

## Upload these files to GitHub (flat — no folders)

| File | Role |
|------|------|
| `app.py` | FastAPI + Telegram webhook |
| `ars_2026_people.csv` | People DB |
| `meet_slots.csv` | Meet slots |
| `ars_schedule.csv` | Schedule |
| `requirements.txt` | Python deps |
| `Dockerfile` | Runs uvicorn on port 8080 |
| `index.js` | Worker → container proxy |
| `wrangler.toml` | Container config |
| `package.json` | npm / wrangler |
| `.gitignore` | Ignore secrets / venv |
| `.dockerignore` | Slimmer Docker build |
| `README.md` | This file |

## Deploy on Cloudflare (Containers)

### 1. Upload the files above to the GitHub repo root (no subfolders)

Do **not** upload `.env` or tokens.

### 2. In Cloudflare Dashboard → Workers & Pages

- Worker name: `ars-for-hosq-bot` (same as in `wrangler.toml`)
- Root = GitHub repo root (where `wrangler.toml` and `Dockerfile` sit)
- **Build command:** `npm install`
- **Deploy command:** `npx wrangler deploy`

### 3. Secrets (Worker → Settings → Variables and Secrets)

- `TELEGRAM_TOKEN` — from BotFather  
- `WEBHOOK_SECRET` — optional (`openssl rand -hex 32`)

Forwarded into the container by `index.js`.

### 4. After deploy

Wait a few minutes, then open:

```
https://ars-for-hosq-bot.upyadushkina.workers.dev/
```

Should return `ok`.

### 5. Set Telegram webhook (browser)

```
https://api.telegram.org/bot<TELEGRAM_TOKEN>/setWebhook?url=https://ars-for-hosq-bot.upyadushkina.workers.dev/webhook&secret_token=<WEBHOOK_SECRET>
```

Expected: `"Webhook was set"`. Check with `getWebhookInfo`.

## Local run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TELEGRAM_TOKEN="..."
export WEBHOOK_SECRET="..."
uvicorn app:app --reload --port 8000
```

## Notes

- First request after idle can be slow (container wake). `sleepAfter` is `2h`.
- Trim `ars_schedule.csv` if you want a personal “Моё расписание”.
- Fill `Photo` URLs in the people CSV if you want image cards.
