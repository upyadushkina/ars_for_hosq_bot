# Ars Electronica 2026 — hosq Meet Bot

Telegram-бот для команды hosq на Ars Electronica 2026.  
**Cloudflare Worker (Free plan)** · данные из **Google Sheets** · кэш ~10 мин.

## Что загрузить на GitHub (плоско, без папок)

| Файл | Нужен? |
|------|--------|
| `worker.js` | да |
| `wrangler.toml` | да |
| `package.json` | да |
| `README.md` | по желанию |

`.gitignore` и `data.js` загружать не нужно.

## Google Sheet

https://docs.google.com/spreadsheets/d/1X6oouceCLD3pY289WtT-gPeU_kxeq6mpilbLNehSGCw/edit

Три вкладки (имена важны): **`people`** · **`meet`** · **`schedule`**

Доступ: **Anyone with the link → Viewer**.

В `meet` заголовок колонки A должен быть просто `Name` (не `Name !Mediengruppe Bitnik`).

## Cloudflare

1. Build: `npm install`  
2. Deploy: `npx wrangler deploy`  
3. Secrets (Variables and Secrets):
   - `TELEGRAM_TOKEN` — обязательно  
   - `WEBHOOK_SECRET` — желательно  
   - `SHEET_ID` — необязательно (Sheet уже прописан в коде)

4. Проверка: `https://ars-for-hosq-bot.upyadushkina.workers.dev/` → `ok`

5. Webhook в браузере:

```
https://api.telegram.org/bot<TELEGRAM_TOKEN>/setWebhook?url=https://ars-for-hosq-bot.upyadushkina.workers.dev/webhook&secret_token=<WEBHOOK_SECRET>
```

Ожидай: `"Webhook was set"`.

## Как пользоваться ботом

- `/start` — главное меню  
- поиск по имени / локации / времени / теме / ивенту  
- 📅 расписание  
- **⚙️ Настройки → 🔄 Обновить данные** — сразу перечитать Sheet (сброс кэша)  

Правки в Sheets подхватываются сами примерно раз в 10 минут или сразу через «Обновить данные».
