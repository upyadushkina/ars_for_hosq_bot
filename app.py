import os, re, logging
from datetime import datetime
import zoneinfo
import pandas as pd

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ae-bot")

# ---------- config ----------
TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
if not TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN env var")

# ---------- data ----------
def load_df():
    path = os.getenv("PEOPLE_CSV_PATH", "data/ars_2025_people.csv")
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, on_bad_lines="skip")
    rename_map = {
        'name':'Name','person':'Name','full_name':'Name',
        'where to meet':'Where to Meet','where':'Where to Meet','location':'Where to Meet',
        'attendance':'Attendance','when':'Attendance','time':'Attendance',
        'bio':'Bio','conversation tip':'Conversation Tip','conversation':'Conversation Tip',
        'institution':'Institution','institution description':'Institution Description',
        'institution link':'Institution Link','link':'Institution Link'
    }
    df.columns = [c.strip() for c in df.columns]
    for k,v in list(rename_map.items()):
        for c in df.columns:
            if c.lower()==k:
                df.rename(columns={c:v}, inplace=True)
    for col in ["Name","Where to Meet","Attendance","Bio","Conversation Tip","Institution","Institution Description","Institution Link"]:
        if col not in df.columns:
            df[col] = ""
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df

DF = load_df()

def unique_locations(limit=60):
    locs = sorted({x.strip() for x in DF["Where to Meet"].tolist() if x and x.strip()})
    return locs[:limit]

def person_card(row):
    parts = [row.get('Name','').strip()]
    inst = row.get('Institution','').strip()
    role = row.get('Festival Role','').strip() if 'Festival Role' in row else ''
    if inst or role:
        parts.append(" — ".join([x for x in [role, inst] if x]))
    if row.get('Where to Meet',''):
        parts.append(f"📍 {row['Where to Meet']}")
    if row.get('Attendance',''):
        parts.append(f"🕒 {row['Attendance']}")
    if row.get('Conversation Tip',''):
        parts.append(f"💬 {row['Conversation Tip']}")
    if row.get('Bio',''):
        txt = row['Bio']
        parts.append(f"ℹ️ {txt[:400]}{'…' if len(txt)>400 else ''}")
    if row.get('Institution Link',''):
        parts.append(row['Institution Link'])
    return "\n".join(parts)

def search_by_name(query, limit=10):
    return DF[DF["Name"].str.contains(re.escape(query), case=False, na=False)].head(limit)

def filter_by_location(loc, limit=40):
    return DF[DF["Where to Meet"].str.contains(re.escape(loc), case=False, na=False)].head(limit)

def vienna_now():
    tz = zoneinfo.ZoneInfo("Europe/Vienna")
    return datetime.now(tz)

def is_available_now(attendance_text: str) -> bool:
    if not attendance_text:
        return False
    now = vienna_now()
    for a,b in re.findall(r'(\d{1,2}[:.]\d{2})\s*[-–]\s*(\d{1,2}[:.]\d{2})', attendance_text):
        try:
            h1,m1 = map(int,a.replace('.',':').split(':'))
            h2,m2 = map(int,b.replace('.',':').split(':'))
            t1 = now.replace(hour=h1, minute=m1, second=0, microsecond=0)
            t2 = now.replace(hour=h2, minute=m2, second=0, microsecond=0)
            if t1 <= now <= t2:
                return True
        except:
            pass
    return False

# ---------- FastAPI + PTB ----------
app = FastAPI()
application = Application.builder().token(TOKEN).build()

@app.on_event("startup")
async def on_startup():
    await application.initialize()
    log.info("PTB initialized")

@app.on_event("shutdown")
async def on_shutdown():
    await application.shutdown()
    log.info("PTB shutdown")

async def start(update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("🔎 Find by name", switch_inline_query_current_chat="")],
        [InlineKeyboardButton("📍 Browse by location", callback_data="loc:menu")],
        [InlineKeyboardButton("🕒 Available now (beta)", callback_data="time:now")],
    ]
    await update.message.reply_text(
        "Привет! Пришли имя или используй кнопки ниже.",
        reply_markup=InlineKeyboardMarkup(kb),
        disable_web_page_preview=True
    )

async def on_text(update, context: ContextTypes.DEFAULT_TYPE):
    q = (update.message.text or "").strip()
    if not q:
        return
    res = search_by_name(q)
    if res.empty:
        await update.message.reply_text("Ничего не нашла. Попробуй другое имя.")
        return
    for _, row in res.iterrows():
        await update.message.reply_text(person_card(row), disable_web_page_preview=True)

async def on_cb(update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    if data == "loc:menu":
        locs = unique_locations()
        rows, row = [], []
        for i,loc in enumerate(locs,1):
            row.append(InlineKeyboardButton(loc[:30], callback_data=f"loc:{loc[:60]}"))
            if len(row)==2:
                rows.append(row); row=[]
        if row: rows.append(row)
        await q.edit_message_text("Выбери локацию:", reply_markup=InlineKeyboardMarkup(rows or [[InlineKeyboardButton("Нет локаций", callback_data="noop")]]))
    elif data.startswith("loc:"):
        loc = data[4:]
        res = filter_by_location(loc)
        await q.edit_message_text(f"В «{loc}»:")
        for _, row in res.iterrows():
            await q.message.reply_text(person_card(row), disable_web_page_preview=True)
    elif data == "time:now":
        subset = DF[DF["Attendance"].apply(is_available_now)]
        await q.edit_message_text("Доступны сейчас:" if not subset.empty else "Сейчас никого не вижу по расписанию.")
        for _, row in subset.iterrows():
            await q.message.reply_text(person_card(row), disable_web_page_preview=True)

application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(on_cb))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

@app.get("/", response_class=PlainTextResponse)
def root():
    return "ok"

@app.post("/webhook")
async def webhook(request: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)):
    # Проверяем секрет от Telegram (если указали при setWebhook)
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        log.warning("Bad webhook secret: got=%s", x_telegram_bot_api_secret_token)
        raise HTTPException(status_code=403, detail="bad secret")
    try:
        data = await request.json()
        log.info("update %s", data.get("update_id"))
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return JSONResponse({"status":"ok"})
    except Exception as e:
        # Логируем весь стектрейс, чтобы понять первопричину
        log.exception("webhook error: %s", e)
        # Возвращаем 200, чтобы Telegram не забивал очередь ретраями
        return JSONResponse({"status":"error"}, status_code=200)
