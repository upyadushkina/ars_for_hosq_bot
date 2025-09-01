import os
import re
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import PlainTextResponse
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from datetime import datetime
import zoneinfo

# --- Config ---
TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

if not TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN env var")

# --- Data loading ---
def load_df():
    path = os.getenv("PEOPLE_CSV_PATH", "ars_2025_people.csv")
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, on_bad_lines="skip")
    # auto-rename распространённых заголовков
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

# --- Bot logic ---
app = FastAPI()
application = Application.builder().token(TOKEN).build()

def person_card(row):
    parts = []
    parts.append(f"*{row.get('Name','').strip()}*")
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
        parts.append(f"ℹ️ {row['Bio'][:400]}{'…' if len(row['Bio'])>400 else ''}")
    if row.get('Institution Link',''):
        parts.append(f"{row['Institution Link']}")
    return "\n".join(parts)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("🔎 Find by name", switch_inline_query_current_chat="")],
        [InlineKeyboardButton("📍 Browse by location", callback_data="loc:menu")],
        [InlineKeyboardButton("🕒 Available now (beta)", callback_data="time:now")]
    ]
    await update.message.reply_text(
        "Привет! Пришли имя или используй кнопки ниже.",
        reply_markup=InlineKeyboardMarkup(kb)
    )

def search_by_name(query, limit=10):
    mask = DF["Name"].str.contains(re.escape(query), case=False, na=False)
    return DF[mask].head(limit)

def filter_by_location(loc, limit=40):
    mask = DF["Where to Meet"].str.contains(re.escape(loc), case=False, na=False)
    return DF[mask].head(limit)

def vienna_now():
    tz = zoneinfo.ZoneInfo("Europe/Vienna")
    return datetime.now(tz)

def is_available_now(attendance_text: str) -> bool:
    if not attendance_text:
        return False
    now = vienna_now()
    wk = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][now.weekday()]
    if wk.lower() in attendance_text.lower():
        return True
    # интервалы вида 14:00–16:00
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

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = (update.message.text or "").strip()
    if not q:
        return
    res = search_by_name(q)
    if res.empty:
        await update.message.reply_text("Ничего не нашла. Попробуй другое имя.")
        return
    for _, row in res.iterrows():
        await update.message.reply_markdown(person_card(row))

async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        if res.empty:
            await q.edit_message_text(f"Никого не нашла в «{loc}».")
            return
        await q.edit_message_text(f"В «{loc}»:")
        for _, row in res.iterrows():
            await q.message.reply_markdown(person_card(row))
    elif data == "time:now":
        subset = DF[DF["Attendance"].apply(is_available_now)]
        if subset.empty:
            await q.edit_message_text("Сейчас никого не вижу по расписанию (или расписание не распознано).")
            return
        await q.edit_message_text("Доступны сейчас:")
        for _, row in subset.iterrows():
            await q.message.reply_markdown(person_card(row))
    else:
        await q.edit_message_text("Ок.")

application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(on_cb))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

@app.get("/", response_class=PlainTextResponse)
def root():
    return "ok"

@app.post("/webhook")
async def webhook(request: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"status":"ok"}
