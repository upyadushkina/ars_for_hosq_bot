import os, re, logging
from datetime import datetime
import zoneinfo
import pandas as pd
from difflib import SequenceMatcher

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ae-bot")

# ---------- MEET SLOTS (normalized search over meet_slots.csv) ----------
import pandas as _pd
from zoneinfo import ZoneInfo as _ZoneInfo

MEET_CSV_PATH = os.getenv("MEET_SLOTS_CSV", "/mnt/data/meet_slots.csv")
MEET_TZ = _ZoneInfo("Europe/Vienna")
MEET_YEAR = 2025

def _meet_norm_key(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^a-z0-9]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s

def _meet_parse_when_to_meet(s: str):
    if not isinstance(s, str) or not s.strip():
        return (None, None, None, None)
    t = s.strip().replace("—","–")
    m = re.match(r'^[A-Za-z]{3}\s+(\d{2})\.(\d{2})\s*,\s*(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})', t)
    if not m:
        m = re.match(r'^(\d{2})\.(\d{2})\s*,\s*(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})', t)
    if not m:
        return (None, None, None, None)
    dd, mm, h1, mi1, h2, mi2 = map(int, m.groups())
    start_dt = datetime(MEET_YEAR, mm, dd, h1, mi1, tzinfo=MEET_TZ)
    end_dt   = datetime(MEET_YEAR, mm, dd, h2, mi2, tzinfo=MEET_TZ)
    return (start_dt, end_dt, f"{start_dt:%Y-%m-%d}", f"{start_dt:%H:%M}–{end_dt:%H:%M}")

def load_meet_df():
    try:
        df = _pd.read_csv(MEET_CSV_PATH)
    except Exception as e:
        log.warning("Failed to load meet slots CSV: %s", e)
        return _pd.DataFrame(columns=["Name","When to Meet","Where to Meet","Event name","Topic","Event type",
                                      "start_dt","end_dt","date","timespan","location_key","topic_key","event_key"])
    col_name = next((c for c in df.columns if c.lower()=="name"), None)
    col_when = next((c for c in df.columns if c.lower().startswith("when")), None)
    col_loc  = next((c for c in df.columns if c.lower().startswith("where")), None)
    col_event= next((c for c in df.columns if "event name" in c.lower() or c.lower().strip()=="event"), None)
    col_topic= next((c for c in df.columns if c.lower().strip()=="topic" or "theme" in c.lower()), None)

    out = _pd.DataFrame()
    out["name"] = df[col_name] if col_name else ""
    out["when_raw"] = df[col_when] if col_when else ""
    out["location"] = df[col_loc] if col_loc else ""
    out["event_name"] = df[col_event] if col_event else ""
    out["topic"] = df[col_topic] if col_topic else ""

    starts, ends, dates, spans = [], [], [], []
    for v in out["when_raw"]:
        st, en, d, sp = _meet_parse_when_to_meet(v)
        starts.append(st); ends.append(en); dates.append(d); spans.append(sp)
    out["start_dt"] = starts
    out["end_dt"] = ends
    out["date"] = dates
    out["timespan"] = spans
    out["location_key"] = out["location"].map(_meet_norm_key)
    out["topic_key"] = out["topic"].map(_meet_norm_key)
    out["event_key"] = out["event_name"].map(_meet_norm_key)
    return out

MEET_DF = load_meet_df()

def meet_format_rows(rows, limit=40):
    if rows is None or len(rows)==0:
        return "Ничего не найдено."
    parts = []
    cut = rows.head(limit)
    for _, r in cut.iterrows():
        name = (r.get("name") or "").strip()
        ev = (r.get("event_name") or "").strip()
        tp = (r.get("topic") or "").strip()
        loc= (r.get("location") or "").strip()
        date = r.get("date") or ""
        span = r.get("timespan") or ""
        line = " · ".join([
            f"👤 {name}" if name else "",
            f"🎫 {ev}" if ev else "",
            f"🏷️ {tp}" if tp else "",
            f"📍 {loc}" if loc else "",
            f"🕒 {date} {span}".strip()
        ])
        parts.append(re.sub(r'\s+·\s+', ' · ', line).strip(" ·"))
    if len(rows) > limit:
        parts.append(f"\n… и ещё {len(rows)-limit}")
    return "\n\n".join(parts)

def meet_list_unique(series):
    vals = sorted({str(v).strip() for v in series.dropna().tolist() if str(v).strip()})
    return vals

def parse_user_time_str(s: str):
    s = (s or "").strip().lower().replace("—","-").replace("–","-")
    if s in ("сейчас", "now"):
        return datetime.now(MEET_TZ)
    m = re.match(r'^(\d{2})\.(\d{2})\s+(\d{1,2}):(\d{2})$', s)
    if m:
        dd, mm, hh, mi = map(int, m.groups())
        return datetime(MEET_YEAR, mm, dd, hh, mi, tzinfo=MEET_TZ)
    m = re.match(r'^(\d{1,2}):(\d{2})$', s)
    if m:
        hh, mi = map(int, m.groups())
        today = datetime.now(MEET_TZ).date()
        return datetime(today.year, today.month, today.day, hh, mi, tzinfo=MEET_TZ)
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})$', s)
    if m:
        yyyy, mm, dd, hh, mi = map(int, m.groups())
        return datetime(yyyy, mm, dd, hh, mi, tzinfo=MEET_TZ)
    return None

# ========== команды для meet_slots ==========
async def ms_time(update, context: ContextTypes.DEFAULT_TYPE):
    kb = ReplyKeyboardMarkup([[KeyboardButton("Сейчас")]], resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text("Введи время (форматы: 'Сейчас', 'HH:MM', 'DD.MM HH:MM')", reply_markup=kb)
    context.user_data["ms_expect_time"] = True

async def ms_text_router(update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("ms_expect_time"):
        txt = (update.message.text or "").strip()
        qdt = parse_user_time_str(txt)
        if not qdt:
            await update.message.reply_text("Не понял время. Примеры: 20:15, 06.09 21:00, Сейчас")
            return
        subset = MEET_DF[(MEET_DF["start_dt"].notna()) & (MEET_DF["end_dt"].notna()) &
                         (MEET_DF["start_dt"] <= qdt) & (qdt < MEET_DF["end_dt"])]
        await update.message.reply_text(meet_format_rows(subset), disable_web_page_preview=True)
        context.user_data["ms_expect_time"] = False
        return

async def ms_topics(update, context: ContextTypes.DEFAULT_TYPE):
    topics = meet_list_unique(MEET_DF["topic"])
    rows, row = [], []
    for i,t in enumerate(topics,1):
        row.append(InlineKeyboardButton(t[:30], callback_data=f"ms:topic:{t}"))
        if len(row)==2:
            rows.append(row); row=[]
    if row: rows.append(row)
    await update.message.reply_text("Выбери тему:", reply_markup=InlineKeyboardMarkup(rows))

async def ms_events(update, context: ContextTypes.DEFAULT_TYPE):
    events = meet_list_unique(MEET_DF["event_name"])
    rows, row = [], []
    for i,e in enumerate(events,1):
        row.append(InlineKeyboardButton(e[:30], callback_data=f"ms:event:{e}"))
        if len(row)==2:
            rows.append(row); row=[]
    if row: rows.append(row)
    await update.message.reply_text("Выбери ивент:", reply_markup=InlineKeyboardMarkup(rows))

async def ms_locations(update, context: ContextTypes.DEFAULT_TYPE):
    locs = meet_list_unique(MEET_DF["location"])
    rows, row = [], []
    for i,l in enumerate(locs,1):
        row.append(InlineKeyboardButton(l[:30], callback_data=f"ms:loc:{l}"))
        if len(row)==2:
            rows.append(row); row=[]
    if row: rows.append(row)
    await update.message.reply_text("Где ты сейчас?", reply_markup=InlineKeyboardMarkup(rows))

async def ms_on_cb(update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    if not data.startswith("ms:"):
        return
    await q.answer()
    _, kind, value = data.split(":", 2)
    if kind == "topic":
        rows = MEET_DF.loc[MEET_DF["topic"]==value]
        await q.edit_message_text(f"Тема: {value}\n\n{meet_format_rows(rows)}", disable_web_page_preview=True)
    elif kind == "event":
        rows = MEET_DF.loc[MEET_DF["event_name"]==value]
        await q.edit_message_text(f"Ивент: {value}\n\n{meet_format_rows(rows)}", disable_web_page_preview=True)
    elif kind == "loc":
        rows = MEET_DF.loc[MEET_DF["location"]==value]
        await q.edit_message_text(f"Локация: {value}\n\n{meet_format_rows(rows)}", disable_web_page_preview=True)

# ========= регистрация новых хендлеров =========
application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
application.add_handler(CommandHandler("time", ms_time))
application.add_handler(CommandHandler("topics", ms_topics))
application.add_handler(CommandHandler("events", ms_events))
application.add_handler(CommandHandler("locations", ms_locations))
application.add_handler(CallbackQueryHandler(ms_on_cb, pattern=r"^ms:"))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ms_text_router), group=0)

app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root():
    return "ok"

@app.post("/webhook")
async def webhook(request: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        log.warning("Bad webhook secret: got=%s", x_telegram_bot_api_secret_token)
        raise HTTPException(status_code=403, detail="bad secret")
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return JSONResponse({"status":"ok"})
    except Exception as e:
        log.exception("webhook error: %s", e)
        return JSONResponse({"status":"error"}, status_code=200)
