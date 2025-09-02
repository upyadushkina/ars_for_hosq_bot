
import os, re, logging
from datetime import datetime
from typing import List
import pandas as pd
from difflib import SequenceMatcher

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ae-bot")

# ---------- config ----------
TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
if not TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN env var")

# =============================================
#            A) PEOPLE (ars_2025_people.csv)
# =============================================
PEOPLE_CSV_PATH = os.getenv("PEOPLE_CSV", "/mnt/data/ars_2025_people.csv")
DF = pd.DataFrame()

def load_people_df():
    global DF
    try:
        DF = pd.read_csv(PEOPLE_CSV_PATH)
        # Ensure 'Name' column exists
        if "Name" not in DF.columns and len(DF.columns) > 0:
            DF.rename(columns={DF.columns[0]: "Name"}, inplace=True)
    except Exception as e:
        log.warning("Failed to load people CSV: %s", e)
        DF = pd.DataFrame(columns=["Name", "Institution", "Festival Role", "Where to Meet", "Attendance"])

load_people_df()

ALPHABET_ORDER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

def split_name(name: str):
    name = str(name or "").strip()
    if not name:
        return ("", "")
    parts = re.split(r"\s+", name)
    first = parts[0]
    last = parts[-1] if len(parts) > 1 else ""
    return (first, last)

def unique_letters() -> List[str]:
    letters = set()
    for v in DF.get("Name", pd.Series(dtype=str)).fillna(""):
        first, last = split_name(v)
        for token in [first, last]:
            if token:
                letters.add(token[0].upper())
    letters = [ch for ch in ALPHABET_ORDER if ch in letters]
    return letters

def letters_keyboard():
    letters = unique_letters()
    rows, row = [], []
    for i, ch in enumerate(letters, 1):
        row.append(InlineKeyboardButton(ch, callback_data=f"name:letter:{ch}"))
        if len(row) == 8:
            rows.append(row); row = []
    if row: rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
    return InlineKeyboardMarkup(rows)

def people_by_letter(letter: str, limit=40):
    letter = (letter or "").upper()
    mask_rows = []
    for idx, v in DF.get("Name", pd.Series(dtype=str)).fillna("").items():
        first, last = split_name(v)
        if first.upper().startswith(letter) or last.upper().startswith(letter):
            mask_rows.append(idx)
    subset = DF.iloc[mask_rows].head(limit)
    return subset

def person_card(row):
    parts = [str(row.get('Name','')).strip()]
    inst = str(row.get('Institution','')).strip() if 'Institution' in row else ''
    role = str(row.get('Festival Role','')).strip() if 'Festival Role' in row else ''
    if inst or role:
        parts.append(" — ".join([x for x in [role, inst] if x]))
    if row.get('Where to Meet',''):
        parts.append(f"📍 {row['Where to Meet']}")
    if row.get('Attendance',''):
        parts.append(f"🕒 {row['Attendance']}")
    return "\n".join([p for p in parts if p])

def _ratio(a,b):
    try:
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    except Exception:
        return 0.0

def search_by_name(query, limit=20):
    query = (query or "").strip()
    if not query:
        return DF.head(limit)
    scores = []
    for idx, name in DF.get("Name", pd.Series(dtype=str)).fillna("").items():
        scores.append((idx, _ratio(name, query)))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_idx = [i for i,_ in scores[: max(limit, 1)]]
    return DF.loc[top_idx]

# =============================================
#        B) MEET SLOTS (meet_slots.csv)
# =============================================
from zoneinfo import ZoneInfo as _ZoneInfo
import pandas as _pd

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
    # "WED 03.09, 11:00 – 12:00" (weekday ignored)
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

# =============================================
#                 BOT UI / HANDLERS
# =============================================
def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔎 Поиск по имени", callback_data="name:menu")],
        [InlineKeyboardButton("📍 По локации (meet slots)", callback_data="ms:loc_menu")],
        [InlineKeyboardButton("🕒 По времени (meet slots)", callback_data="ms:time_menu")],
        [InlineKeyboardButton("🏷️ По теме (meet slots)", callback_data="ms:topic_menu")],
        [InlineKeyboardButton("🎫 По ивенту (meet slots)", callback_data="ms:event_menu")],
    ])

def name_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔤 Имена в алфавитном порядке", callback_data="name:alpha")],
        [InlineKeyboardButton("⌨️ Введите имя", callback_data="name:typing")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back:home")],
    ])

# Single app and application
app = FastAPI()
application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

# ---------- COMMANDS ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Главное меню:", reply_markup=main_menu_keyboard())

async def ms_time(update, context: ContextTypes.DEFAULT_TYPE):
    kb = ReplyKeyboardMarkup([[KeyboardButton("Сейчас")]], resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text("Введи время (форматы: 'Сейчас', 'HH:MM', 'DD.MM HH:MM')", reply_markup=kb)
    context.user_data["ms_expect_time"] = True

# ---------- TEXT ROUTER ----------
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # name typing mode
    if context.user_data.get("expect_name_typing"):
        q = (update.message.text or "").strip()
        res = search_by_name(q, limit=20)
        if res.empty:
            await update.message.reply_text("Ничего не нашлось. Попробуй по-другому.")
        else:
            for _, row in res.iterrows():
                await update.message.reply_text(person_card(row), disable_web_page_preview=True)
        context.user_data["expect_name_typing"] = False
        await update.message.reply_text("Вернуться в меню?", reply_markup=main_menu_keyboard())
        return

    # meet time input mode
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
        await update.message.reply_text("Вернуться в меню?", reply_markup=main_menu_keyboard())
        return

# ---------- CALLBACKS ----------
async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if data == "back:home":
        await q.edit_message_text("Главное меню:", reply_markup=main_menu_keyboard())
        return

    # --- name ---
    if data == "name:menu":
        await q.edit_message_text("Как искать по имени?", reply_markup=name_menu_keyboard())
        return

    if data == "name:alpha":
        await q.edit_message_text("Выбери букву:", reply_markup=letters_keyboard())
        return

    if data.startswith("name:letter:"):
        letter = data.split(":",2)[2]
        res = people_by_letter(letter)
        await q.edit_message_text(f"Имена на букву {letter}:")
        for _, row in res.iterrows():
            await q.message.reply_text(person_card(row), disable_web_page_preview=True)
        await q.message.reply_text("Вернуться в меню?", reply_markup=main_menu_keyboard())
        return

    if data == "name:typing":
        context.user_data["expect_name_typing"] = True
        await q.edit_message_text("Введи имя/фамилию для поиска:")
        return

    # --- meet slots flows ---
    if data == "ms:time_menu":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Сейчас", callback_data="ms:time:now")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back:home")],
        ])
        await q.edit_message_text("Введи время (форматы: 'Сейчас', 'HH:MM', 'DD.MM HH:MM')", reply_markup=kb)
        context.user_data["ms_expect_time"] = True
        return

    if data == "ms:time:now":
        now_dt = datetime.now(MEET_TZ)
        subset = MEET_DF[(MEET_DF["start_dt"].notna()) & (MEET_DF["end_dt"].notna()) &
                         (MEET_DF["start_dt"] <= now_dt) & (now_dt < MEET_DF["end_dt"])]
        await q.edit_message_text("Доступны сейчас:\n\n" + meet_format_rows(subset))
        context.user_data["ms_expect_time"] = False
        return

    if data == "ms:topic_menu":
        topics = meet_list_unique(MEET_DF["topic"])
        rows, row = [], []
        for t in topics:
            row.append(InlineKeyboardButton(t[:30], callback_data=f"ms:topic:{t}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        await q.edit_message_text("Выбери тему:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:topic:"):
        topic = data.split(":",2)[2]
        rows = MEET_DF.loc[MEET_DF["topic"]==topic]
        await q.edit_message_text(f"Тема: {topic}\n\n{meet_format_rows(rows)}", disable_web_page_preview=True)
        return

    if data == "ms:event_menu":
        events = meet_list_unique(MEET_DF["event_name"])
        rows, row = [], []
        for e in events:
            row.append(InlineKeyboardButton(e[:30], callback_data=f"ms:event:{e}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        await q.edit_message_text("Выбери ивент:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:event:"):
        ev = data.split(":",2)[2]
        rows = MEET_DF.loc[MEET_DF["event_name"]==ev]
        await q.edit_message_text(f"Ивент: {ev}\n\n{meet_format_rows(rows)}", disable_web_page_preview=True)
        return

    if data == "ms:loc_menu":
        locs = meet_list_unique(MEET_DF["location"])
        rows, row = [], []
        for l in locs:
            row.append(InlineKeyboardButton(l[:30], callback_data=f"ms:loc:{l}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        await q.edit_message_text("Где ты сейчас?", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:loc:"):
        loc = data.split(":",2)[2]
        rows = MEET_DF.loc[MEET_DF["location"]==loc]
        await q.edit_message_text(f"Локация: {loc}\n\n{meet_format_rows(rows)}", disable_web_page_preview=True)
        return

# ====== handlers registration ======
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(on_cb))
application.add_handler(CommandHandler("time", ms_time))  # command mirror for convenience
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

# ====== FastAPI endpoints ======
@app.get("/", response_class=PlainTextResponse)
def root():
    return "ok"

@app.post("/webhook")
async def webhook(request: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)):
    # Optional webhook secret validation
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
