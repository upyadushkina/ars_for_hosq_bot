import os, re, logging
from datetime import datetime
from typing import List
import pandas as pd
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.error import TimedOut, NetworkError, BadRequest

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ae-bot")

app = FastAPI()

# =============================================
#            A) PEOPLE (ars_2026_people.csv)
# =============================================
PEOPLE_CSV_PATH = os.getenv("PEOPLE_CSV", "ars_2026_people.csv")

def load_people_df():
    try:
        df = pd.read_csv(PEOPLE_CSV_PATH)
        if "Name" not in df.columns and len(df.columns) > 0:
            df.rename(columns={df.columns[0]: "Name"}, inplace=True)
        return df
    except Exception as e:
        log.warning("Failed to load people CSV: %s", e)
        return pd.DataFrame(columns=[
            "Name", "Institution", "Festival Role", "Role", "Where to Meet",
            "Attendance", "Bio", "Conversation Tip", "Institution Link", "Contact", "Photo"
        ])

PEOPLE_DF = load_people_df()

ALPHABET_ORDER = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def _clean(val):
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    return str(val).strip()

def split_name(name: str):
    parts = re.split(r"[\s\-]+", (name or "").strip())
    parts = [p for p in parts if p]
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) > 1 else ""
    return first, last

def unique_letters():
    letters = set()
    for _, row in PEOPLE_DF.iterrows():
        first, last = split_name(row.get("Name", ""))
        for token in (first, last):
            m = re.search("[A-Za-zА-Яа-яЁё]", token or "")
            if m:
                letters.add(m.group(0).upper())
    ordered = [ch for ch in ALPHABET_ORDER if ch in letters]
    extra = sorted([ch for ch in letters if ch not in ALPHABET_ORDER])
    return ordered + extra

def people_by_letter(letter: str, limit: int = 40):
    def starts_with(token):
        return token.lower().startswith(letter.lower()) if token else False
    mask = []
    for idx, row in PEOPLE_DF.iterrows():
        first, last = split_name(row.get("Name", ""))
        if starts_with(first) or starts_with(last):
            mask.append(idx)
    return PEOPLE_DF.iloc[mask].sort_values("Name").head(limit)

def search_by_name(query, limit: int = 20):
    return PEOPLE_DF[
        PEOPLE_DF["Name"].str.contains(re.escape(query), case=False, na=False)
    ].sort_values("Name").head(limit)

def person_card(row):
    name = _clean(row.get("Name"))
    role = _clean(row.get("Role"))
    bio = _clean(row.get("Bio"))
    tip = _clean(row.get("Conversation Tip"))
    inst = _clean(row.get("Institution"))
    inst_link = _clean(row.get("Institution Link"))
    contact = _clean(row.get("Contact"))

    blocks: List[str] = []
    if name: blocks.append(name)
    if role: blocks.append(role)
    if bio: blocks.append(f"✏️bio: {bio}")
    if tip: blocks.append(f"💡tip: {tip}")
    inst_block = []
    if inst: inst_block.append(inst)
    if inst_link: inst_block.append(inst_link)
    if inst_block: blocks.append("\n".join(inst_block))
    if contact: blocks.append(f"📱inst: {contact}")
    return "\n\n".join(blocks)

async def safe_send_message(message, text, **kwargs):
    """Safely send a message with error handling"""
    try:
        return await message.reply_text(text, **kwargs)
    except (TimedOut, NetworkError) as e:
        log.warning("Network error sending message: %s", e)
        # Try again with a simpler message
        try:
            return await message.reply_text("Сообщение временно недоступно. Попробуйте еще раз.")
        except Exception:
            log.error("Failed to send fallback message")
            return None
    except Exception as e:
        log.error("Error sending message: %s", e)
        return None

async def safe_edit_message(query, text, **kwargs):
    """Safely edit a message with error handling"""
    try:
        return await query.edit_message_text(text, **kwargs)
    except (TimedOut, NetworkError) as e:
        log.warning("Network error editing message: %s", e)
        # Try to send a new message instead
        try:
            return await query.message.reply_text("Обновление временно недоступно. Попробуйте еще раз.")
        except Exception:
            log.error("Failed to send fallback message")
            return None
    except Exception as e:
        log.error("Error editing message: %s", e)
        return None

async def send_person_card(message, row):
    caption = person_card(row)
    photo = _clean(row.get("Photo")) or _clean(row.get("photo"))
    if photo:
        try:
            await message.reply_photo(photo, caption=caption)
            return
        except Exception as e:
            log.warning("Photo send failed for %s: %s", _clean(row.get("Name")), e)
    await safe_send_message(message, caption, disable_web_page_preview=True)

# =============================================
#        B) MEET SLOTS (meet_slots.csv)
# =============================================
MEET_CSV_PATH = os.getenv("MEET_SLOTS_CSV", "meet_slots.csv")
MEET_TZ = ZoneInfo("Europe/Vienna")
MEET_YEAR = 2026

def _meet_norm_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")

def load_meet_df():
    try:
        df = pd.read_csv(MEET_CSV_PATH)
    except Exception as e:
        log.warning("Failed to load meet slots CSV: %s", e)
        return pd.DataFrame(columns=[
            "name", "date", "time_start", "time_finish", "location",
            "event_name", "topic", "start_dt", "end_dt", "timespan",
            "location_key", "topic_key", "event_key"
        ])

    col_name   = next((c for c in df.columns if c.lower() == "name"), None)
    col_date   = next((c for c in df.columns if c.lower().strip() == "date"), None)
    col_start  = next((c for c in df.columns if c.lower().startswith("time start")), None)
    col_finish = next((c for c in df.columns if c.lower().startswith("time finish")), None)
    col_loc    = next((c for c in df.columns if c.lower().startswith("where")), None)
    col_event  = next((c for c in df.columns if "event name" in c.lower() or c.lower().strip() == "event"), None)
    col_topic  = next((c for c in df.columns if c.lower().strip() == "topic" or "theme" in c.lower()), None)

    out = pd.DataFrame()
    out["name"]       = df[col_name]   if col_name   else ""
    out["date_raw"]   = df[col_date]   if col_date   else ""
    out["time_start"] = df[col_start]  if col_start  else ""
    out["time_finish"]= df[col_finish] if col_finish else ""
    out["location"]   = df[col_loc]    if col_loc    else ""
    out["event_name"] = df[col_event]  if col_event  else ""
    out["topic"]      = df[col_topic]  if col_topic  else ""

    starts, ends, dates, spans = [], [], [], []
    for d, ts, tf in zip(out["date_raw"], out["time_start"], out["time_finish"]):
        m = re.match(r"(\d{1,2})\.(\d{1,2})", str(d).strip())
        if not m:
            starts.append(None); ends.append(None); dates.append(""); spans.append(""); continue
        dd, mm = map(int, m.groups())
        try:
            h1, m1 = map(int, str(ts).split(":"))
            h2, m2 = map(int, str(tf).split(":"))
        except Exception:
            starts.append(None); ends.append(None); dates.append(""); spans.append(""); continue
        st = datetime(MEET_YEAR, mm, dd, h1, m1, tzinfo=MEET_TZ)
        en = datetime(MEET_YEAR, mm, dd, h2, m2, tzinfo=MEET_TZ)
        starts.append(st); ends.append(en)
        dates.append(f"{st:%d.%m}")
        spans.append(f"{st:%H:%M}–{en:%H:%M}")
    out["start_dt"] = starts
    out["end_dt"]   = ends
    out["date"]     = dates
    out["timespan"] = spans
    out["location_key"] = out["location"].map(_meet_norm_key)
    out["topic_key"]    = out["topic"].map(_meet_norm_key)
    out["event_key"]    = out["event_name"].map(_meet_norm_key)
    return out

MEET_DF = load_meet_df()

# =============================================
#        C) PERSONAL SCHEDULE (ars_schedule.csv)
# =============================================
SCHEDULE_CSV_PATH = os.getenv("SCHEDULE_CSV", "ars_schedule.csv")

def load_schedule_df():
    try:
        df = pd.read_csv(SCHEDULE_CSV_PATH)
    except Exception as e:
        log.warning("Failed to load schedule CSV: %s", e)
        return pd.DataFrame(columns=[
            "event_name", "people", "date", "time start", "time finish", 
            "event_description", "event_type", "link_to_event", "where", "registration_status"
        ])

    # Normalize column names
    col_name = next((c for c in df.columns if c.lower() == "event_name"), None)
    col_people = next((c for c in df.columns if c.lower() == "people"), None)
    col_date = next((c for c in df.columns if c.lower().strip() == "date"), None)
    col_start = next((c for c in df.columns if c.lower().startswith("time start")), None)
    col_finish = next((c for c in df.columns if c.lower().startswith("time finish")), None)
    col_desc = next((c for c in df.columns if "description" in c.lower()), None)
    col_type = next((c for c in df.columns if "type" in c.lower()), None)
    col_link = next((c for c in df.columns if "link" in c.lower()), None)
    col_where = next((c for c in df.columns if c.lower().strip() == "where"), None)
    col_reg = next((c for c in df.columns if "registration" in c.lower()), None)

    out = pd.DataFrame()
    out["event_name"] = df[col_name] if col_name else ""
    out["people"] = df[col_people] if col_people else ""
    out["date_raw"] = df[col_date] if col_date else ""
    out["time_start"] = df[col_start] if col_start else ""
    out["time_finish"] = df[col_finish] if col_finish else ""
    out["event_description"] = df[col_desc] if col_desc else ""
    out["event_type"] = df[col_type] if col_type else ""
    out["link_to_event"] = df[col_link] if col_link else ""
    out["where"] = df[col_where] if col_where else ""
    out["registration_status"] = df[col_reg] if col_reg else ""

    # Parse dates and times
    starts, ends, dates, spans = [], [], [], []
    for d, ts, tf in zip(out["date_raw"], out["time_start"], out["time_finish"]):
        m = re.match(r"(\d{1,2})\.(\d{1,2})", str(d).strip())
        if not m:
            starts.append(None); ends.append(None); dates.append(""); spans.append(""); continue
        dd, mm = map(int, m.groups())
        try:
            h1, m1 = map(int, str(ts).split(":"))
            h2, m2 = map(int, str(tf).split(":"))
        except Exception:
            starts.append(None); ends.append(None); dates.append(""); spans.append(""); continue
        st = datetime(MEET_YEAR, mm, dd, h1, m1, tzinfo=MEET_TZ)
        en = datetime(MEET_YEAR, mm, dd, h2, m2, tzinfo=MEET_TZ)
        starts.append(st); ends.append(en)
        dates.append(f"{st:%d.%m}")
        spans.append(f"{st:%H:%M}–{en:%H:%M}")
    
    out["start_dt"] = starts
    out["end_dt"] = ends
    out["date"] = dates
    out["timespan"] = spans
    return out

SCHEDULE_DF = load_schedule_df()

def get_schedule_dates():
    """Get unique dates from schedule"""
    dates = sorted({d for d in SCHEDULE_DF["date"].dropna() if d})
    return dates

def get_schedule_events_by_date(date):
    """Get all events for a specific date"""
    return SCHEDULE_DF[SCHEDULE_DF["date"] == date].sort_values("start_dt")

def get_schedule_events_by_hour(date, hour):
    """Get events that include a specific hour on a specific date"""
    dd, mm = map(int, date.split("."))
    qdt = datetime(MEET_YEAR, mm, dd, hour, 0, tzinfo=MEET_TZ)
    subset = SCHEDULE_DF[
        (SCHEDULE_DF["date"] == date) &
        (SCHEDULE_DF["start_dt"].notna()) & (SCHEDULE_DF["end_dt"].notna()) &
        (SCHEDULE_DF["start_dt"] <= qdt) & (qdt < SCHEDULE_DF["end_dt"])
    ]
    return subset.sort_values("start_dt")

def format_schedule_events(rows):
    """Format schedule events for display"""
    if rows is None or len(rows) == 0:
        return "Ничего не найдено."
    
    lines = []
    for _, r in rows.iterrows():
        name = _clean(r.get("event_name"))
        timespan = _clean(r.get("timespan"))
        where = _clean(r.get("where"))
        if name and timespan:
            line = f"🎫 {name}\n{timespan}"
            if where:
                line += f"\n📍 {where}"
            lines.append(line)
    
    return "\n\n".join(lines) if lines else "Ничего не найдено."

def format_schedule_event_card(row):
    """Format a single schedule event card"""
    name = _clean(row.get("event_name"))
    timespan = _clean(row.get("timespan"))
    where = _clean(row.get("where"))
    event_type = _clean(row.get("event_type"))
    description = _clean(row.get("event_description"))
    people = _clean(row.get("people"))
    registration = _clean(row.get("registration_status"))
    link = _clean(row.get("link_to_event"))

    blocks = []
    if name: blocks.append(f"🎫 {name}")
    if timespan: blocks.append(f"🕒 {timespan}")
    if where: blocks.append(f"📍 {where}")
    if event_type: blocks.append(f"🏷️ {event_type}")
    if description: blocks.append(f"📝 {description}")
    if people: blocks.append(f"👥 {people}")
    if registration: blocks.append(f"📋 {registration}")
    if link: blocks.append(f"🔗 {link}")
    
    return "\n\n".join(blocks)

def get_people_from_schedule_event(row):
    """Get people from schedule event that exist in PEOPLE_DF"""
    people_str = _clean(row.get("people"))
    if not people_str:
        return []
    
    # Split people by comma and clean names
    people_names = [name.strip() for name in people_str.split(",") if name.strip()]
    found_people = []
    
    for person_name in people_names:
        # Try to find exact match first
        matches = PEOPLE_DF[PEOPLE_DF["Name"].str.strip().str.lower() == person_name.strip().lower()]
        if not matches.empty:
            found_people.append(matches.iloc[0])
            continue
        
        # Try partial match
        for _, person_row in PEOPLE_DF.iterrows():
            person_full_name = _clean(person_row.get("Name"))
            if person_name.lower() in person_full_name.lower() or person_full_name.lower() in person_name.lower():
                found_people.append(person_row)
                break
    
    return found_people

def format_people_times(rows):
    if rows is None or len(rows) == 0:
        return "Ничего не найдено."
    by_name: dict[str, List[str]] = {}
    for _, r in rows.iterrows():
        nm  = _clean(r.get("name"))
        st  = r.get("start_dt")
        en  = r.get("end_dt")
        loc = _clean(r.get("location"))
        if not nm or not isinstance(st, datetime) or not isinstance(en, datetime):
            continue
        entry = f"{st:%d.%m %H:%M}–{en:%H:%M}\n📍 {loc}"
        by_name.setdefault(nm, []).append(entry)
    lines: List[str] = []
    for nm in sorted(by_name.keys(), key=lambda s: s.lower()):
        parts = [f"👤 {nm}"] + by_name[nm]
        lines.append("\n".join(parts))
    return "\n\n".join(lines) if lines else "Ничего не найдено."

# =============================================
#                 BOT UI / HANDLERS
# =============================================
application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

@app.on_event("startup")
async def on_startup():
    await application.initialize()
    await application.start()

@app.on_event("shutdown")
async def on_shutdown():
    await application.stop()
    await application.shutdown()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text(
            "Главное меню:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔎 Поиск по имени", callback_data="name:menu")],
                [InlineKeyboardButton("📍 По локации", callback_data="ms:loc_menu")],
                [InlineKeyboardButton("🕒 По времени", callback_data="ms:time_menu")],
                [InlineKeyboardButton("🏷️ По теме", callback_data="ms:topic_menu")],
                [InlineKeyboardButton("🎫 По ивенту", callback_data="ms:event_menu")],
                [InlineKeyboardButton("📅 Моё расписание", callback_data="schedule:menu")],
            ])
        )
    except (TimedOut, NetworkError) as e:
        log.warning("Network error in start command: %s", e)
        # Try to send a simple text message without keyboard
        try:
            await update.message.reply_text("Главное меню загружается... Попробуйте /start еще раз.")
        except Exception:
            log.error("Failed to send fallback message")
    except Exception as e:
        log.error("Unexpected error in start command: %s", e)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("expect_name_typing"):
        q = (update.message.text or "").strip()
        res = search_by_name(q, limit=20)
        context.user_data["expect_name_typing"] = False
        if res.empty:
            await update.message.reply_text(
                "Ничего не нашлось. Попробуй по-другому.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back:home")]]),
            )
            return
        records = res.reset_index().to_dict("records")
        context.user_data["name_results"] = records
        context.user_data["name_prev_mode"] = "typing"
        context.user_data["name_prev_arg"] = q
        rows, row = [], []
        for i, r in enumerate(records):
            row.append(InlineKeyboardButton(r.get("Name", "")[:30] or "—", callback_data=f"name:person#{i}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ В меню", callback_data="back:home")])
        await update.message.reply_text("Выбери человека:", reply_markup=InlineKeyboardMarkup(rows))

async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception as e:
        log.warning("Failed to answer callback query: %s", e)
    
    data = q.data or ""

    if data == "back:home" or data == "home:menu":
        await q.edit_message_text(
            "Главное меню:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔎 Поиск по имени", callback_data="name:menu")],
                [InlineKeyboardButton("📍 По локации", callback_data="ms:loc_menu")],
                [InlineKeyboardButton("🕒 По времени", callback_data="ms:time_menu")],
                [InlineKeyboardButton("🏷️ По теме", callback_data="ms:topic_menu")],
                [InlineKeyboardButton("🎫 По ивенту", callback_data="ms:event_menu")],
                [InlineKeyboardButton("📅 Моё расписание", callback_data="schedule:menu")],
                [InlineKeyboardButton("🔄 Перезапустить", callback_data="restart:bot")],
            ])
        )
        return

    # ---------- RESTART BOT ----------
    if data == "restart:bot":
        # Clear user data to reset the bot state
        context.user_data.clear()
        await q.edit_message_text(
            "🔄 Бот перезапущен!\n\nГлавное меню:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔎 Поиск по имени", callback_data="name:menu")],
                [InlineKeyboardButton("📍 По локации", callback_data="ms:loc_menu")],
                [InlineKeyboardButton("🕒 По времени", callback_data="ms:time_menu")],
                [InlineKeyboardButton("🏷️ По теме", callback_data="ms:topic_menu")],
                [InlineKeyboardButton("🎫 По ивенту", callback_data="ms:event_menu")],
                [InlineKeyboardButton("📅 Моё расписание", callback_data="schedule:menu")],
                [InlineKeyboardButton("🔄 Перезапустить", callback_data="restart:bot")],
            ])
        )
        return

    # ---------- NAME SEARCH ----------
    if data == "name:menu":
        await q.edit_message_text(
            "Как искать по имени?",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔤 Имена в алфавитном порядке", callback_data="name:alpha")],
                [InlineKeyboardButton("⌨️ Введите имя", callback_data="name:typing")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="back:home")],
                [InlineKeyboardButton("🔄 Перезапустить", callback_data="restart:bot")],
            ]),
        )
        return

    if data == "name:alpha":
        letters = unique_letters()
        rows, row = [], []
        for ch in letters:
            row.append(InlineKeyboardButton(ch, callback_data=f"name:letter:{ch}"))
            if len(row) == 8:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        await q.edit_message_text("Выбери букву:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("name:letter:"):
        letter = data.split(":", 2)[2]
        res = people_by_letter(letter)
        if res.empty:
            await q.edit_message_text(
                "Ничего не найдено.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="name:alpha")]]),
            )
            return
        records = res.reset_index().to_dict("records")
        context.user_data["name_results"] = records
        context.user_data["name_prev_mode"] = "letter"
        context.user_data["name_prev_arg"] = letter
        rows, row = [], []
        for i, r in enumerate(records):
            row.append(InlineKeyboardButton(r.get("Name", "")[:30] or "—", callback_data=f"name:person#{i}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад к буквам", callback_data="name:alpha")])
        rows.append([InlineKeyboardButton("⬅️ В меню", callback_data="back:home")])
        await q.edit_message_text(f"Имена на букву {letter}:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data == "name:list":
        records = context.user_data.get("name_results", [])
        mode = context.user_data.get("name_prev_mode")
        arg = context.user_data.get("name_prev_arg")
        if not records:
            await q.edit_message_text(
                "Список пуст.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ В меню", callback_data="back:home")]]),
            )
            return
        rows, row = [], []
        for i, r in enumerate(records):
            row.append(InlineKeyboardButton(r.get("Name", "")[:30] or "—", callback_data=f"name:person#{i}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row: rows.append(row)
        if mode == "letter":
            rows.append([InlineKeyboardButton("⬅️ Назад к буквам", callback_data="name:alpha")])
        elif mode == "typing":
            rows.append([InlineKeyboardButton("⬅️ Ввести заново", callback_data="name:typing")])
        rows.append([InlineKeyboardButton("⬅️ В меню", callback_data="back:home")])
        title = f"Имена на букву {arg}:" if mode == "letter" else "Результаты поиска:"
        await q.edit_message_text(title, reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("name:person#"):
        try:
            idx = int(data.split("#", 1)[1])
        except Exception:
            idx = -1
        records = context.user_data.get("name_results", [])
        if 0 <= idx < len(records):
            rec = records[idx]
            row = PEOPLE_DF.loc[rec["index"]]
            await send_person_card(q.message, row)
            await q.message.reply_text(
                "Что дальше?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("⬅️ Назад к людям", callback_data="name:list")],
                    [InlineKeyboardButton("⬅️ В меню", callback_data="back:home")],
                ]),
            )
        else:
            await q.edit_message_text("Не удалось найти человека.")
        return

    if data == "name:typing":
        context.user_data["expect_name_typing"] = True
        await q.edit_message_text("Введи имя/фамилию для поиска:")
        return

    # ---------- MEET SLOTS: LOCATION ----------
    if data == "ms:loc_menu":
        locs = sorted({str(v).strip() for v in MEET_DF["location"].dropna() if str(v).strip()})
        context.user_data["_locs"] = locs
        rows, row = [], []
        for i, l in enumerate(locs):
            row.append(InlineKeyboardButton(l[:30] or "—", callback_data=f"ms:loc#{i}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        rows.append([InlineKeyboardButton("🔄 Перезапустить", callback_data="restart:bot")])
        await q.edit_message_text("Где ты сейчас?", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:loc#"):
        try:
            idx = int(data.split("#", 1)[1])
        except Exception:
            idx = -1
        locs = context.user_data.get("_locs", [])
        if 0 <= idx < len(locs):
            loc = locs[idx]
            subset = MEET_DF[MEET_DF["location"] == loc]
            people = sorted({str(n).strip() for n in subset["name"].dropna() if str(n).strip()})
            context.user_data["_people_from_loc"] = people
            text = f"Локация: {loc}\n\n" + format_people_times(subset)
            kb_rows, r = [], []
            for i, nm in enumerate(people):
                r.append(InlineKeyboardButton(nm[:30] or "—", callback_data=f"ms:loc_person#{i}"))
                if len(r) == 2:
                    kb_rows.append(r); r = []
            if r: kb_rows.append(r)
            kb_rows.append([InlineKeyboardButton("⬅️ Назад к локациям", callback_data="ms:loc_menu")])
            await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb_rows), disable_web_page_preview=True)
        else:
            await q.edit_message_text("Ошибка выбора локации")
        return

    if data.startswith("ms:loc_person#"):
        try:
            idx = int(data.split("#", 1)[1])
        except Exception:
            idx = -1
        people = context.user_data.get("_people_from_loc", [])
        if 0 <= idx < len(people):
            nm = people[idx]
            row = PEOPLE_DF[PEOPLE_DF["Name"].str.strip().str.lower() == nm.strip().lower()]
            if not row.empty:
                await send_person_card(q.message, row.iloc[0])
                await q.message.reply_text(
                    "Что дальше?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⬅️ Назад к людям", callback_data="ms:loc_menu")],
                        [InlineKeyboardButton("⬅️ В меню", callback_data="back:home")],
                    ]),
                )
            else:
                await q.edit_message_text("Не удалось найти человека.")
        else:
            await q.edit_message_text("Не удалось найти человека.")
        return

    # ---------- MEET SLOTS: TIME ----------
    if data == "ms:time_menu":
        dates = ["08.09", "09.09", "10.09", "11.09", "12.09", "13.09"]
        rows = []
        row = []
        for d in dates:
            row.append(InlineKeyboardButton(d, callback_data=f"ms:time_date#{d}"))
        rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        rows.append([InlineKeyboardButton("🔄 Перезапустить", callback_data="restart:bot")])
        await q.edit_message_text("Выбери дату:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:time_date#"):
        date = data.split("#", 1)[1]
        hours = [f"{h:02d}:00" for h in list(range(10, 24)) + [0]]
        rows, row = [], []
        for h in hours:
            row.append(InlineKeyboardButton(h, callback_data=f"ms:time_hour#{date}#{h[:2]}"))
            if len(row) == 4:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад к датам", callback_data="ms:time_menu")])
        await q.edit_message_text("Который час?", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:time_hour#"):
        _, date, hour_str = data.split("#")
        hour = int(hour_str)
        dd, mm = map(int, date.split("."))
        qdt = datetime(MEET_YEAR, mm, dd, hour, 0, tzinfo=MEET_TZ)
        subset = MEET_DF[
            (MEET_DF["start_dt"].notna()) & (MEET_DF["end_dt"].notna()) &
            (MEET_DF["start_dt"] <= qdt) & (qdt < MEET_DF["end_dt"])
        ]
        people = sorted({str(n).strip() for n in subset["name"].dropna() if str(n).strip()})
        context.user_data["_people_from_time"] = people
        text = f"{date} {hour:02d}:00\n\n" + format_people_times(subset)
        kb_rows, r = [], []
        for i, nm in enumerate(people):
            r.append(InlineKeyboardButton(nm[:30] or "—", callback_data=f"ms:time_person#{i}"))
            if len(r) == 2:
                kb_rows.append(r); r = []
        if r: kb_rows.append(r)
        kb_rows.append([InlineKeyboardButton("⬅️ Назад к часам", callback_data=f"ms:time_date#{date}")])
        await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb_rows), disable_web_page_preview=True)
        return

    if data.startswith("ms:time_person#"):
        try:
            idx = int(data.split("#", 1)[1])
        except Exception:
            idx = -1
        people = context.user_data.get("_people_from_time", [])
        if 0 <= idx < len(people):
            nm = people[idx]
            row = PEOPLE_DF[PEOPLE_DF["Name"].str.strip().str.lower() == nm.strip().lower()]
            if not row.empty:
                await send_person_card(q.message, row.iloc[0])
                await q.message.reply_text(
                    "Что дальше?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⬅️ Назад к людям", callback_data="ms:time_menu")],
                        [InlineKeyboardButton("⬅️ В меню", callback_data="back:home")],
                    ]),
                )
            else:
                await q.edit_message_text("Не удалось найти человека.")
        else:
            await q.edit_message_text("Не удалось найти человека.")
        return

    # ---------- MEET SLOTS: TOPIC ----------
    if data == "ms:topic_menu":
        topics = sorted({str(v).strip() for v in MEET_DF["topic"].dropna() if str(v).strip()})
        context.user_data["_topics"] = topics
        rows, row = [], []
        for i, t in enumerate(topics):
            row.append(InlineKeyboardButton(t[:30] or "—", callback_data=f"ms:topic#{i}"))
            if len(row) == 3:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        rows.append([InlineKeyboardButton("🔄 Перезапустить", callback_data="restart:bot")])
        await q.edit_message_text("Выбери тему:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:topic#"):
        try:
            idx = int(data.split("#", 1)[1])
        except Exception:
            idx = -1
        topics = context.user_data.get("_topics", [])
        if 0 <= idx < len(topics):
            topic = topics[idx]
            rows = MEET_DF[MEET_DF["topic"] == topic]
            people = sorted({str(n).strip() for n in rows["name"].dropna() if str(n).strip()})
            context.user_data["_people_from_topic"] = people
            text = f"Тема: {topic}\n\n" + format_people_times(rows)
            kb_rows, r = [], []
            for i, nm in enumerate(people):
                r.append(InlineKeyboardButton(nm[:30] or "—", callback_data=f"ms:topic_person#{i}"))
                if len(r) == 2:
                    kb_rows.append(r); r = []
            if r: kb_rows.append(r)
            kb_rows.append([InlineKeyboardButton("⬅️ Назад к темам", callback_data="ms:topic_menu")])
            await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb_rows), disable_web_page_preview=True)
        else:
            await q.edit_message_text("Ошибка выбора темы")
        return

    if data.startswith("ms:topic_person#"):
        try:
            idx = int(data.split("#", 1)[1])
        except Exception:
            idx = -1
        people = context.user_data.get("_people_from_topic", [])
        if 0 <= idx < len(people):
            nm = people[idx]
            row = PEOPLE_DF[PEOPLE_DF["Name"].str.strip().str.lower() == nm.strip().lower()]
            if not row.empty:
                await send_person_card(q.message, row.iloc[0])
                await q.message.reply_text(
                    "Что дальше?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⬅️ Назад к людям", callback_data="ms:topic_menu")],
                        [InlineKeyboardButton("⬅️ В меню", callback_data="back:home")],
                    ]),
                )
            else:
                await q.edit_message_text("Не удалось найти человека.")
        else:
            await q.edit_message_text("Не удалось найти человека.")
        return

    # ---------- MEET SLOTS: EVENT ----------
    if data == "ms:event_menu":
        events = sorted({str(v).strip() for v in MEET_DF["event_name"].dropna() if str(v).strip()})
        context.user_data["_events"] = events
        rows, row = [], []
        for i, e in enumerate(events):
            row.append(InlineKeyboardButton(e[:30] or "—", callback_data=f"ms:event#{i}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        rows.append([InlineKeyboardButton("🔄 Перезапустить", callback_data="restart:bot")])
        await q.edit_message_text("Выбери ивент:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:event#"):
        try:
            idx = int(data.split("#", 1)[1])
        except Exception:
            idx = -1
        events = context.user_data.get("_events", [])
        if 0 <= idx < len(events):
            ev = events[idx]
            rows = MEET_DF[MEET_DF["event_name"] == ev]
            people = sorted({str(n).strip() for n in rows["name"].dropna() if str(n).strip()})
            context.user_data["_people_from_event"] = people
            text = f"Ивент: {ev}\n\n" + format_people_times(rows)
            kb_rows, r = [], []
            for i, nm in enumerate(people):
                r.append(InlineKeyboardButton(nm[:30] or "—", callback_data=f"ms:event_person#{i}"))
                if len(r) == 2:
                    kb_rows.append(r); r = []
            if r: kb_rows.append(r)
            kb_rows.append([InlineKeyboardButton("⬅️ Назад к ивентам", callback_data="ms:event_menu")])
            await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb_rows), disable_web_page_preview=True)
        else:
            await q.edit_message_text("Ошибка выбора ивента")
        return

    if data.startswith("ms:event_person#"):
        try:
            idx = int(data.split("#", 1)[1])
        except Exception:
            idx = -1
        people = context.user_data.get("_people_from_event", [])
        if 0 <= idx < len(people):
            nm = people[idx]
            row = PEOPLE_DF[PEOPLE_DF["Name"].str.strip().str.lower() == nm.strip().lower()]
            if not row.empty:
                await send_person_card(q.message, row.iloc[0])
                await q.message.reply_text(
                    "Что дальше?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⬅️ Назад к людям", callback_data="ms:event_menu")],
                        [InlineKeyboardButton("⬅️ В меню", callback_data="back:home")],
                    ]),
                )
            else:
                await q.edit_message_text("Не удалось найти человека.")
        else:
            await q.edit_message_text("Не удалось найти человека.")
        return

    # ---------- SCHEDULE ----------
    if data == "schedule:menu":
        dates = get_schedule_dates()
        if not dates:
            await q.edit_message_text(
                "В расписании нет событий.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("⬅️ Назад", callback_data="back:home")],
                    [InlineKeyboardButton("🔄 Перезапустить", callback_data="restart:bot")]
                ]),
            )
            return
        
        rows, row = [], []
        for date in dates:
            row.append(InlineKeyboardButton(date, callback_data=f"schedule:date#{date}"))
            if len(row) == 2:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        rows.append([InlineKeyboardButton("🔄 Перезапустить", callback_data="restart:bot")])
        await q.edit_message_text("Выбери дату:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("schedule:date#"):
        date = data.split("#", 1)[1]
        hours = [f"{h:02d}:00" for h in list(range(10, 24)) + [0]]
        rows, row = [], []
        
        # Add "full day" button first
        row.append(InlineKeyboardButton("📅 Весь день", callback_data=f"schedule:fullday#{date}"))
        if len(row) == 2:
            rows.append(row); row = []
        
        for h in hours:
            row.append(InlineKeyboardButton(h, callback_data=f"schedule:hour#{date}#{h[:2]}"))
            if len(row) == 4:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад к датам", callback_data="schedule:menu")])
        await q.edit_message_text(f"Выбери время для {date}:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("schedule:hour#"):
        _, date, hour_str = data.split("#")
        hour = int(hour_str)
        events = get_schedule_events_by_hour(date, hour)
        
        if events.empty:
            await q.edit_message_text(
                f"На {date} в {hour:02d}:00 нет событий.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"schedule:date#{date}")]]),
            )
            return
        
        text = f"{date} {hour:02d}:00\n\n" + format_schedule_events(events)
        records = events.reset_index().to_dict("records")
        context.user_data["schedule_results"] = records
        context.user_data["schedule_prev_date"] = date
        context.user_data["schedule_prev_hour"] = hour
        
        kb_rows, r = [], []
        for i, event in enumerate(records):
            event_name = _clean(event.get("event_name", ""))
            r.append(InlineKeyboardButton(event_name[:30] or "—", callback_data=f"schedule:event#{i}"))
            if len(r) == 2:
                kb_rows.append(r); r = []
        if r: kb_rows.append(r)
        kb_rows.append([InlineKeyboardButton("⬅️ Назад к времени", callback_data=f"schedule:date#{date}")])
        await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb_rows), disable_web_page_preview=True)
        return

    if data.startswith("schedule:fullday#"):
        date = data.split("#", 1)[1]
        events = get_schedule_events_by_date(date)
        
        if events.empty:
            await q.edit_message_text(
                f"На {date} нет событий.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"schedule:date#{date}")]]),
            )
            return
        
        text = f"Расписание на {date}\n\n" + format_schedule_events(events)
        records = events.reset_index().to_dict("records")
        context.user_data["schedule_results"] = records
        context.user_data["schedule_prev_date"] = date
        context.user_data["schedule_prev_hour"] = "fullday"
        
        kb_rows, r = [], []
        for i, event in enumerate(records):
            event_name = _clean(event.get("event_name", ""))
            r.append(InlineKeyboardButton(event_name[:30] or "—", callback_data=f"schedule:event#{i}"))
            if len(r) == 2:
                kb_rows.append(r); r = []
        if r: kb_rows.append(r)
        kb_rows.append([InlineKeyboardButton("⬅️ Назад к времени", callback_data=f"schedule:date#{date}")])
        await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb_rows), disable_web_page_preview=True)
        return

    if data.startswith("schedule:event#"):
        try:
            idx = int(data.split("#", 1)[1])
        except Exception:
            idx = -1
        
        records = context.user_data.get("schedule_results", [])
        if 0 <= idx < len(records):
            event_record = records[idx]
            # Get the actual row from SCHEDULE_DF
            event_row = SCHEDULE_DF.loc[event_record["index"]]
            
            # Format and send event card
            event_card = format_schedule_event_card(event_row)
            await q.message.reply_text(event_card, disable_web_page_preview=True)
            
            # Get people from this event
            people = get_people_from_schedule_event(event_row)
            
            if people:
                kb_rows, r = [], []
                for i, person in enumerate(people):
                    person_name = _clean(person.get("Name", ""))
                    r.append(InlineKeyboardButton(person_name[:30] or "—", callback_data=f"schedule:person#{idx}#{i}"))
                    if len(r) == 2:
                        kb_rows.append(r); r = []
                if r: kb_rows.append(r)
                kb_rows.append([InlineKeyboardButton("⬅️ Назад к событиям", callback_data="schedule:back_to_events")])
                await q.message.reply_text("Люди на этом событии:", reply_markup=InlineKeyboardMarkup(kb_rows))
            else:
                await q.message.reply_text(
                    "Что дальше?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⬅️ Назад к событиям", callback_data="schedule:back_to_events")],
                    ]),
                )
        else:
            await q.edit_message_text("Не удалось найти событие.")
        return

    if data.startswith("schedule:person#"):
        try:
            event_idx, person_idx = map(int, data.split("#")[1:])
        except Exception:
            await q.edit_message_text("Ошибка выбора человека.")
            return
        
        records = context.user_data.get("schedule_results", [])
        if 0 <= event_idx < len(records):
            event_record = records[event_idx]
            event_row = SCHEDULE_DF.loc[event_record["index"]]
            people = get_people_from_schedule_event(event_row)
            
            if 0 <= person_idx < len(people):
                person = people[person_idx]
                await send_person_card(q.message, person)
                await q.message.reply_text(
                    "Что дальше?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⬅️ Назад к людям", callback_data=f"schedule:event#{event_idx}")],
                        [InlineKeyboardButton("⬅️ Назад к событиям", callback_data="schedule:back_to_events")],
                    ]),
                )
            else:
                await q.edit_message_text("Не удалось найти человека.")
        else:
            await q.edit_message_text("Не удалось найти событие.")
        return

    if data == "schedule:back_to_events":
        records = context.user_data.get("schedule_results", [])
        date = context.user_data.get("schedule_prev_date")
        hour = context.user_data.get("schedule_prev_hour")
        
        if not records or not date:
            await q.edit_message_text("Ошибка навигации.")
            return
        
        if hour == "fullday":
            text = f"Расписание на {date}\n\n" + format_schedule_events(SCHEDULE_DF[SCHEDULE_DF["date"] == date])
        else:
            events = get_schedule_events_by_hour(date, int(hour))
            text = f"{date} {int(hour):02d}:00\n\n" + format_schedule_events(events)
        
        kb_rows, r = [], []
        for i, event in enumerate(records):
            event_name = _clean(event.get("event_name", ""))
            r.append(InlineKeyboardButton(event_name[:30] or "—", callback_data=f"schedule:event#{i}"))
            if len(r) == 2:
                kb_rows.append(r); r = []
        if r: kb_rows.append(r)
        kb_rows.append([InlineKeyboardButton("⬅️ Назад к времени", callback_data=f"schedule:date#{date}")])
        await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb_rows), disable_web_page_preview=True)
        return

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors that occur during bot operation"""
    log.error("Exception while handling an update:", exc_info=context.error)
    
    # Try to send a user-friendly error message
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Произошла ошибка. Попробуйте еще раз или используйте /start для перезапуска."
            )
        except Exception:
            log.error("Failed to send error message to user")

# ====== handlers registration ======
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(on_cb))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

# Add error handler
application.add_error_handler(error_handler)

# ====== FastAPI endpoints ======
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
        return JSONResponse({"status": "ok"})
    except Exception as e:
        log.exception("webhook error: %s", e)
        return JSONResponse({"status": "error"}, status_code=200)
