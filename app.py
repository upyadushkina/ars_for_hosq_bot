import os, re, logging
from datetime import datetime
from typing import List
import pandas as pd
from difflib import SequenceMatcher
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from zoneinfo import ZoneInfo

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ae-bot")

# Create FastAPI app EARLY so decorators below can reference it
app = FastAPI()

# =============================================
#            A) PEOPLE (ars_2025_people.csv)
# =============================================
PEOPLE_CSV_PATH = os.getenv("PEOPLE_CSV", "ars_2025_people.csv")

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
            "Attendance", "Bio", "Conversation Tip", "Institution Link", "Contact"
        ])

PEOPLE_DF = load_people_df()

# ---------- helpers ----------
ALPHABET_ORDER = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def _clean(value):
    """Return a trimmed string or empty string for NaN/None."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass    # not pandas
    return str(value).strip()

def get_alpha_char(s: str) -> str | None:
    m = re.search("[A-Za-zА-Яа-яЁё]", s or "")
    if not m:
        return None
    return m.group(0).upper()

def split_name(name: str):
    parts = re.split("[\\s\\-]+", (name or "").strip())
    parts = [p for p in parts if p]
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) > 1 else ""
    return first, last

def unique_letters():
    """Return list of unique initial letters present in the people table."""
    letters = set()
    for _, row in PEOPLE_DF.iterrows():
        first, last = split_name(row.get("Name", ""))
        for token in (first, last):
            ch = get_alpha_char(token)
            if ch:
                letters.add(ch)
    ordered = [ch for ch in ALPHABET_ORDER if ch in letters]
    extra = sorted([ch for ch in letters if ch not in ALPHABET_ORDER])
    return ordered + extra

def people_by_letter(letter: str, limit: int = 40):
    def starts_with(letter: str, token: str) -> bool:
        return bool(token) and token.lower().startswith(letter.lower())
    mask_rows: List[int] = []
    for idx, row in PEOPLE_DF.iterrows():
        first, last = split_name(row.get("Name", ""))
        if starts_with(letter, first) or starts_with(letter, last):
            mask_rows.append(idx)
    return PEOPLE_DF.iloc[mask_rows].sort_values("Name").head(limit)

def search_by_name(query: str, limit: int = 20):
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
    if bio: blocks.append(f"ℹ️ {bio}")
    if tip: blocks.append(f"💬 {tip}")
    inst_block = []
    if inst: inst_block.append(inst)
    if inst_link: inst_block.append(inst_link)
    if inst_block: blocks.append("\n".join(inst_block))
    if contact: blocks.append(f"📱inst: {contact}")
    return "\n\n".join(blocks)

async def send_person_card(message, row):
    caption = person_card(row)
    photo = _clean(row.get("Photo")) or _clean(row.get("photo"))
    if photo:
        try:
            await message.reply_photo(photo, caption=caption)
            return
        except Exception as e:
            log.warning("Failed to send photo for %s: %s", _clean(row.get("Name")), e)
    await message.reply_text(caption, disable_web_page_preview=True)

# =============================================
#        B) MEET SLOTS (meet_slots.csv)
# =============================================
MEET_CSV_PATH = os.getenv("MEET_SLOTS_CSV", "meet_slots.csv")
MEET_TZ = ZoneInfo("Europe/Vienna")
MEET_YEAR = 2025

def _meet_norm_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s

def _meet_parse_when_to_meet(s: str):
    if not isinstance(s, str) or not s.strip():
        return (None, None, None, None)
    t = s.strip().replace("—", "–")
    m = re.match(r'^[A-Za-z]{3}\s+(\d{2})\.(\d{2})\s*,\s*(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})', t)
    if not m:
        m = re.match(r'^(\d{2})\.(\d{2})\s*,\s*(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})', t)
    if not m:
        return (None, None, None, None)
    dd, mm, h1, mi1, h2, mi2 = map(int, m.groups())
    start_dt = datetime(MEET_YEAR, mm, dd, h1, mi1, tzinfo=MEET_TZ)
    end_dt = datetime(MEET_YEAR, mm, dd, h2, mi2, tzinfo=MEET_TZ)
    return (start_dt, end_dt, f"{start_dt:%d.%m}", f"{start_dt:%H:%M}–{end_dt:%H:%M}")

def load_meet_df():
    try:
        df = pd.read_csv(MEET_CSV_PATH)
    except Exception as e:
        log.warning("Failed to load meet slots CSV: %s", e)
        return pd.DataFrame(columns=[
            "Name", "When to Meet", "Where to Meet", "Event name", "Topic", "Event type",
            "start_dt", "end_dt", "date", "timespan", "location_key", "topic_key", "event_key"
        ])

    col_name = next((c for c in df.columns if c.lower() == "name"), None)
    col_when = next((c for c in df.columns if c.lower().startswith("when")), None)
    col_loc  = next((c for c in df.columns if c.lower().startswith("where")), None)
    col_event = next((c for c in df.columns if "event name" in c.lower() or c.lower().strip() == "event"), None)
    col_topic = next((c for c in df.columns if c.lower().strip() == "topic" or "theme" in c.lower()), None)

    out = pd.DataFrame()
    out["name"]       = df[col_name]  if col_name  else ""
    out["when_raw"]   = df[col_when]  if col_when  else ""
    out["location"]   = df[col_loc]   if col_loc   else ""
    out["event_name"] = df[col_event] if col_event else ""
    out["topic"]      = df[col_topic] if col_topic else ""

    starts, ends, dates, spans = [], [], [], []
    for v in out["when_raw"]:
        st, en, d, sp = _meet_parse_when_to_meet(v)
        starts.append(st); ends.append(en); dates.append(d); spans.append(sp)
    out["start_dt"] = starts
    out["end_dt"]   = ends
    out["date"]     = dates
    out["timespan"] = spans
    out["location_key"] = out["location"].map(_meet_norm_key)
    out["topic_key"]    = out["topic"].map(_meet_norm_key)
    out["event_key"]    = out["event_name"].map(_meet_norm_key)
    return out

MEET_DF = load_meet_df()

def format_people_times(rows):
    if rows is None or len(rows) == 0:
        return "Ничего не найдено."
    by_name: dict[str, List[str]] = {}
    for _, r in rows.iterrows():
        nm = _clean(r.get("name"))
        start = r.get("start_dt")
        end = r.get("end_dt")
        loc = _clean(r.get("location"))
        if (
            not nm
            or not isinstance(start, datetime)
            or not isinstance(end, datetime)
            or pd.isna(start)
            or pd.isna(end)
        ):
            continue
        entry = f"🕒 {start:%d.%m %H:%M}–{end:%H:%M}\n📍 {loc}"
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
    await update.message.reply_text(
        "Главное меню:",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔎 Поиск по имени", callback_data="name:menu")],
            [InlineKeyboardButton("📍 По локации (meet slots)", callback_data="ms:loc_menu")],
            [InlineKeyboardButton("🕒 По времени (meet slots)", callback_data="ms:time_menu")],
            [InlineKeyboardButton("🏷️ По теме (meet slots)", callback_data="ms:topic_menu")],
            [InlineKeyboardButton("🎫 По ивенту (meet slots)", callback_data="ms:event_menu")],
        ])
    )

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
        return

async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if data == "back:home":
        await q.edit_message_text(
            "Главное меню:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔎 Поиск по имени", callback_data="name:menu")],
                [InlineKeyboardButton("📍 По локации (meet slots)", callback_data="ms:loc_menu")],
                [InlineKeyboardButton("🕒 По времени (meet slots)", callback_data="ms:time_menu")],
                [InlineKeyboardButton("🏷️ По теме (meet slots)", callback_data="ms:topic_menu")],
                [InlineKeyboardButton("🎫 По ивенту (meet slots)", callback_data="ms:event_menu")],
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
        dates = ["03.09", "04.09", "05.09", "06.09", "07.09"]
        rows = [[InlineKeyboardButton(d, callback_data=f"ms:time_date#{d}") for d in dates]]
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        await q.edit_message_text("Выбери дату:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:time_date#"):
        date = data.split("#", 1)[1]
        hours = [f"{h:02d}:00" for h in list(range(10,24)) + [0]]
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
        parts = data.split("#")
        date = parts[1]
        hour = int(parts[2])
        dd, mm = map(int, date.split("."))
        qdt = datetime(MEET_YEAR, mm, dd, hour, 0, tzinfo=MEET_TZ)
        subset = MEET_DF[(MEET_DF["start_dt"].notna()) & (MEET_DF["end_dt"].notna()) &
                         (MEET_DF["start_dt"] <= qdt) & (qdt < MEET_DF["end_dt"])]
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
            idx = int(data.split("#",1)[1])
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
        await q.edit_message_text("Выбери тему:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:topic#"):
        try:
            idx = int(data.split("#",1)[1])
        except Exception:
            idx = -1
        topics = context.user_data.get("_topics", [])
        if 0 <= idx < len(topics):
            topic = topics[idx]
            rows_df = MEET_DF[MEET_DF["topic"] == topic]
            people = sorted({str(n).strip() for n in rows_df["name"].dropna() if str(n).strip()})
            context.user_data["_people_from_topic"] = people
            text = f"Тема: {topic}\n\n" + format_people_times(rows_df)
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
            idx = int(data.split("#",1)[1])
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
        await q.edit_message_text("Выбери ивент:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("ms:event#"):
        try:
            idx = int(data.split("#",1)[1])
        except Exception:
            idx = -1
        events = context.user_data.get("_events", [])
        if 0 <= idx < len(events):
            ev = events[idx]
            rows_df = MEET_DF[MEET_DF["event_name"] == ev]
            people = sorted({str(n).strip() for n in rows_df["name"].dropna() if str(n).strip()})
            context.user_data["_people_from_event"] = people
            text = f"Ивент: {ev}\n\n" + format_people_times(rows_df)
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
            idx = int(data.split("#",1)[1])
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

# ====== handlers registration ======
application.add_handler(CommandHandler("start", start))
application.add_handler(CallbackQueryHandler(on_cb))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

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
