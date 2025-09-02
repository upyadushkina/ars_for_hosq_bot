import os, re, logging
from datetime import datetime
from typing import List
import pandas as pd
from difflib import SequenceMatcher

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

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
        return pd.DataFrame(columns=["Name", "Institution", "Festival Role", "Role", "Where to Meet", "Attendance", "Bio", "Conversation Tip", "Institution Link"])  

PEOPLE_DF = load_people_df()

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
    for v in PEOPLE_DF.get("Name", pd.Series(dtype=str)).fillna(""):
        first, last = split_name(v)
        for token in [first, last]:
            if token:
                letters.add(token[0].upper())
    letters = [ch for ch in ALPHABET_ORDER if ch in letters]
    return letters

def people_by_letter(letter: str, limit=40):
    letter = (letter or "").upper()
    mask_rows = []
    for idx, v in PEOPLE_DF.get("Name", pd.Series(dtype=str)).fillna("").items():
        first, last = split_name(v)
        if first.upper().startswith(letter) or last.upper().startswith(letter):
            mask_rows.append(idx)
    subset = PEOPLE_DF.iloc[mask_rows].head(limit)
    return subset

def person_card(row_or_name):
    if isinstance(row_or_name, str):
        name = row_or_name
        cand = PEOPLE_DF[PEOPLE_DF["Name"].fillna("").str.strip().str.lower() == str(name).strip().lower()]
        if cand.empty:
            # самое близкое совпадение
            cand = PEOPLE_DF.iloc[[0]] if not PEOPLE_DF.empty else pd.DataFrame()
        row = cand.iloc[0] if not cand.empty else {}
    else:
        row = row_or_name
    name = str(row.get('Name','')).strip()
    role = str(row.get('Role', row.get('Festival Role',''))).strip()
    inst = str(row.get('Institution','')).strip()
    bio = str(row.get('Bio','')).strip()
    tip = str(row.get('Conversation Tip','')).strip()
    link = str(row.get('Institution Link','')).strip()
    lines = [f"👤 {name}"]
    if role: lines.append(f"🧭 Role: {role}")
    if inst: lines.append(f"🏛️ Institution: {inst}")
    if bio:  lines.append(f"📝 Bio: {bio}")
    if tip:  lines.append(f"💬 Tip: {tip}")
    if link: lines.append(f"🔗 {link}")
    return "\n".join(lines)

def _ratio(a,b):
    try:
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    except Exception:
        return 0.0

def search_by_name(query, limit=20):
    query = (query or "").strip()
    if not query:
        return PEOPLE_DF.head(limit)
    scores = []
    for idx, name in PEOPLE_DF.get("Name", pd.Series(dtype=str)).fillna("").items():
        scores.append((idx, _ratio(name, query)))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_idx = [i for i,_ in scores[: max(limit, 1)]]
    return PEOPLE_DF.loc[top_idx]

# =============================================
#        B) MEET SLOTS (meet_slots.csv)
# =============================================
import pandas as _pd
from zoneinfo import ZoneInfo as _ZoneInfo

MEET_CSV_PATH = os.getenv("MEET_SLOTS_CSV", "meet_slots.csv")
MEET_TZ = _ZoneInfo("Europe/Vienna")
MEET_YEAR = 2025

def _meet_norm_key(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
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

# Для списка людей в выбранной локации: только имя и время
def format_people_times(rows):
    if rows is None or len(rows)==0:
        return "Ничего не найдено."
    by_name = {}
    for _, r in rows.iterrows():
        nm = (r.get("name") or "").strip()
        t = (r.get("date") or "")
        span = (r.get("timespan") or "")
        if not nm:
            continue
        by_name.setdefault(nm, []).append(f"{t} {span}".strip())
    lines = []
    for nm in sorted(by_name.keys(), key=lambda s: s.lower()):
        times = "; ".join(sorted(set(by_name[nm])))
        lines.append(f"👤 {nm} — 🕒 {times}")
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
        if res.empty:
            await update.message.reply_text("Ничего не нашлось. Попробуй по-другому.")
        else:
            for _, row in res.iterrows():
                await update.message.reply_text(person_card(row), disable_web_page_preview=True)
        context.user_data["expect_name_typing"] = False
        await update.message.reply_text("Вернуться в меню?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back:home")]]))
        return

    if context.user_data.get("ms_expect_time"):
        txt = (update.message.text or "").strip()
        qdt = parse_user_time_str(txt)
        if not qdt:
            await update.message.reply_text("Не понял время. Примеры: 20:15, 06.09 21:00, Сейчас")
            return
        subset = MEET_DF[(MEET_DF["start_dt"].notna()) & (MEET_DF["end_dt"].notna()) & (MEET_DF["start_dt"] <= qdt) & (qdt < MEET_DF["end_dt"]) ]
        await update.message.reply_text(format_people_times(subset), disable_web_page_preview=True)
        context.user_data["ms_expect_time"] = False
        await update.message.reply_text("Вернуться в меню?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back:home")]]))
        return

async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if data == "back:home":
        await q.edit_message_text("Главное меню:", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔎 Поиск по имени", callback_data="name:menu")],
            [InlineKeyboardButton("📍 По локации (meet slots)", callback_data="ms:loc_menu")],
            [InlineKeyboardButton("🕒 По времени (meet slots)", callback_data="ms:time_menu")],
            [InlineKeyboardButton("🏷️ По теме (meet slots)", callback_data="ms:topic_menu")],
            [InlineKeyboardButton("🎫 По ивенту (meet slots)", callback_data="ms:event_menu")],
        ]))
        return

    # имя
    if data == "name:menu":
        await q.edit_message_text("Как искать по имени?", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔤 Имена в алфавитном порядке", callback_data="name:alpha")],
            [InlineKeyboardButton("⌨️ Введите имя", callback_data="name:typing")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back:home")],
        ]))
        return

    if data == "name:alpha":
        # небольшая клавиатура букв
        letters = unique_letters()
        rows, row = [], []
        for i, ch in enumerate(letters, 1):
            row.append(InlineKeyboardButton(ch, callback_data=f"name:letter:{ch}"))
            if len(row) == 8:
                rows.append(row); row = []
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        await q.edit_message_text("Выбери букву:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("name:letter:"):
        letter = data.split(":",2)[2]
        res = people_by_letter(letter)
        await q.edit_message_text(f"Имена на букву {letter}:")
        for _, row in res.iterrows():
            await q.message.reply_text(person_card(row), disable_web_page_preview=True)
        return

    if data == "name:typing":
        context.user_data["expect_name_typing"] = True
        await q.edit_message_text("Введи имя/фамилию для поиска:")
        return

    # meet: время
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
        subset = MEET_DF[(MEET_DF["start_dt"].notna()) & (MEET_DF["end_dt"].notna()) & (MEET_DF["start_dt"] <= now_dt) & (now_dt < MEET_DF["end_dt"]) ]
        await q.edit_message_text("Доступны сейчас:\n\n" + format_people_times(subset))
        context.user_data["ms_expect_time"] = False
        return

    # meet: локации (новая логика)
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
            idx = int(data.split("#",1)[1])
        except Exception:
            idx = -1
        locs = context.user_data.get("_locs", [])
        if 0 <= idx < len(locs):
            loc = locs[idx]
            subset = MEET_DF[MEET_DF["location"] == loc]
            # список людей на кнопках
            people = sorted({str(n).strip() for n in subset["name"].dropna() if str(n).strip()})
            context.user_data["_people_from_loc"] = people
            text = f"Локация: {loc}\n\n" + format_people_times(subset)
            kb_rows, r = [], []
            for i, nm in enumerate(people):
                r.append(InlineKeyboardButton(nm[:30] or "—", callback_data=f"ms:person#{i}"))
                if len(r) == 2:
                    kb_rows.append(r); r = []
            if r: kb_rows.append(r)
            kb_rows.append([InlineKeyboardButton("⬅️ Назад к локациям", callback_data="ms:loc_menu")])
            await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb_rows), disable_web_page_preview=True)
        else:
            await q.edit_message_text("Ошибка выбора локации")
        return

    if data.startswith("ms:person#"):
        try:
            idx = int(data.split("#",1)[1])
        except Exception:
            idx = -1
        people = context.user_data.get("_people_from_loc", [])
        if 0 <= idx < len(people):
            nm = people[idx]
            card = person_card(nm)
            await q.edit_message_text(card, disable_web_page_preview=True, reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⬅️ Назад к людям", callback_data="ms:loc_menu")],
                [InlineKeyboardButton("⬅️ В меню", callback_data="back:home")],
            ]))
        else:
            await q.edit_message_text("Не удалось найти человека.")
        return

    # темы/ивенты оставлены рабочими (пригодится позже)
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
            rows = MEET_DF[MEET_DF["topic"] == topic]
            await q.edit_message_text(f"Тема: {topic}\n\n{format_people_times(rows)}", disable_web_page_preview=True)
        else:
            await q.edit_message_text("Ошибка выбора темы")
        return

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
            rows = MEET_DF[MEET_DF["event_name"] == ev]
            await q.edit_message_text(f"Ивент: {ev}\n\n{format_people_times(rows)}", disable_web_page_preview=True)
        else:
            await q.edit_message_text("Ошибка выбора ивента")
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
        return JSONResponse({"status":"ok"})
    except Exception as e:
        log.exception("webhook error: %s", e)
        return JSONResponse({"status":"error"}, status_code=200)
