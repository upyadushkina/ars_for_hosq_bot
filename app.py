import os, re, logging
from datetime import datetime
import zoneinfo
import pandas as pd
from difflib import SequenceMatcher

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
    # стабильный ID строки для callback-кнопок
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"index":"RowID"}, inplace=True)
    return df

DF = load_df()

# ---------- helpers ----------
ALPHABET_ORDER = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def get_alpha_char(s: str) -> str | None:
    m = re.search("[A-Za-zА-Яа-яЁё]", s or "")
    if not m:
        return None
    return m.group(0).upper()

def split_name(name: str):
    parts = re.split("[\s\-]+", (name or "").strip())
    parts = [p for p in parts if p]
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) > 1 else ""
    return first, last

def unique_letters():
    letters = set()
    for _, row in DF.iterrows():
        first, last = split_name(row.get("Name", ""))
        for token in (first, last):
            ch = get_alpha_char(token)
            if ch:
                letters.add(ch)
    ordered = [ch for ch in ALPHABET_ORDER if ch in letters]
    extra = sorted([ch for ch in letters if ch not in ALPHABET_ORDER])
    return ordered + extra

def people_by_letter(letter: str, limit=40):
    def starts_with(letter: str, token: str) -> bool:
        if not token:
            return False
        return token.lower().startswith(letter.lower())
    mask_rows = []
    for idx, row in DF.iterrows():
        first, last = split_name(row.get("Name", ""))
        if starts_with(letter, first) or starts_with(letter, last):
            mask_rows.append(idx)
    subset = DF.iloc[mask_rows].head(limit)
    return subset

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

def suggest_by_name(query: str, n: int = 5):
    q = (query or "").strip().lower()
    if not q:
        return DF.head(0)
    scores = []
    for idx, row in DF.iterrows():
        name = (row.get("Name", "") or "").strip()
        if not name:
            continue
        first, last = split_name(name)
        candidates = [name.lower()]
        if first:
            candidates.append(first.lower())
        if last:
            candidates.append(last.lower())
        best = max(SequenceMatcher(None, q, c).ratio() for c in candidates)
        scores.append((best, idx))
    scores.sort(key=lambda x: x[0], reverse=True)
    picked = [idx for score, idx in scores if score >= 0.45][:n]
    if len(picked) < n:
        for score, idx in scores:
            if idx not in picked:
                picked.append(idx)
            if len(picked) == n:
                break
    return DF.loc[picked] if picked else DF.head(0)

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

# ---- UI helpers ----

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔎 Поиск по имени", callback_data="name:menu")],
        [InlineKeyboardButton("📍 Поиск по локации", callback_data="loc:menu")],
        [InlineKeyboardButton("🕒 Поиск по времени", callback_data="time:menu")],
    ])

def name_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔤 Имена в алфавитном порядке", callback_data="name:alpha")],
        [InlineKeyboardButton("⌨️ Введите имя", callback_data="name:typing")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back:home")],
    ])

def letters_keyboard():
    letters = unique_letters()
    rows, row = [], []
    for i, ch in enumerate(letters, 1):
        row.append(InlineKeyboardButton(ch, callback_data=f"name:letter:{ch}"))
        if len(row) == 8:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="name:menu")])
    return InlineKeyboardMarkup(rows)

async def start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Выбери, как искать:",
        reply_markup=main_menu_keyboard(),
        disable_web_page_preview=True
    )

async def on_text(update, context: ContextTypes.DEFAULT_TYPE):
    q = (update.message.text or "").strip()
    if not q:
        return
    res = search_by_name(q, limit=15)
    if res.empty:
        sug = suggest_by_name(q, n=5)
        if sug.empty:
            await update.message.reply_text("Ничего не нашла. Попробуй другое имя.")
            return
        buttons = [[InlineKeyboardButton(sug.iloc[i]['Name'][:40], callback_data=f"name:id:{int(sug.iloc[i]['RowID'])}")] for i in range(len(sug))]
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="name:menu")])
        await update.message.reply_text(
            "Возможно, вы имели в виду:",
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return
    for _, row in res.iterrows():
        await update.message.reply_text(person_card(row), disable_web_page_preview=True)

async def on_cb(update, context: ContextTypes.DEFAULT_TYPE):
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
        letter = data.split(":", 2)[2]
        res = people_by_letter(letter)
        if res.empty:
            await q.edit_message_text(f"Имен на «{letter}» не нашла.", reply_markup=letters_keyboard())
            return
        await q.edit_message_text(f"Имена на «{letter}»:")
        for _, row in res.iterrows():
            await q.message.reply_text(person_card(row), disable_web_page_preview=True)
        await q.message.reply_text("Готово. Вернуться в меню?", reply_markup=name_menu_keyboard())
        return

    if data == "name:typing":
        await q.edit_message_text("Введите имя (или часть имени) в чате. Я предложу варианты, если не найду точное совпадение.", reply_markup=name_menu_keyboard())
        return

    if data.startswith("name:id:"):
        rid = int(data.split(":")[-1])
        row = DF[DF["RowID"]==rid].iloc[0]
        await q.message.reply_text(person_card(row), disable_web_page_preview=True)
        return

    # --- location ---
    if data == "loc:menu":
        locs = sorted({x.strip() for x in DF["Where to Meet"].tolist() if x and x.strip()})
        rows, row = [], []
        for i,loc in enumerate(locs,1):
            row.append(InlineKeyboardButton(loc[:30], callback_data=f"loc:{loc[:60]}"))
            if len(row)==2:
                rows.append(row); row=[]
        if row: rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back:home")])
        await q.edit_message_text("Выбери локацию:", reply_markup=InlineKeyboardMarkup(rows or [[InlineKeyboardButton("Нет локаций", callback_data="back:home")]]))
        return

    if data.startswith("loc:") and not data.startswith("loc:menu"):
        loc = data[4:]
        res = filter_by_location(loc)
        await q.edit_message_text(f"В «{loc}»:")
        for _, row in res.iterrows():
            await q.message.reply_text(person_card(row), disable_web_page_preview=True)
        await q.message.reply_text("Вернуться в меню?", reply_markup=main_menu_keyboard())
        return

    # --- time ---
    if data == "time:menu":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Сейчас", callback_data="time:now")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back:home")],
        ])
        await q.edit_message_text("Какое время интересно?", reply_markup=kb)
        return

    if data == "time:now":
        subset = DF[DF["Attendance"].apply(is_available_now)]
        await q.edit_message_text("Доступны сейчас:" if not subset.empty else "Сейчас никого не вижу по расписанию.")
        for _, row in subset.iterrows():
            await q.message.reply_text(person_card(row), disable_web_page_preview=True)
        await q.message.reply_text("Вернуться в меню?", reply_markup=main_menu_keyboard())
        return

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
        log.exception("webhook error: %s", e)
        return JSONResponse({"status":"error"}, status_code=200)
