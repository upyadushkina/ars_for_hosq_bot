/**
 * Ars Electronica 2026 — hosq Meet Bot
 * Cloudflare Worker (Free plan) — same idea as hosq Task bot.
 *
 * Secrets / vars (Dashboard → Variables and Secrets):
 *   TELEGRAM_TOKEN          (or TELEGRAM_BOT_TOKEN)
 *   WEBHOOK_SECRET          (optional; or TELEGRAM_WEBHOOK_SECRET)
 *   SHEET_ID                (optional override)
 *
 * Data: Google Sheet tabs `people` / `meet` / `schedule`
 * Cache: in-memory, ~10 minutes (force refresh in Настройки)
 */

const DEFAULT_SHEET_ID = "1X6oouceCLD3pY289WtT-gPeU_kxeq6mpilbLNehSGCw";
const CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes (within 5–15)
const FESTIVAL_DATES = ["08.09", "09.09", "10.09", "11.09", "12.09", "13.09"];
const ALPHABET =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";

/** @type {{ data: any, loadedAt: number, source: string } | null} */
let CACHE = null;

function token(env) {
  return env.TELEGRAM_TOKEN || env.TELEGRAM_BOT_TOKEN || "";
}

function webhookSecret(env) {
  return env.WEBHOOK_SECRET || env.TELEGRAM_WEBHOOK_SECRET || "";
}

function sheetId(env) {
  return clean(env.SHEET_ID) || DEFAULT_SHEET_ID;
}

function clean(v) {
  return (v == null ? "" : String(v)).trim();
}

function normHeader(h) {
  return clean(h).toLowerCase().replace(/\s+/g, " ");
}

function pickCol(headers, predicates) {
  for (let i = 0; i < headers.length; i++) {
    const h = normHeader(headers[i]);
    for (const pred of predicates) {
      if (typeof pred === "string") {
        if (h === pred || h.startsWith(pred)) return i;
      } else if (pred(h)) return i;
    }
  }
  return -1;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cur += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(cur);
      cur = "";
    } else if (ch === "\n") {
      row.push(cur);
      rows.push(row);
      row = [];
      cur = "";
    } else if (ch === "\r") {
      // skip
    } else {
      cur += ch;
    }
  }
  if (cur.length || row.length) {
    row.push(cur);
    rows.push(row);
  }
  return rows.filter((r) => r.some((c) => clean(c)));
}

async function fetchSheetCsv(id, sheetName) {
  const url = `https://docs.google.com/spreadsheets/d/${id}/gviz/tq?tqx=out:csv&sheet=${encodeURIComponent(sheetName)}`;
  const res = await fetch(url, {
    headers: { "user-agent": "ars-hosq-bot/1.0" },
    cf: { cacheTtl: 0, cacheEverything: false },
  });
  if (!res.ok) throw new Error(`Sheet "${sheetName}" HTTP ${res.status}`);
  return res.text();
}

function mapPeople(rows) {
  if (!rows.length) return [];
  const h = rows[0];
  const iName = pickCol(h, ["name"]);
  const iPhoto = pickCol(h, ["photo"]);
  const iContact = pickCol(h, ["contact"]);
  const iRole = pickCol(h, ["role"]);
  const iInst = pickCol(h, ["institution", "country"]);
  const iFr = pickCol(h, ["festival role", "festival"]);
  const iBio = pickCol(h, ["bio"]);
  const iTip = pickCol(h, ["conversation tip", "tip"]);
  const iLink = pickCol(h, ["institution link", "link"]);
  if (iName < 0) throw new Error("people: no Name column");
  const out = [];
  for (let r = 1; r < rows.length; r++) {
    const row = rows[r];
    const n = clean(row[iName]);
    if (!n) continue;
    out.push({
      n,
      ph: iPhoto >= 0 ? clean(row[iPhoto]) : "",
      c: iContact >= 0 ? clean(row[iContact]) : "",
      r: iRole >= 0 ? clean(row[iRole]) : "",
      i: iInst >= 0 ? clean(row[iInst]) : "",
      fr: iFr >= 0 ? clean(row[iFr]) : "",
      b: iBio >= 0 ? clean(row[iBio]).slice(0, 500) : "",
      t: iTip >= 0 ? clean(row[iTip]) : "",
      l: iLink >= 0 ? clean(row[iLink]) : "",
    });
  }
  return out;
}

function mapMeet(rows) {
  if (!rows.length) return [];
  const h = rows[0];
  // Sheet may have broken header like "Name !Mediengruppe Bitnik"
  const iName = pickCol(h, [(x) => x === "name" || x.startsWith("name ")]);
  const iWhere = pickCol(h, ["where to meet", "where"]);
  const iEvent = pickCol(h, ["event name", "event"]);
  const iTopic = pickCol(h, ["topic"]);
  const iDate = pickCol(h, ["date"]);
  const iStart = pickCol(h, ["time start", "start"]);
  const iFinish = pickCol(h, ["time finish", "time end", "finish", "end"]);
  if (iName < 0 || iDate < 0) throw new Error("meet: missing Name/Date columns");
  const out = [];
  for (let r = 1; r < rows.length; r++) {
    const row = rows[r];
    const d = iDate >= 0 ? clean(row[iDate]) : "";
    if (!d) continue;
    out.push({
      n: clean(row[iName]),
      w: iWhere >= 0 ? clean(row[iWhere]) : "",
      e: iEvent >= 0 ? clean(row[iEvent]) : "",
      tp: iTopic >= 0 ? clean(row[iTopic]) : "",
      d,
      s: iStart >= 0 ? clean(row[iStart]) : "",
      f: iFinish >= 0 ? clean(row[iFinish]) : "",
    });
  }
  return out;
}

function mapSched(rows) {
  if (!rows.length) return [];
  const h = rows[0];
  const iE = pickCol(h, ["event_name", "event name"]);
  const iP = pickCol(h, ["people"]);
  const iD = pickCol(h, ["date"]);
  const iS = pickCol(h, ["time start"]);
  const iF = pickCol(h, ["time finish"]);
  const iDs = pickCol(h, ["event_description", "description"]);
  const iTy = pickCol(h, ["event_type", "type"]);
  const iU = pickCol(h, ["link_to_event", "link"]);
  const iW = pickCol(h, ["where"]);
  const iRg = pickCol(h, ["registration_status", "registration"]);
  if (iE < 0 || iD < 0) throw new Error("schedule: missing event_name/date");
  const out = [];
  for (let r = 1; r < rows.length; r++) {
    const row = rows[r];
    const e = clean(row[iE]);
    const d = clean(row[iD]);
    if (!e || !d) continue;
    out.push({
      e,
      p: iP >= 0 ? clean(row[iP]) : "",
      d,
      s: iS >= 0 ? clean(row[iS]) : "",
      f: iF >= 0 ? clean(row[iF]) : "",
      ds: iDs >= 0 ? clean(row[iDs]).slice(0, 400) : "",
      ty: iTy >= 0 ? clean(row[iTy]) : "",
      u: iU >= 0 ? clean(row[iU]) : "",
      w: iW >= 0 ? clean(row[iW]) : "",
      rg: iRg >= 0 ? clean(row[iRg]) : "",
    });
  }
  return out;
}

function cacheAgeSec() {
  if (!CACHE?.loadedAt) return null;
  return Math.max(0, Math.floor((Date.now() - CACHE.loadedAt) / 1000));
}

function formatCacheAge() {
  const s = cacheAgeSec();
  if (s == null) return "кэш пуст";
  if (s < 60) return `${s} сек назад`;
  return `${Math.floor(s / 60)} мин назад`;
}

async function loadData(env, { force = false } = {}) {
  const now = Date.now();
  if (!force && CACHE?.data && now - CACHE.loadedAt < CACHE_TTL_MS) {
    return CACHE.data;
  }
  const id = sheetId(env);
  const [peopleTxt, meetTxt, schedTxt] = await Promise.all([
    fetchSheetCsv(id, "people"),
    fetchSheetCsv(id, "meet"),
    fetchSheetCsv(id, "schedule"),
  ]);
  const data = {
    people: mapPeople(parseCsv(peopleTxt)),
    meet: mapMeet(parseCsv(meetTxt)),
    sched: mapSched(parseCsv(schedTxt)),
  };
  if (!data.people.length) throw new Error("people sheet is empty");
  CACHE = { data, loadedAt: Date.now(), source: id };
  return data;
}

function homeKeyboard() {
  return {
    inline_keyboard: [
      [{ text: "🔎 Поиск по имени", callback_data: "name:menu" }],
      [{ text: "📍 По локации", callback_data: "ms:loc_menu" }],
      [{ text: "🕒 По времени", callback_data: "ms:time_menu" }],
      [{ text: "🏷️ По теме", callback_data: "ms:topic_menu" }],
      [{ text: "🎫 По ивенту", callback_data: "ms:event_menu" }],
      [{ text: "📅 Моё расписание", callback_data: "schedule:menu" }],
      [{ text: "⚙️ Настройки", callback_data: "settings:menu" }],
      [{ text: "🔄 Перезапустить", callback_data: "restart:bot" }],
    ],
  };
}

function settingsKeyboard() {
  return {
    inline_keyboard: [
      [{ text: "🔄 Обновить данные", callback_data: "settings:refresh" }],
      [{ text: "⬅️ В меню", callback_data: "back:home" }],
    ],
  };
}

function btnRows(items, cols = 2) {
  const rows = [];
  let row = [];
  for (const it of items) {
    row.push(it);
    if (row.length === cols) {
      rows.push(row);
      row = [];
    }
  }
  if (row.length) rows.push(row);
  return rows;
}

function parseDate(d) {
  const m = String(d || "").trim().match(/^(\d{1,2})\.(\d{1,2})$/);
  if (!m) return null;
  return { day: +m[1], month: +m[2], key: `${String(+m[1]).padStart(2, "0")}.${String(+m[2]).padStart(2, "0")}` };
}

function parseHm(t) {
  const p = String(t || "").split(":");
  if (p.length < 2) return null;
  return { h: +p[0], m: +p[1] };
}

function mins(h, m) {
  return h * 60 + m;
}

function splitName(name) {
  const parts = clean(name).split(/[\s\-]+/).filter(Boolean);
  return { first: parts[0] || "", last: parts.length > 1 ? parts[parts.length - 1] : "" };
}

function personCard(p) {
  const blocks = [];
  if (p.n) blocks.push(p.n);
  if (p.r) blocks.push(p.r);
  if (p.fr) blocks.push(`🎫 ${p.fr}`);
  if (p.b) blocks.push(`✏️bio: ${p.b}`);
  if (p.t) blocks.push(`💡tip: ${p.t}`);
  const inst = [p.i, p.l].filter(Boolean);
  if (inst.length) blocks.push(inst.join("\n"));
  if (p.c) blocks.push(`📱 ${p.c}`);
  return blocks.join("\n\n");
}

function peopleByLetter(people, letter) {
  const L = letter.toLowerCase();
  return people
    .map((p, idx) => ({ p, idx }))
    .filter(({ p }) => {
      const { first, last } = splitName(p.n);
      return first.toLowerCase().startsWith(L) || last.toLowerCase().startsWith(L);
    })
    .sort((a, b) => a.p.n.localeCompare(b.p.n))
    .slice(0, 40);
}

function searchByName(people, q) {
  const qq = q.toLowerCase();
  return people
    .map((p, idx) => ({ p, idx }))
    .filter(({ p }) => p.n.toLowerCase().includes(qq))
    .sort((a, b) => a.p.n.localeCompare(b.p.n))
    .slice(0, 20);
}

function uniqueLetters(people) {
  const set = new Set();
  for (const p of people) {
    const { first, last } = splitName(p.n);
    for (const tok of [first, last]) {
      const m = tok.match(/[A-Za-zА-Яа-яЁё]/);
      if (m) set.add(m[0].toUpperCase());
    }
  }
  return [...ALPHABET].filter((ch) => set.has(ch)).concat(
    [...set].filter((ch) => !ALPHABET.includes(ch)).sort()
  );
}

function uniqueSorted(arr) {
  return [...new Set(arr.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function formatPeopleTimes(rows) {
  if (!rows.length) return "Ничего не найдено.";
  const by = {};
  for (const r of rows) {
    const name = r.n;
    if (!name) continue;
    const line = `${r.d} ${r.s}–${r.f}\n📍 ${r.w || "—"}`;
    (by[name] ||= []).push(line);
  }
  return Object.keys(by)
    .sort((a, b) => a.localeCompare(b))
    .map((nm) => `👤 ${nm}\n${by[nm].join("\n")}`)
    .join("\n\n");
}

function formatScheduleEvents(rows) {
  if (!rows.length) return "Ничего не найдено.";
  return rows
    .map((r) => {
      let line = `🎫 ${r.e}\n${r.s}–${r.f}`;
      if (r.w) line += `\n📍 ${r.w}`;
      return line;
    })
    .join("\n\n");
}

function formatScheduleCard(r) {
  const blocks = [];
  if (r.e) blocks.push(`🎫 ${r.e}`);
  if (r.s) blocks.push(`🕒 ${r.s}–${r.f}`);
  if (r.w) blocks.push(`📍 ${r.w}`);
  if (r.ty) blocks.push(`🏷️ ${r.ty}`);
  if (r.ds) blocks.push(`📝 ${r.ds}`);
  if (r.p) blocks.push(`👥 ${r.p}`);
  if (r.rg) blocks.push(`📋 ${r.rg}`);
  if (r.u) blocks.push(`🔗 ${r.u}`);
  return blocks.join("\n\n");
}

function meetAtHour(meet, dateKey, hour) {
  return meet.filter((r) => {
    const pd = parseDate(r.d);
    if (!pd || pd.key !== dateKey) return false;
    const a = parseHm(r.s);
    const b = parseHm(r.f);
    if (!a || !b) return false;
    let start = mins(a.h, a.m);
    let end = mins(b.h, b.m);
    if (end <= start) end += 24 * 60; // overnight
    const q = mins(hour, 0);
    return start <= q && q < end;
  });
}

function schedOnDate(sched, dateKey) {
  return sched
    .filter((r) => {
      const pd = parseDate(r.d);
      return pd && pd.key === dateKey;
    })
    .sort((a, b) => clean(a.s).localeCompare(clean(b.s)));
}

function schedAtHour(sched, dateKey, hour) {
  return schedOnDate(sched, dateKey).filter((r) => {
    const a = parseHm(r.s);
    const b = parseHm(r.f);
    if (!a || !b) return false;
    let start = mins(a.h, a.m);
    let end = mins(b.h, b.m);
    if (end <= start) end += 24 * 60;
    const q = mins(hour, 0);
    return start <= q && q < end;
  });
}

function findPersonByName(people, name) {
  const q = clean(name).toLowerCase();
  return people.findIndex((p) => clean(p.n).toLowerCase() === q);
}

async function tg(env, method, body) {
  const res = await fetch(`https://api.telegram.org/bot${token(env)}/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function sendMessage(env, chatId, text, reply_markup) {
  return tg(env, "sendMessage", {
    chat_id: chatId,
    text,
    reply_markup,
    disable_web_page_preview: true,
  });
}

async function editMessage(env, chatId, messageId, text, reply_markup) {
  return tg(env, "editMessageText", {
    chat_id: chatId,
    message_id: messageId,
    text,
    reply_markup,
    disable_web_page_preview: true,
  });
}

async function answerCb(env, id) {
  try {
    await tg(env, "answerCallbackQuery", { callback_query_id: id });
  } catch (_) {}
}

async function sendPerson(env, chatId, p) {
  const caption = personCard(p);
  if (p.ph) {
    const photo = await tg(env, "sendPhoto", {
      chat_id: chatId,
      photo: p.ph,
      caption: caption.slice(0, 1024),
    });
    if (photo && photo.ok) return;
  }
  await sendMessage(env, chatId, caption);
}

function peopleKeyboard(items, backCb) {
  const buttons = items.map(({ p, idx }) => ({
    text: (p.n || "—").slice(0, 30),
    callback_data: `np:${idx}`,
  }));
  const rows = btnRows(buttons, 2);
  if (backCb) rows.push([{ text: "⬅️ Назад", callback_data: backCb }]);
  rows.push([{ text: "⬅️ В меню", callback_data: "back:home" }]);
  return { inline_keyboard: rows };
}

async function handleStart(env, chatId) {
  await sendMessage(env, chatId, "Главное меню:", homeKeyboard());
}

async function handleText(env, data, chatId, text) {
  const q = clean(text);
  if (!q) return;
  const hits = searchByName(data.people, q);
  if (!hits.length) {
    await sendMessage(env, chatId, "Ничего не нашлось. Попробуй по-другому.", {
      inline_keyboard: [[{ text: "⬅️ Назад", callback_data: "back:home" }]],
    });
    return;
  }
  await sendMessage(env, chatId, "Выбери человека:", peopleKeyboard(hits, "name:typing"));
}

async function handleCallback(env, data, cq) {
  const chatId = cq.message.chat.id;
  const messageId = cq.message.message_id;
  const raw = cq.data || "";
  await answerCb(env, cq.id);

  const edit = (text, markup) => editMessage(env, chatId, messageId, text, markup);
  const { people, meet, sched } = data;

  if (raw === "back:home" || raw === "home:menu" || raw === "restart:bot") {
    await edit(raw === "restart:bot" ? "🔄 Бот перезапущен!\n\nГлавное меню:" : "Главное меню:", homeKeyboard());
    return;
  }

  // ---- settings ----
  if (raw === "settings:menu") {
    const nPeople = data.people?.length ?? 0;
    const nMeet = data.meet?.length ?? 0;
    const nSched = data.sched?.length ?? 0;
    await edit(
      `⚙️ Настройки\n\nИсточник: Google Sheet\nКэш: ${formatCacheAge()}\nTTL: ${Math.round(CACHE_TTL_MS / 60000)} мин\n\npeople: ${nPeople}\nmeet: ${nMeet}\nschedule: ${nSched}`,
      settingsKeyboard()
    );
    return;
  }

  if (raw === "settings:refresh") {
    await edit("⏳ Обновляю данные из Google Sheet…");
    try {
      const fresh = await loadData(env, { force: true });
      await edit(
        `✅ Данные обновлены\n\nКэш: только что\npeople: ${fresh.people.length}\nmeet: ${fresh.meet.length}\nschedule: ${fresh.sched.length}`,
        settingsKeyboard()
      );
    } catch (err) {
      console.error(err);
      await edit(
        `❌ Не удалось обновить:\n${clean(err.message || err)}\n\nПроверь, что Sheet доступен по ссылке «Anyone with the link».`,
        settingsKeyboard()
      );
    }
    return;
  }

  // ---- name ----
  if (raw === "name:menu") {
    await edit("Как искать по имени?", {
      inline_keyboard: [
        [{ text: "🔤 Имена в алфавитном порядке", callback_data: "name:alpha" }],
        [{ text: "⌨️ Введите имя", callback_data: "name:typing" }],
        [{ text: "⬅️ Назад", callback_data: "back:home" }],
      ],
    });
    return;
  }

  if (raw === "name:typing") {
    await edit("Введи имя/фамилию для поиска (обычным сообщением):");
    return;
  }

  if (raw === "name:alpha") {
    const letters = uniqueLetters(people);
    const buttons = letters.map((ch) => ({ text: ch, callback_data: `name:letter:${ch}` }));
    const rows = btnRows(buttons, 8);
    rows.push([{ text: "⬅️ Назад", callback_data: "back:home" }]);
    await edit("Выбери букву:", { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("name:letter:")) {
    const letter = raw.slice("name:letter:".length);
    const hits = peopleByLetter(people, letter);
    if (!hits.length) {
      await edit("Ничего не найдено.", {
        inline_keyboard: [[{ text: "⬅️ Назад", callback_data: "name:alpha" }]],
      });
      return;
    }
    await edit(`Имена на букву ${letter}:`, peopleKeyboard(hits, "name:alpha"));
    return;
  }

  if (raw.startsWith("np:")) {
    const idx = +raw.slice(3);
    const p = people[idx];
    if (!p) {
      await edit("Не удалось найти человека.");
      return;
    }
    await sendPerson(env, chatId, p);
    await sendMessage(env, chatId, "Что дальше?", {
      inline_keyboard: [
        [{ text: "⬅️ В меню", callback_data: "back:home" }],
      ],
    });
    return;
  }

  // ---- meet: location ----
  if (raw === "ms:loc_menu") {
    const locs = uniqueSorted(meet.map((r) => r.w));
    const buttons = locs.map((loc, i) => ({
      text: loc.slice(0, 30) || "—",
      callback_data: `ms:l:${i}`,
    }));
    // Telegram max ~100 buttons; truncate
    const rows = btnRows(buttons.slice(0, 60), 1);
    rows.push([{ text: "⬅️ Назад", callback_data: "back:home" }]);
    await edit("Где ты сейчас?", { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:l:")) {
    const locs = uniqueSorted(meet.map((r) => r.w));
    const li = +raw.slice(5);
    const loc = locs[li];
    if (!loc) {
      await edit("Ошибка выбора локации");
      return;
    }
    const subset = meet.filter((r) => r.w === loc);
    const names = uniqueSorted(subset.map((r) => r.n));
    const buttons = names.slice(0, 40).map((nm) => {
      const idx = findPersonByName(people, nm);
      return {
        text: nm.slice(0, 30) || "—",
        callback_data: idx >= 0 ? `np:${idx}` : "back:home",
      };
    });
    const rows = btnRows(buttons, 2);
    rows.push([{ text: "⬅️ Назад к локациям", callback_data: "ms:loc_menu" }]);
    await edit(`Локация: ${loc}\n\n${formatPeopleTimes(subset)}`, { inline_keyboard: rows });
    return;
  }

  // ---- meet: time ----
  if (raw === "ms:time_menu") {
    const rows = [FESTIVAL_DATES.map((d) => ({ text: d, callback_data: `ms:td:${d}` }))];
    rows.push([{ text: "⬅️ Назад", callback_data: "back:home" }]);
    await edit("Выбери дату:", { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:td:")) {
    const date = raw.slice(6);
    const hours = [...Array.from({ length: 14 }, (_, i) => i + 10), 0];
    const buttons = hours.map((h) => ({
      text: `${String(h).padStart(2, "0")}:00`,
      callback_data: `ms:th:${date}:${String(h).padStart(2, "0")}`,
    }));
    const rows = btnRows(buttons, 4);
    rows.push([{ text: "⬅️ Назад к датам", callback_data: "ms:time_menu" }]);
    await edit("Который час?", { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:th:")) {
    const [, , date, hh] = raw.split(":");
    const hour = +hh;
    const subset = meetAtHour(meet, date, hour);
    const names = uniqueSorted(subset.map((r) => r.n));
    const buttons = names.slice(0, 40).map((nm) => {
      const idx = findPersonByName(people, nm);
      return { text: nm.slice(0, 30) || "—", callback_data: idx >= 0 ? `np:${idx}` : "back:home" };
    });
    const rows = btnRows(buttons, 2);
    rows.push([{ text: "⬅️ Назад к часам", callback_data: `ms:td:${date}` }]);
    await edit(`${date} ${String(hour).padStart(2, "0")}:00\n\n${formatPeopleTimes(subset)}`, {
      inline_keyboard: rows,
    });
    return;
  }

  // ---- meet: topic ----
  if (raw === "ms:topic_menu") {
    const topics = uniqueSorted(meet.map((r) => r.tp));
    const buttons = topics.slice(0, 60).map((t, i) => ({
      text: t.slice(0, 30) || "—",
      callback_data: `ms:t:${i}`,
    }));
    const rows = btnRows(buttons, 1);
    rows.push([{ text: "⬅️ Назад", callback_data: "back:home" }]);
    await edit("Выбери тему:", { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:t:")) {
    const topics = uniqueSorted(meet.map((r) => r.tp));
    const ti = +raw.slice(5);
    const topic = topics[ti];
    if (!topic) {
      await edit("Ошибка выбора темы");
      return;
    }
    const subset = meet.filter((r) => r.tp === topic);
    const names = uniqueSorted(subset.map((r) => r.n));
    const buttons = names.slice(0, 40).map((nm) => {
      const idx = findPersonByName(people, nm);
      return { text: nm.slice(0, 30) || "—", callback_data: idx >= 0 ? `np:${idx}` : "back:home" };
    });
    const rows = btnRows(buttons, 2);
    rows.push([{ text: "⬅️ Назад к темам", callback_data: "ms:topic_menu" }]);
    await edit(`Тема: ${topic}\n\n${formatPeopleTimes(subset)}`, { inline_keyboard: rows });
    return;
  }

  // ---- meet: event ----
  if (raw === "ms:event_menu") {
    const events = uniqueSorted(meet.map((r) => r.e));
    const buttons = events.slice(0, 60).map((e, i) => ({
      text: e.slice(0, 30) || "—",
      callback_data: `ms:e:${i}`,
    }));
    const rows = btnRows(buttons, 1);
    rows.push([{ text: "⬅️ Назад", callback_data: "back:home" }]);
    await edit("Выбери ивент:", { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:e:")) {
    const events = uniqueSorted(meet.map((r) => r.e));
    const ei = +raw.slice(5);
    const ev = events[ei];
    if (!ev) {
      await edit("Ошибка выбора ивента");
      return;
    }
    const subset = meet.filter((r) => r.e === ev);
    const names = uniqueSorted(subset.map((r) => r.n));
    const buttons = names.slice(0, 40).map((nm) => {
      const idx = findPersonByName(people, nm);
      return { text: nm.slice(0, 30) || "—", callback_data: idx >= 0 ? `np:${idx}` : "back:home" };
    });
    const rows = btnRows(buttons, 2);
    rows.push([{ text: "⬅️ Назад к ивентам", callback_data: "ms:event_menu" }]);
    await edit(`Ивент: ${ev}\n\n${formatPeopleTimes(subset)}`, { inline_keyboard: rows });
    return;
  }

  // ---- schedule ----
  if (raw === "schedule:menu") {
    const dates = FESTIVAL_DATES.filter((d) => schedOnDate(sched, d).length);
    const use = dates.length ? dates : FESTIVAL_DATES;
    const buttons = use.map((d) => ({ text: d, callback_data: `schedule:date:${d}` }));
    const rows = btnRows(buttons, 2);
    rows.push([{ text: "⬅️ Назад", callback_data: "back:home" }]);
    await edit("Выбери дату:", { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("schedule:date:")) {
    const date = raw.slice("schedule:date:".length);
    const hours = [...Array.from({ length: 14 }, (_, i) => i + 10), 0];
    const buttons = [
      { text: "📅 Весь день", callback_data: `schedule:full:${date}` },
      ...hours.map((h) => ({
        text: `${String(h).padStart(2, "0")}:00`,
        callback_data: `schedule:h:${date}:${String(h).padStart(2, "0")}`,
      })),
    ];
    const rows = btnRows(buttons, 4);
    rows.push([{ text: "⬅️ Назад к датам", callback_data: "schedule:menu" }]);
    await edit(`Выбери время для ${date}:`, { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("schedule:full:")) {
    const date = raw.slice("schedule:full:".length);
    const events = schedOnDate(sched, date);
    if (!events.length) {
      await edit(`На ${date} нет событий.`, {
        inline_keyboard: [[{ text: "⬅️ Назад", callback_data: `schedule:date:${date}` }]],
      });
      return;
    }
    const buttons = events.slice(0, 40).map((ev, i) => ({
      text: (ev.e || "—").slice(0, 30),
      callback_data: `schedule:ev:${date}:f:${i}`,
    }));
    const rows = btnRows(buttons, 1);
    rows.push([{ text: "⬅️ Назад к времени", callback_data: `schedule:date:${date}` }]);
    await edit(`Расписание на ${date}\n\n${formatScheduleEvents(events)}`, { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("schedule:h:")) {
    const parts = raw.split(":");
    const date = parts[2];
    const hour = +parts[3];
    const events = schedAtHour(sched, date, hour);
    if (!events.length) {
      await edit(`На ${date} в ${String(hour).padStart(2, "0")}:00 нет событий.`, {
        inline_keyboard: [[{ text: "⬅️ Назад", callback_data: `schedule:date:${date}` }]],
      });
      return;
    }
    const buttons = events.slice(0, 40).map((ev, i) => ({
      text: (ev.e || "—").slice(0, 30),
      callback_data: `schedule:ev:${date}:${String(hour).padStart(2, "0")}:${i}`,
    }));
    const rows = btnRows(buttons, 1);
    rows.push([{ text: "⬅️ Назад к времени", callback_data: `schedule:date:${date}` }]);
    await edit(
      `${date} ${String(hour).padStart(2, "0")}:00\n\n${formatScheduleEvents(events)}`,
      { inline_keyboard: rows }
    );
    return;
  }

  if (raw.startsWith("schedule:ev:")) {
    // schedule:ev:DD.MM:f:i  OR schedule:ev:DD.MM:HH:i
    const rest = raw.slice("schedule:ev:".length);
    const m = rest.match(/^(\d{2}\.\d{2}):(f|\d{2}):(\d+)$/);
    if (!m) {
      await edit("Не удалось найти событие.");
      return;
    }
    const date = m[1];
    const mode = m[2];
    const i = +m[3];
    const events = mode === "f" ? schedOnDate(sched, date) : schedAtHour(sched, date, +mode);
    const ev = events[i];
    if (!ev) {
      await edit("Не удалось найти событие.");
      return;
    }
    await sendMessage(env, chatId, formatScheduleCard(ev));
    const names = clean(ev.p)
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
    const buttons = [];
    for (const nm of names.slice(0, 20)) {
      const idx = findPersonByName(people, nm);
      if (idx >= 0) buttons.push({ text: nm.slice(0, 30), callback_data: `np:${idx}` });
    }
    const rows = btnRows(buttons, 2);
    rows.push([{ text: "⬅️ В меню", callback_data: "back:home" }]);
    await sendMessage(
      env,
      chatId,
      buttons.length ? "Люди на этом событии:" : "Что дальше?",
      { inline_keyboard: rows }
    );
    return;
  }

  await edit("Неизвестная команда. Нажми /start");
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/") {
      return new Response("ok");
    }

    if (request.method === "POST" && (url.pathname === "/webhook" || url.pathname === "/")) {
      const secret = webhookSecret(env);
      if (secret) {
        const hdr = request.headers.get("X-Telegram-Bot-Api-Secret-Token") || "";
        if (hdr !== secret) return new Response("forbidden", { status: 403 });
      }
      if (!token(env)) return new Response("no token", { status: 500 });

      let update;
      try {
        update = await request.json();
      } catch {
        return new Response("bad json", { status: 400 });
      }

      try {
        const data = await loadData(env);
        if (update.callback_query) {
          await handleCallback(env, data, update.callback_query);
        } else if (update.message) {
          const msg = update.message;
          const chatId = msg.chat.id;
          const text = msg.text || "";
          if (text.startsWith("/start")) await handleStart(env, chatId);
          else if (text.trim()) await handleText(env, data, chatId, text);
        }
      } catch (err) {
        console.error(err);
        try {
          const chatId =
            update?.message?.chat?.id || update?.callback_query?.message?.chat?.id;
          if (chatId) {
            await sendMessage(
              env,
              chatId,
              "Произошла ошибка. Попробуйте /start ещё раз."
            );
          }
        } catch (_) {}
      }
      return new Response("ok");
    }

    return new Response("not found", { status: 404 });
  },
};
