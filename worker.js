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
 * Texts: messages.js (edit manually)
 * Cache: in-memory, ~10 minutes (force refresh in Настройки)
 */

import { t } from "./messages.js";

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

function escHtml(v) {
  return String(v == null ? "" : v)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/** Bold date for list UIs (Telegram HTML). */
function fmtDate(d) {
  return `<b>${escHtml(d || t("dash"))}</b>`;
}

/** Event title as hyperlink when schedule has link_to_event. */
function fmtEvent(name, url) {
  const label = escHtml(name || t("dash"));
  const href = clean(url);
  if (href && /^https?:\/\//i.test(href)) {
    const safeHref = href.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
    return `<a href="${safeHref}">${label}</a>`;
  }
  return label;
}

function eventLinkMap(sched) {
  const map = new Map();
  for (const r of sched) {
    const e = clean(r.e);
    const u = clean(r.u);
    if (e && u && !map.has(e)) map.set(e, u);
  }
  return map;
}

function enrichMeetLinks(meet, linkMap) {
  for (const r of meet) {
    r.u = linkMap.get(clean(r.e)) || "";
  }
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
  if (s == null) return t("cache_empty");
  if (s < 60) return t("cache_sec_ago", { n: s });
  return t("cache_min_ago", { n: Math.floor(s / 60) });
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
  const meet = mapMeet(parseCsv(meetTxt));
  const sched = mapSched(parseCsv(schedTxt));
  enrichMeetLinks(meet, eventLinkMap(sched));
  const data = {
    people: mapPeople(parseCsv(peopleTxt)),
    meet,
    sched,
  };
  if (!data.people.length) throw new Error("people sheet is empty");
  CACHE = { data, loadedAt: Date.now(), source: id };
  return data;
}

function homeKeyboard() {
  return {
    inline_keyboard: [
      [{ text: t("btn_search_name"), callback_data: "name:menu" }],
      [{ text: t("btn_by_location"), callback_data: "ms:loc_menu" }],
      [{ text: t("btn_by_time"), callback_data: "ms:time_menu" }],
      [{ text: t("btn_by_topic"), callback_data: "ms:topic_menu" }],
      [{ text: t("btn_by_event"), callback_data: "ms:event_menu" }],
      [{ text: t("btn_schedule"), callback_data: "schedule:menu" }],
      [{ text: t("btn_settings"), callback_data: "settings:menu" }],
      [{ text: t("btn_restart"), callback_data: "restart:bot" }],
    ],
  };
}

function settingsKeyboard() {
  return {
    inline_keyboard: [
      [{ text: t("btn_refresh_data"), callback_data: "settings:refresh" }],
      [{ text: t("btn_back_menu"), callback_data: "back:home" }],
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

/** "Project – Foo; Concert – Bar" → ["Foo", "Bar"] */
function parseFestivalRoleEvents(fr) {
  const out = [];
  for (const chunk of clean(fr).split(";")) {
    const part = clean(chunk);
    if (!part) continue;
    const bits = part.split(/\s+[–—-]\s+/);
    const title = clean(bits.length > 1 ? bits.slice(1).join(" – ") : part);
    if (title) out.push(title);
  }
  return out;
}

function resolveEventLink(name, linkMap) {
  const n = clean(name);
  if (!n || !linkMap?.size) return "";
  if (linkMap.has(n)) return linkMap.get(n);
  const lower = n.toLowerCase();
  for (const [k, v] of linkMap) {
    if (k.toLowerCase() === lower) return v;
  }
  let best = "";
  let bestLen = 0;
  for (const [k, v] of linkMap) {
    const kk = k.toLowerCase();
    if (kk.includes(lower) || lower.includes(kk)) {
      if (k.length > bestLen) {
        best = v;
        bestLen = k.length;
      }
    }
  }
  return best;
}

/** Unique event titles for a person: meet slots first, then Festival Role. */
function personEventTitles(p, meet) {
  const titles = [];
  const seen = new Set();
  const add = (name) => {
    const n = clean(name);
    if (!n) return;
    const key = n.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    titles.push(n);
  };
  const pname = clean(p.n);
  for (const r of meet) {
    if (clean(r.n) === pname) add(r.e);
  }
  for (const title of parseFestivalRoleEvents(p.fr)) add(title);
  return titles;
}

function formatPersonEvents(p, meet, linkMap) {
  const titles = personEventTitles(p, meet);
  if (!titles.length) return "";
  return titles
    .map((name) => t("person_event_item", { event: fmtEvent(name, resolveEventLink(name, linkMap)) }))
    .join("\n");
}

function personCard(p, meet = [], sched = []) {
  const linkMap = eventLinkMap(sched);
  const blocks = [];
  if (p.n) blocks.push(escHtml(p.n));
  if (p.r) blocks.push(escHtml(p.r));
  const events = formatPersonEvents(p, meet, linkMap);
  if (events) blocks.push(events);
  if (p.b) blocks.push(t("person_bio", { value: escHtml(p.b) }));
  if (p.t) blocks.push(t("person_tip", { value: escHtml(p.t) }));
  const inst = [p.i, p.l].filter(Boolean).map(escHtml);
  if (inst.length) blocks.push(inst.join("\n"));
  if (p.c) blocks.push(t("person_contact", { value: escHtml(p.c) }));
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

const TG_TEXT_MAX = 3800;
const LOC_PAGE_SIZE = 28;

function clipTg(text, max = 4096) {
  const s = String(text || "");
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + "…";
}

/** Rows that actually have a person at a place (skip empty shells). */
function meetActive(meet) {
  return meet.filter((r) => clean(r.w) && clean(r.n));
}

/** "Ars Electronica Center, Level 3, Sky Loft" → parent + room */
function splitWhere(w) {
  const s = clean(w);
  const i = s.indexOf(",");
  if (i < 0) return { parent: s, sub: "" };
  return { parent: s.slice(0, i).trim(), sub: s.slice(i + 1).trim() };
}

/** parent → Set of full "Where to Meet" strings (only active rows) */
function locationByParent(meet) {
  const byParent = new Map();
  for (const r of meetActive(meet)) {
    const w = clean(r.w);
    const { parent } = splitWhere(w);
    if (!byParent.has(parent)) byParent.set(parent, new Set());
    byParent.get(parent).add(w);
  }
  return byParent;
}

function locationParents(meet) {
  return [...locationByParent(meet).keys()].sort((a, b) => a.localeCompare(b));
}

/** Unique room labels under a parent. null = no submenu needed. */
function locationSubs(byParent, parent) {
  const wheres = [...(byParent.get(parent) || [])];
  if (wheres.length <= 1) return null;
  return uniqueSorted(wheres.map((w) => splitWhere(w).sub));
}

function meetAtParent(meet, parent) {
  return meetActive(meet).filter((r) => {
    const w = clean(r.w);
    return w === parent || w.startsWith(parent + ",");
  });
}

function meetAtSub(meet, parent, sub) {
  const active = meetActive(meet);
  if (!sub) return active.filter((r) => clean(r.w) === parent);
  const full = `${parent}, ${sub}`;
  return active.filter((r) => clean(r.w) === full);
}

function locListText(subset, namesOnPage) {
  const pageRows = subset.filter((r) => namesOnPage.includes(clean(r.n)));
  return formatPeopleTimes(pageRows, { hideWhere: true });
}

function showLocPeople(edit, people, subset, locLabel, backCb, pagePrefix, page = 0) {
  const names = uniqueSorted(subset.map((r) => r.n).filter(Boolean));
  if (!names.length) {
    return edit(t("loc_header", { loc: escHtml(locLabel), list: t("nothing_found") }), {
      inline_keyboard: [[{ text: t("btn_back_locations"), callback_data: backCb }]],
    });
  }
  const pages = Math.max(1, Math.ceil(names.length / LOC_PAGE_SIZE));
  const p = Math.min(Math.max(0, page | 0), pages - 1);
  const slice = names.slice(p * LOC_PAGE_SIZE, (p + 1) * LOC_PAGE_SIZE);
  const list = locListText(subset, slice);
  const header =
    pages > 1
      ? t("loc_header_paged", {
          loc: escHtml(locLabel),
          page: p + 1,
          pages,
          list,
        })
      : t("loc_header", { loc: escHtml(locLabel), list });

  const buttons = slice.map((nm) => {
    const idx = findPersonByName(people, nm);
    return {
      text: nm.slice(0, 30) || t("dash"),
      callback_data: idx >= 0 ? `np:${idx}` : "back:home",
    };
  });
  const rows = btnRows(buttons, 2);
  if (pages > 1) {
    const nav = [];
    if (p > 0) {
      nav.push({ text: t("btn_loc_prev"), callback_data: `${pagePrefix}:${p - 1}` });
    }
    nav.push({ text: t("btn_loc_page", { page: p + 1, pages }), callback_data: `${pagePrefix}:${p}` });
    if (p < pages - 1) {
      nav.push({ text: t("btn_loc_next"), callback_data: `${pagePrefix}:${p + 1}` });
    }
    rows.push(nav);
  }
  rows.push([{ text: t("btn_back_locations"), callback_data: backCb }]);
  return edit(header, { inline_keyboard: rows });
}

/** @type {Map<string, { mode: string, at: number }>} */
const PENDING = new Map();
const PENDING_TTL_MS = 15 * 60 * 1000;

function setPending(chatId, mode) {
  PENDING.set(String(chatId), { mode, at: Date.now() });
}

function takePending(chatId) {
  const key = String(chatId);
  const v = PENDING.get(key);
  PENDING.delete(key);
  if (!v) return null;
  if (Date.now() - v.at > PENDING_TTL_MS) return null;
  return v.mode;
}

function allEventNames(meet, sched) {
  return uniqueSorted(
    [...meet.map((r) => r.e), ...sched.map((r) => r.e)].filter(Boolean)
  );
}

function searchEvents(events, q) {
  const qq = q.toLowerCase();
  return events
    .map((name, idx) => ({ name, idx }))
    .filter(({ name }) => name.toLowerCase().includes(qq))
    .slice(0, 30);
}

function eventsAtHour(meet, sched, dateKey, hour) {
  const fromMeet = meetAtHour(meet, dateKey, hour).map((r) => r.e);
  const fromSched = schedAtHour(sched, dateKey, hour).map((r) => r.e);
  return uniqueSorted([...fromMeet, ...fromSched].filter(Boolean));
}

function eventKeyboard(items, backCb) {
  // items: { name, idx } where idx is index in allEventNames
  const buttons = items.map(({ name, idx }) => ({
    text: (name || t("dash")).slice(0, 40),
    callback_data: `ms:ev:g:${idx}`,
  }));
  const rows = btnRows(buttons, 1);
  if (backCb) rows.push([{ text: t("btn_back"), callback_data: backCb }]);
  rows.push([{ text: t("btn_back_menu"), callback_data: "back:home" }]);
  return { inline_keyboard: rows };
}

function peopleOnEvent(meet, sched, eventName) {
  const fromMeet = meet.filter((r) => r.e === eventName);
  if (fromMeet.length) return fromMeet;
  const out = [];
  for (const r of sched.filter((x) => x.e === eventName)) {
    const names = clean(r.p)
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
    if (!names.length) {
      out.push({
        n: "",
        w: r.w,
        e: r.e,
        tp: r.e,
        d: r.d,
        s: r.s,
        f: r.f,
        u: r.u || "",
      });
      continue;
    }
    for (const nm of names) {
      out.push({
        n: nm,
        w: r.w,
        e: r.e,
        tp: r.e,
        d: r.d,
        s: r.s,
        f: r.f,
        u: r.u || "",
      });
    }
  }
  return out.filter((r) => r.n || r.e);
}

function formatPeopleTimes(rows, { hideWhere = false } = {}) {
  if (!rows.length) return t("nothing_found");
  const by = {};
  for (const r of rows) {
    const name = r.n;
    if (!name) continue;
    const slotKey = hideWhere ? "people_times_slot_noloc" : "people_times_slot";
    const line = t(slotKey, {
      date: fmtDate(r.d),
      start: escHtml(r.s || t("dash")),
      finish: escHtml(r.f || t("dash")),
      where: escHtml(r.w || t("dash")),
      event: fmtEvent(r.e || r.tp, r.u),
    });
    (by[name] ||= []).push(line);
  }
  const names = Object.keys(by).sort((a, b) => a.localeCompare(b));
  if (!names.length) return t("nothing_found");

  // Between people: blank line (ZWSP keeps Telegram from collapsing \n\n).
  // Within a person, multiple slots also separated by a blank line.
  const PEOPLE_SEP = "\n\n\u200B";
  const SLOT_SEP = "\n\n\u200B";
  const blocks = names.map((nm) =>
    t("people_times_person", {
      name: escHtml(nm),
      slots: by[nm].join(SLOT_SEP),
    })
  );

  let text = blocks.join(PEOPLE_SEP);
  if (text.length <= TG_TEXT_MAX) return text;

  // Keep the same multiline layout; drop trailing people if over Telegram limit
  const kept = [];
  for (const block of blocks) {
    const next = kept.length ? `${kept.join(PEOPLE_SEP)}${PEOPLE_SEP}${block}` : block;
    if (next.length > TG_TEXT_MAX - 5) break;
    kept.push(block);
  }
  if (!kept.length) return clipTg(blocks[0], TG_TEXT_MAX);
  text = kept.join(PEOPLE_SEP);
  return kept.length < blocks.length ? `${text}${PEOPLE_SEP}…` : text;
}

function formatScheduleEvents(rows) {
  if (!rows.length) return t("nothing_found");
  return rows
    .map((r) => {
      let line = t("schedule_list_item", {
        name: fmtEvent(r.e, r.u),
        date: fmtDate(r.d),
        start: escHtml(r.s || t("dash")),
        finish: escHtml(r.f || t("dash")),
      });
      if (r.w) line += t("schedule_list_where", { where: escHtml(r.w) });
      return line;
    })
    .join("\n\n");
}

function formatScheduleCard(r) {
  const blocks = [];
  if (r.e) blocks.push(t("schedule_card_name", { value: fmtEvent(r.e, r.u) }));
  if (r.d) blocks.push(t("schedule_card_date", { value: fmtDate(r.d) }));
  if (r.s) {
    blocks.push(
      t("schedule_card_time", {
        start: escHtml(r.s),
        finish: escHtml(r.f || t("dash")),
      })
    );
  }
  if (r.w) blocks.push(t("schedule_card_where", { value: escHtml(r.w) }));
  if (r.ty) blocks.push(t("schedule_card_type", { value: escHtml(r.ty) }));
  if (r.ds) blocks.push(t("schedule_card_desc", { value: escHtml(r.ds) }));
  if (r.p) blocks.push(t("schedule_card_people", { value: escHtml(r.p) }));
  if (r.rg) blocks.push(t("schedule_card_reg", { value: escHtml(r.rg) }));
  // link is already on the event title when present
  if (r.u && !/^https?:\/\//i.test(clean(r.u))) {
    blocks.push(t("schedule_card_link", { value: escHtml(r.u) }));
  }
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

/** Hours shown in time menus: 10:00–23:00 + 00:00 */
const CANDIDATE_HOURS = [...Array.from({ length: 14 }, (_, i) => i + 10), 0];

function hoursWithMeet(meet, dateKey) {
  return CANDIDATE_HOURS.filter((h) => meetAtHour(meet, dateKey, h).length > 0);
}

function hoursWithEventSearch(meet, sched, dateKey) {
  return CANDIDATE_HOURS.filter((h) => eventsAtHour(meet, sched, dateKey, h).length > 0);
}

function hoursWithSched(sched, dateKey) {
  return CANDIDATE_HOURS.filter((h) => schedAtHour(sched, dateKey, h).length > 0);
}

function hourButtons(hours, date, cbPrefix) {
  return hours.map((h) => ({
    text: `${String(h).padStart(2, "0")}:00`,
    callback_data: `${cbPrefix}:${date}:${String(h).padStart(2, "0")}`,
  }));
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
    text: clipTg(text),
    reply_markup,
    parse_mode: "HTML",
    disable_web_page_preview: true,
  });
}

async function editMessage(env, chatId, messageId, text, reply_markup) {
  const payload = {
    chat_id: chatId,
    message_id: messageId,
    text: clipTg(text),
    reply_markup,
    parse_mode: "HTML",
    disable_web_page_preview: true,
  };
  const res = await tg(env, "editMessageText", payload);
  // If edit fails (e.g. oversized markup edge cases), fall back to a new message
  if (res && res.ok === false) {
    return sendMessage(env, chatId, text, reply_markup);
  }
  return res;
}

async function answerCb(env, id) {
  try {
    await tg(env, "answerCallbackQuery", { callback_query_id: id });
  } catch (_) {}
}

async function sendPerson(env, chatId, p, meet = [], sched = []) {
  const caption = personCard(p, meet, sched);
  if (p.ph) {
    if (caption.length <= 1024) {
      const photo = await tg(env, "sendPhoto", {
        chat_id: chatId,
        photo: p.ph,
        caption,
        parse_mode: "HTML",
      });
      if (photo && photo.ok) return;
    } else {
      const photo = await tg(env, "sendPhoto", {
        chat_id: chatId,
        photo: p.ph,
        caption: escHtml(p.n || "").slice(0, 1024),
        parse_mode: "HTML",
      });
      if (photo && photo.ok) {
        await sendMessage(env, chatId, caption);
        return;
      }
    }
  }
  await sendMessage(env, chatId, caption);
}

function peopleKeyboard(items, backCb) {
  const buttons = items.map(({ p, idx }) => ({
    text: (p.n || t("dash")).slice(0, 30),
    callback_data: `np:${idx}`,
  }));
  const rows = btnRows(buttons, 2);
  if (backCb) rows.push([{ text: t("btn_back"), callback_data: backCb }]);
  rows.push([{ text: t("btn_back_menu"), callback_data: "back:home" }]);
  return { inline_keyboard: rows };
}

async function handleStart(env, chatId) {
  await sendMessage(env, chatId, t("main_menu"), homeKeyboard());
}

async function handleText(env, data, chatId, text) {
  const q = clean(text);
  if (!q) return;
  const mode = takePending(chatId) || "name";

  if (mode === "event") {
    const events = allEventNames(data.meet, data.sched);
    const hits = searchEvents(events, q);
    if (!hits.length) {
      await sendMessage(env, chatId, t("event_not_found_search"), {
        inline_keyboard: [
          [{ text: t("btn_back"), callback_data: "ms:event_menu" }],
          [{ text: t("btn_back_menu"), callback_data: "back:home" }],
        ],
      });
      return;
    }
    await sendMessage(env, chatId, t("event_pick"), eventKeyboard(hits, "ms:event_menu"));
    return;
  }

  const hits = searchByName(data.people, q);
  if (!hits.length) {
    await sendMessage(env, chatId, t("name_not_found_search"), {
      inline_keyboard: [[{ text: t("btn_back"), callback_data: "back:home" }]],
    });
    return;
  }
  await sendMessage(env, chatId, t("name_pick_person"), peopleKeyboard(hits, "name:typing"));
}

async function handleCallback(env, data, cq) {
  const chatId = cq.message.chat.id;
  const messageId = cq.message.message_id;
  const raw = cq.data || "";
  await answerCb(env, cq.id);

  const edit = (text, markup) => editMessage(env, chatId, messageId, text, markup);
  const { people, meet, sched } = data;

  if (raw === "back:home" || raw === "home:menu" || raw === "restart:bot") {
    await edit(raw === "restart:bot" ? t("restarted") : t("main_menu"), homeKeyboard());
    return;
  }

  // ---- settings ----
  if (raw === "settings:menu") {
    const nPeople = data.people?.length ?? 0;
    const nMeet = data.meet?.length ?? 0;
    const nSched = data.sched?.length ?? 0;
    await edit(
      t("settings_title", {
        cacheAge: formatCacheAge(),
        ttlMin: Math.round(CACHE_TTL_MS / 60000),
        nPeople,
        nMeet,
        nSched,
      }),
      settingsKeyboard()
    );
    return;
  }

  if (raw === "settings:refresh") {
    await edit(t("settings_refreshing"));
    try {
      const fresh = await loadData(env, { force: true });
      await edit(
        t("settings_refreshed", {
          nPeople: fresh.people.length,
          nMeet: fresh.meet.length,
          nSched: fresh.sched.length,
        }),
        settingsKeyboard()
      );
    } catch (err) {
      console.error(err);
      await edit(
        t("settings_refresh_failed", { error: escHtml(clean(err.message || err)) }),
        settingsKeyboard()
      );
    }
    return;
  }

  // ---- name ----
  if (raw === "name:menu") {
    await edit(t("name_menu_title"), {
      inline_keyboard: [
        [{ text: t("btn_name_alpha"), callback_data: "name:alpha" }],
        [{ text: t("btn_name_typing"), callback_data: "name:typing" }],
        [{ text: t("btn_back"), callback_data: "back:home" }],
      ],
    });
    return;
  }

  if (raw === "name:typing") {
    setPending(chatId, "name");
    await edit(t("name_typing_prompt"));
    return;
  }

  if (raw === "name:alpha") {
    const letters = uniqueLetters(people);
    const buttons = letters.map((ch) => ({ text: ch, callback_data: `name:letter:${ch}` }));
    const rows = btnRows(buttons, 8);
    rows.push([{ text: t("btn_back"), callback_data: "back:home" }]);
    await edit(t("name_pick_letter"), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("name:letter:")) {
    const letter = raw.slice("name:letter:".length);
    const hits = peopleByLetter(people, letter);
    if (!hits.length) {
      await edit(t("name_not_found_letter"), {
        inline_keyboard: [[{ text: t("btn_back"), callback_data: "name:alpha" }]],
      });
      return;
    }
    await edit(t("name_letter_title", { letter }), peopleKeyboard(hits, "name:alpha"));
    return;
  }

  if (raw.startsWith("np:")) {
    const idx = +raw.slice(3);
    const p = people[idx];
    if (!p) {
      await edit(t("person_not_found"));
      return;
    }
    await sendPerson(env, chatId, p, meet, sched);
    await sendMessage(env, chatId, t("what_next"), {
      inline_keyboard: [[{ text: t("btn_back_menu"), callback_data: "back:home" }]],
    });
    return;
  }

  // ---- meet: location (venue → room) ----
  if (raw === "ms:loc_menu") {
    const parents = locationParents(meet);
    const buttons = parents.map((loc, i) => ({
      text: loc.slice(0, 40) || t("dash"),
      callback_data: `ms:lp:${i}`,
    }));
    const rows = btnRows(buttons.slice(0, 60), 1);
    rows.push([{ text: t("btn_back"), callback_data: "back:home" }]);
    await edit(t("loc_menu_title"), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:lp:")) {
    const parents = locationParents(meet);
    const pi = +raw.slice(6);
    const parent = parents[pi];
    if (!parent) {
      await edit(t("loc_error"));
      return;
    }
    const byParent = locationByParent(meet);
    const subs = locationSubs(byParent, parent);
    if (!subs) {
      const wheres = [...(byParent.get(parent) || [])];
      const only = wheres[0] || parent;
      const subset = meet.filter((r) => clean(r.w) === only && clean(r.n));
      await showLocPeople(edit, people, subset, only, "ms:loc_menu", `ms:l1:${pi}`, 0);
      return;
    }
    const buttons = [
      { text: t("btn_loc_all"), callback_data: `ms:la:${pi}:0` },
      ...subs.map((sub, si) => ({
        text: (sub || t("btn_loc_unspecified")).slice(0, 40),
        callback_data: `ms:ls:${pi}:${si}:0`,
      })),
    ];
    const rows = btnRows(buttons, 1);
    rows.push([{ text: t("btn_back_locations"), callback_data: "ms:loc_menu" }]);
    await edit(t("loc_pick_room", { loc: escHtml(parent) }), { inline_keyboard: rows });
    return;
  }

  // single-venue people list with pages: ms:l1:{pi}:{page}
  if (raw.startsWith("ms:l1:")) {
    const parts = raw.split(":");
    const pi = +parts[2];
    const page = parts[3] != null ? +parts[3] : 0;
    const parents = locationParents(meet);
    const parent = parents[pi];
    if (!parent) {
      await edit(t("loc_error"));
      return;
    }
    const byParent = locationByParent(meet);
    const wheres = [...(byParent.get(parent) || [])];
    const only = wheres[0] || parent;
    const subset = meet.filter((r) => clean(r.w) === only && clean(r.n));
    await showLocPeople(edit, people, subset, only, "ms:loc_menu", `ms:l1:${pi}`, page);
    return;
  }

  if (raw.startsWith("ms:la:")) {
    const parts = raw.slice(6).split(":");
    const pi = +parts[0];
    const page = parts[1] != null ? +parts[1] : 0;
    const parents = locationParents(meet);
    const parent = parents[pi];
    if (!parent) {
      await edit(t("loc_error"));
      return;
    }
    await showLocPeople(
      edit,
      people,
      meetAtParent(meet, parent),
      parent,
      `ms:lp:${pi}`,
      `ms:la:${pi}`,
      page
    );
    return;
  }

  if (raw.startsWith("ms:ls:")) {
    const parts = raw.split(":");
    const pi = +parts[2];
    const si = +parts[3];
    const page = parts[4] != null ? +parts[4] : 0;
    const parents = locationParents(meet);
    const parent = parents[pi];
    if (!parent) {
      await edit(t("loc_error"));
      return;
    }
    const byParent = locationByParent(meet);
    const subs = locationSubs(byParent, parent) || [];
    const sub = subs[si];
    if (sub == null) {
      await edit(t("loc_error"));
      return;
    }
    const label = sub ? `${parent}, ${sub}` : parent;
    await showLocPeople(
      edit,
      people,
      meetAtSub(meet, parent, sub),
      label,
      `ms:lp:${pi}`,
      `ms:ls:${pi}:${si}`,
      page
    );
    return;
  }

  // ---- meet: time ----
  if (raw === "ms:time_menu") {
    const dates = FESTIVAL_DATES.filter((d) => hoursWithMeet(meet, d).length);
    const use = dates.length ? dates : FESTIVAL_DATES;
    const rows = [use.map((d) => ({ text: d, callback_data: `ms:td:${d}` }))];
    rows.push([{ text: t("btn_back"), callback_data: "back:home" }]);
    await edit(t("time_pick_date"), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:td:")) {
    const date = raw.slice(6);
    const hours = hoursWithMeet(meet, date);
    if (!hours.length) {
      await edit(t("time_empty_hours", { date: fmtDate(date) }), {
        inline_keyboard: [[{ text: t("btn_back_dates"), callback_data: "ms:time_menu" }]],
      });
      return;
    }
    const rows = btnRows(hourButtons(hours, date, "ms:th"), 4);
    rows.push([{ text: t("btn_back_dates"), callback_data: "ms:time_menu" }]);
    await edit(t("time_pick_hour"), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:th:")) {
    const [, , date, hh] = raw.split(":");
    const hour = +hh;
    const hourLabel = String(hour).padStart(2, "0");
    const subset = meetAtHour(meet, date, hour);
    const names = uniqueSorted(subset.map((r) => r.n));
    const buttons = names.slice(0, 40).map((nm) => {
      const idx = findPersonByName(people, nm);
      return { text: nm.slice(0, 30) || t("dash"), callback_data: idx >= 0 ? `np:${idx}` : "back:home" };
    });
    const rows = btnRows(buttons, 2);
    rows.push([{ text: t("btn_back_hours"), callback_data: `ms:td:${date}` }]);
    await edit(t("time_header", { date: fmtDate(date), hour: hourLabel, list: formatPeopleTimes(subset) }), {
      inline_keyboard: rows,
    });
    return;
  }

  // ---- meet: topic ----
  if (raw === "ms:topic_menu") {
    const topics = uniqueSorted(meet.map((r) => r.tp));
    const buttons = topics.slice(0, 60).map((topicName, i) => ({
      text: topicName.slice(0, 30) || t("dash"),
      callback_data: `ms:t:${i}`,
    }));
    const rows = btnRows(buttons, 1);
    rows.push([{ text: t("btn_back"), callback_data: "back:home" }]);
    await edit(t("topic_menu_title"), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:t:")) {
    const topics = uniqueSorted(meet.map((r) => r.tp));
    const ti = +raw.slice(5);
    const topic = topics[ti];
    if (!topic) {
      await edit(t("topic_error"));
      return;
    }
    const subset = meet.filter((r) => r.tp === topic);
    const names = uniqueSorted(subset.map((r) => r.n));
    const buttons = names.slice(0, 40).map((nm) => {
      const idx = findPersonByName(people, nm);
      return { text: nm.slice(0, 30) || t("dash"), callback_data: idx >= 0 ? `np:${idx}` : "back:home" };
    });
    const rows = btnRows(buttons, 2);
    rows.push([{ text: t("btn_back_topics"), callback_data: "ms:topic_menu" }]);
    await edit(t("topic_header", { topic: escHtml(topic), list: formatPeopleTimes(subset) }), { inline_keyboard: rows });
    return;
  }

  // ---- meet: event ----
  if (raw === "ms:event_menu") {
    await edit(t("event_menu_title"), {
      inline_keyboard: [
        [{ text: t("btn_event_by_name"), callback_data: "ms:ev:name" }],
        [{ text: t("btn_event_by_time"), callback_data: "ms:ev:time" }],
        [{ text: t("btn_back"), callback_data: "back:home" }],
      ],
    });
    return;
  }

  if (raw === "ms:ev:name") {
    setPending(chatId, "event");
    await edit(t("event_typing_prompt"));
    return;
  }

  if (raw === "ms:ev:time") {
    const dates = FESTIVAL_DATES.filter((d) => hoursWithEventSearch(meet, sched, d).length);
    const use = dates.length ? dates : FESTIVAL_DATES;
    const rows = [use.map((d) => ({ text: d, callback_data: `ms:ev:td:${d}` }))];
    rows.push([{ text: t("btn_back_event_menu"), callback_data: "ms:event_menu" }]);
    await edit(t("event_time_pick_date"), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:ev:td:")) {
    const date = raw.slice("ms:ev:td:".length);
    const hours = hoursWithEventSearch(meet, sched, date);
    if (!hours.length) {
      await edit(t("time_empty_hours", { date: fmtDate(date) }), {
        inline_keyboard: [[{ text: t("btn_back_dates"), callback_data: "ms:ev:time" }]],
      });
      return;
    }
    const rows = btnRows(hourButtons(hours, date, "ms:ev:th"), 4);
    rows.push([{ text: t("btn_back_dates"), callback_data: "ms:ev:time" }]);
    await edit(t("event_time_pick_hour"), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:ev:th:")) {
    const parts = raw.split(":");
    // ms:ev:th:DD.MM:HH
    const date = parts[3];
    const hour = +parts[4];
    const hourLabel = String(hour).padStart(2, "0");
    const namesAt = eventsAtHour(meet, sched, date, hour);
    if (!namesAt.length) {
      await edit(t("event_empty_hour", { date: fmtDate(date), hour: hourLabel }), {
        inline_keyboard: [[{ text: t("btn_back_hours"), callback_data: `ms:ev:td:${date}` }]],
      });
      return;
    }
    const all = allEventNames(meet, sched);
    const items = namesAt
      .map((name) => ({ name, idx: all.indexOf(name) }))
      .filter((x) => x.idx >= 0)
      .slice(0, 40);
    const rows = eventKeyboard(items, `ms:ev:td:${date}`).inline_keyboard;
    // replace last backs: eventKeyboard already adds back + menu; first back goes to hours via our backCb
    await edit(t("event_time_list_header", { date: fmtDate(date), hour: hourLabel }), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("ms:ev:g:")) {
    const idx = +raw.slice("ms:ev:g:".length);
    const events = allEventNames(meet, sched);
    const ev = events[idx];
    if (!ev) {
      await edit(t("event_error"));
      return;
    }
    const subset = peopleOnEvent(meet, sched, ev);
    const names = uniqueSorted(subset.map((r) => r.n));
    const buttons = names.slice(0, 40).map((nm) => {
      const pi = findPersonByName(people, nm);
      return { text: nm.slice(0, 30) || t("dash"), callback_data: pi >= 0 ? `np:${pi}` : "back:home" };
    });
    const rows = btnRows(buttons, 2);
    rows.push([{ text: t("btn_back_event_menu"), callback_data: "ms:event_menu" }]);
    rows.push([{ text: t("btn_back_menu"), callback_data: "back:home" }]);
    const evLink = eventLinkMap(sched).get(ev) || (subset.find((r) => r.u)?.u ?? "");
    await edit(t("event_header", { event: fmtEvent(ev, evLink), list: formatPeopleTimes(subset) }), {
      inline_keyboard: rows,
    });
    return;
  }

  // ---- schedule ----
  if (raw === "schedule:menu") {
    const dates = FESTIVAL_DATES.filter((d) => schedOnDate(sched, d).length);
    const use = dates.length ? dates : FESTIVAL_DATES;
    const buttons = use.map((d) => ({ text: d, callback_data: `schedule:date:${d}` }));
    const rows = btnRows(buttons, 2);
    rows.push([{ text: t("btn_back"), callback_data: "back:home" }]);
    await edit(t("schedule_pick_date"), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("schedule:date:")) {
    const date = raw.slice("schedule:date:".length);
    const hours = hoursWithSched(sched, date);
    const buttons = [];
    if (schedOnDate(sched, date).length) {
      buttons.push({ text: t("btn_full_day"), callback_data: `schedule:full:${date}` });
    }
    buttons.push(...hourButtons(hours, date, "schedule:h"));
    if (!buttons.length) {
      await edit(t("time_empty_hours", { date: fmtDate(date) }), {
        inline_keyboard: [[{ text: t("btn_back_dates"), callback_data: "schedule:menu" }]],
      });
      return;
    }
    const rows = btnRows(buttons, 4);
    rows.push([{ text: t("btn_back_dates"), callback_data: "schedule:menu" }]);
    await edit(t("schedule_pick_time", { date: fmtDate(date) }), { inline_keyboard: rows });
    return;
  }

  if (raw.startsWith("schedule:full:")) {
    const date = raw.slice("schedule:full:".length);
    const events = schedOnDate(sched, date);
    if (!events.length) {
      await edit(t("schedule_empty_day", { date: fmtDate(date) }), {
        inline_keyboard: [[{ text: t("btn_back"), callback_data: `schedule:date:${date}` }]],
      });
      return;
    }
    const buttons = events.slice(0, 40).map((ev, i) => ({
      text: (ev.e || t("dash")).slice(0, 30),
      callback_data: `schedule:ev:${date}:f:${i}`,
    }));
    const rows = btnRows(buttons, 1);
    rows.push([{ text: t("btn_back_time"), callback_data: `schedule:date:${date}` }]);
    await edit(t("schedule_day_header", { date: fmtDate(date), list: formatScheduleEvents(events) }), {
      inline_keyboard: rows,
    });
    return;
  }

  if (raw.startsWith("schedule:h:")) {
    const parts = raw.split(":");
    const date = parts[2];
    const hour = +parts[3];
    const hourLabel = String(hour).padStart(2, "0");
    const events = schedAtHour(sched, date, hour);
    if (!events.length) {
      await edit(t("schedule_empty_hour", { date: fmtDate(date), hour: hourLabel }), {
        inline_keyboard: [[{ text: t("btn_back"), callback_data: `schedule:date:${date}` }]],
      });
      return;
    }
    const buttons = events.slice(0, 40).map((ev, i) => ({
      text: (ev.e || t("dash")).slice(0, 30),
      callback_data: `schedule:ev:${date}:${hourLabel}:${i}`,
    }));
    const rows = btnRows(buttons, 1);
    rows.push([{ text: t("btn_back_time"), callback_data: `schedule:date:${date}` }]);
    await edit(
      t("schedule_hour_header", { date: fmtDate(date), hour: hourLabel, list: formatScheduleEvents(events) }),
      { inline_keyboard: rows }
    );
    return;
  }

  if (raw.startsWith("schedule:ev:")) {
    const rest = raw.slice("schedule:ev:".length);
    const m = rest.match(/^(\d{2}\.\d{2}):(f|\d{2}):(\d+)$/);
    if (!m) {
      await edit(t("schedule_event_not_found"));
      return;
    }
    const date = m[1];
    const mode = m[2];
    const i = +m[3];
    const events = mode === "f" ? schedOnDate(sched, date) : schedAtHour(sched, date, +mode);
    const ev = events[i];
    if (!ev) {
      await edit(t("schedule_event_not_found"));
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
    rows.push([{ text: t("btn_back_menu"), callback_data: "back:home" }]);
    await sendMessage(
      env,
      chatId,
      buttons.length ? t("schedule_people_header") : t("what_next"),
      { inline_keyboard: rows }
    );
    return;
  }

  await edit(t("unknown_command"));
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
            await sendMessage(env, chatId, t("generic_error"));
          }
        } catch (_) {}
      }
      return new Response("ok");
    }

    return new Response("not found", { status: 404 });
  },
};
