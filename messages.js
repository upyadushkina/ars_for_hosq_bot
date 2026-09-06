/**
 * Тексты бота — редактируй здесь вручную.
 * После правок залей messages.js на GitHub и сделай redeploy.
 *
 * Плейсхолдеры в фигурных скобках подставляются кодом, например {letter}, {date}.
 * Их имена лучше не менять.
 */

export const T = {
  // —— кнопки главного меню ——
  btn_search_name: "🔎 поиск по имени",
  btn_by_location: "📍 по локации",
  btn_by_time: "🕒 по времени",
  btn_by_topic: "по теме",
  btn_by_event: "по ивенту",
  btn_schedule: "✦ моё расписание ✦",
  btn_settings: "⚙️ настройки",
  btn_restart: "restart",

  // —— общие кнопки ——
  btn_back: "⬅ назад",
  btn_back_menu: "⬅ меню",
  btn_back_letters: "⬅ назад к буквам",
  btn_back_locations: "⬅ назад к локациям",
  btn_loc_all: "все",
  btn_loc_unspecified: "без уточнения",
  btn_loc_prev: "◀",
  btn_loc_next: "▶",
  btn_loc_page: "{page}/{pages}",
  btn_back_dates: "⬅ назад к датам",
  btn_back_hours: "⬅ назад к часам",
  btn_back_topics: "⬅ назад к темам",
  btn_back_events: "⬅ назад к ивентам",
  btn_back_time: "⬅ назад к времени",
  btn_full_day: "весь день",
  btn_name_alpha: "🔤 имена в алфавитном порядке",
  btn_name_typing: "введите имя",
  btn_refresh_data: "🔄 обновить данные",
  btn_event_by_name: "🔎 поиск по названию",
  btn_event_by_time: "🕒 поиск по дате и времени",
  btn_back_event_menu: "⬅ поиск ивентов",

  // —— экраны ——
  main_menu: "✦ меню ✦",
  restarted: "🔄 бот перезапущен!\n\n✦ меню ✦",

  name_menu_title: "как искать по имени?",
  name_typing_prompt: "введи имя/фамилию для поиска (обычным сообщением):",
  name_pick_letter: "выбери букву:",
  name_letter_title: "имена на букву {letter}:",
  name_pick_person: "выбери человека:",
  name_not_found_search: "ничего не нашлось. попробуй по-другому.",
  name_not_found_letter: "ничего не найдено.",
  person_not_found: "не удалось найти человека.",
  what_next: "что дальше?",

  loc_menu_title: "где ты сейчас?",
  loc_pick_room: "{loc} — выбери зону:",
  loc_error: "ошибка выбора локации",
  loc_header: "локация: {loc}\n\n{list}",
  loc_header_paged: "локация: {loc}\nстр. {page}/{pages}\n\n{list}",
  loc_people_compact: "{n} человек на этой странице — выбери имя ниже:\n\n{list}",

  time_pick_date: "выбери дату:",
  time_pick_hour: "выбери время:",
  time_header: "{date} {hour}:00\n\n{list}",

  topic_menu_title: "выбери тему:",
  topic_error: "ошибка выбора темы",
  topic_header: "тема: {topic}\n\n{list}",

  event_menu_title: "как искать ивент?",
  event_typing_prompt: "введи название ивента/проекта (обычным сообщением):",
  event_pick: "выбери ивент:",
  event_not_found_search: "ничего не нашлось. попробуй другое название.",
  event_error: "ошибка выбора ивента",
  event_header: "ивент: {event}\n\n{list}",
  event_empty_hour: "на {date} в {hour}:00 нет ивентов.",
  event_time_pick_date: "выбери дату ивента:",
  event_time_pick_hour: "который час?",
  event_time_list_header: "ивенты на {date} в {hour}:00:",

  schedule_pick_date: "выбери дату:",
  schedule_pick_time: "выбери время для {date}:",
  schedule_day_header: "расписание на {date}\n\n{list}",
  schedule_hour_header: "{date} {hour}:00\n\n{list}",
  schedule_empty_day: "на {date} нет событий.",
  schedule_empty_hour: "на {date} в {hour}:00 нет событий.",
  schedule_event_not_found: "не удалось найти событие.",
  schedule_people_header: "люди на этом событии:",

  nothing_found: "ничего не найдено.",
  unknown_command: "неизвестная команда. нажми /start",
  generic_error: "произошла ошибка. попробуйте /start ещё раз.",

  // —— настройки / кэш ——
  settings_title:
    "⚙️ настройки\n\nисточник: Google Sheet: https://docs.google.com/spreadsheets/d/1X6oouceCLD3pY289WtT-gPeU_kxeq6mpilbLNehSGCw/edit?usp=sharing\nКэш: {cacheAge}\nTTL: {ttlMin} мин\n\nлюди: {nPeople}\nивенты: {nMeet}\nрасписание: {nSched}",
  settings_refreshing: "⏳ Обновляю данные из Google Sheet…",
  settings_refreshed:
    "✅ данные обновлены\n\nКэш: только что\npeople: {nPeople}\nmeet: {nMeet}\nschedule: {nSched}",
  settings_refresh_failed:
    "❌ не удалось обновить:\n{error}\n\nпроверь, что Sheet доступен по ссылке «Anyone with the link».",
  cache_empty: "кэш пуст",
  cache_sec_ago: "{n} сек назад",
  cache_min_ago: "{n} мин назад",

  // —— карточка человека ——
  person_festival_role: "{value}",
  person_bio: "bio: {value}",
  person_tip: "✦ tip: {value}",
  person_contact: "📱 {value}",

  // —— списки слотов / расписание ——
  people_times_person: "👤 {name}\n{slots}",
  people_times_slot: "{date} {start}–{finish}\n📍 {where}\n ✦ {event}",
  schedule_list_item: "{name}\n{start}–{finish}",
  schedule_list_where: "\n📍 {where}",
  schedule_card_name: "{value}",
  schedule_card_time: "🕒 {start}–{finish}",
  schedule_card_where: "📍 {value}",
  schedule_card_type: "{value}",
  schedule_card_desc: "{value}",
  schedule_card_people: "👥 {value}",
  schedule_card_reg: "{value}",
  schedule_card_link: "🔗 {value}",

  dash: "—",
};

/** Подстановка {key} → values[key] */
export function t(key, values = {}) {
  let s = T[key];
  if (s == null) return key;
  return s.replace(/\{(\w+)\}/g, (_, k) =>
    values[k] != null ? String(values[k]) : ""
  );
}
