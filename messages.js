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
  btn_by_topic: "🏷️ по теме",
  btn_by_event: "🎫 по ивенту",
  btn_schedule: "📅 моё расписание",
  btn_settings: "⚙️ настройки",
  btn_restart: "🔄 Перезапустить",

  // —— общие кнопки ——
  btn_back: "⬅️ Назад",
  btn_back_menu: "⬅️ В меню",
  btn_back_letters: "⬅️ Назад к буквам",
  btn_back_locations: "⬅️ Назад к локациям",
  btn_back_dates: "⬅️ Назад к датам",
  btn_back_hours: "⬅️ Назад к часам",
  btn_back_topics: "⬅️ Назад к темам",
  btn_back_events: "⬅️ Назад к ивентам",
  btn_back_time: "⬅️ Назад к времени",
  btn_full_day: "📅 Весь день",
  btn_name_alpha: "🔤 Имена в алфавитном порядке",
  btn_name_typing: "⌨️ Введите имя",
  btn_refresh_data: "🔄 Обновить данные",
  btn_event_by_name: "🔎 Поиск по названию",
  btn_event_by_time: "🕒 Поиск по дате и времени",
  btn_back_event_menu: "⬅️ К поиску ивентов",

  // —— экраны ——
  main_menu: "Главное меню:",
  restarted: "🔄 Бот перезапущен!\n\nГлавное меню:",

  name_menu_title: "Как искать по имени?",
  name_typing_prompt: "Введи имя/фамилию для поиска (обычным сообщением):",
  name_pick_letter: "Выбери букву:",
  name_letter_title: "Имена на букву {letter}:",
  name_pick_person: "Выбери человека:",
  name_not_found_search: "Ничего не нашлось. Попробуй по-другому.",
  name_not_found_letter: "Ничего не найдено.",
  person_not_found: "Не удалось найти человека.",
  what_next: "Что дальше?",

  loc_menu_title: "Где ты сейчас?",
  loc_error: "Ошибка выбора локации",
  loc_header: "Локация: {loc}\n\n{list}",

  time_pick_date: "Выбери дату:",
  time_pick_hour: "Который час?",
  time_header: "{date} {hour}:00\n\n{list}",

  topic_menu_title: "Выбери тему:",
  topic_error: "Ошибка выбора темы",
  topic_header: "Тема: {topic}\n\n{list}",

  event_menu_title: "Как искать ивент?",
  event_typing_prompt: "Введи название ивента/проекта (обычным сообщением):",
  event_pick: "Выбери ивент:",
  event_not_found_search: "Ничего не нашлось. Попробуй другое название.",
  event_error: "Ошибка выбора ивента",
  event_header: "Ивент: {event}\n\n{list}",
  event_empty_hour: "На {date} в {hour}:00 нет ивентов.",
  event_time_pick_date: "Выбери дату ивента:",
  event_time_pick_hour: "Который час?",
  event_time_list_header: "Ивенты на {date} в {hour}:00:",

  schedule_pick_date: "Выбери дату:",
  schedule_pick_time: "Выбери время для {date}:",
  schedule_day_header: "Расписание на {date}\n\n{list}",
  schedule_hour_header: "{date} {hour}:00\n\n{list}",
  schedule_empty_day: "На {date} нет событий.",
  schedule_empty_hour: "На {date} в {hour}:00 нет событий.",
  schedule_event_not_found: "Не удалось найти событие.",
  schedule_people_header: "Люди на этом событии:",

  nothing_found: "Ничего не найдено.",
  unknown_command: "Неизвестная команда. Нажми /start",
  generic_error: "Произошла ошибка. Попробуйте /start ещё раз.",

  // —— настройки / кэш ——
  settings_title:
    "⚙️ Настройки\n\nИсточник: Google Sheet\nКэш: {cacheAge}\nTTL: {ttlMin} мин\n\npeople: {nPeople}\nmeet: {nMeet}\nschedule: {nSched}",
  settings_refreshing: "⏳ Обновляю данные из Google Sheet…",
  settings_refreshed:
    "✅ Данные обновлены\n\nКэш: только что\npeople: {nPeople}\nmeet: {nMeet}\nschedule: {nSched}",
  settings_refresh_failed:
    "❌ Не удалось обновить:\n{error}\n\nПроверь, что Sheet доступен по ссылке «Anyone with the link».",
  cache_empty: "кэш пуст",
  cache_sec_ago: "{n} сек назад",
  cache_min_ago: "{n} мин назад",

  // —— карточка человека ——
  person_festival_role: "🎫 {value}",
  person_bio: "✏️bio: {value}",
  person_tip: "💡tip: {value}",
  person_contact: "📱 {value}",

  // —— списки слотов / расписание ——
  people_times_person: "👤 {name}\n{slots}",
  people_times_slot: "{date} {start}–{finish}\n📍 {where}\n🎫 {event}",
  schedule_list_item: "🎫 {name}\n{start}–{finish}",
  schedule_list_where: "\n📍 {where}",
  schedule_card_name: "🎫 {value}",
  schedule_card_time: "🕒 {start}–{finish}",
  schedule_card_where: "📍 {value}",
  schedule_card_type: "🏷️ {value}",
  schedule_card_desc: "📝 {value}",
  schedule_card_people: "👥 {value}",
  schedule_card_reg: "📋 {value}",
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
