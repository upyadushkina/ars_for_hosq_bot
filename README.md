# Ars Electronica 2026 — hosq Meet Bot

Telegram bot for finding people at Ars Electronica Festival 2026 (Negotiating Humanity, 9–13 Sep, plus Pre-Opening 8 Sep).

Same UX as the 2025 bot: search by name, location, time, topic, event, plus personal schedule.

## Data

Built from the official festival CMS export:
https://ars.electronica.art/negotiatinghumanity/hackathondata/

| File | Contents |
|------|----------|
| `ars_2026_people.csv` | People + groups (bios, roles, hosq tips) |
| `meet_slots.csv` | When/where each person appears |
| `ars_schedule.csv` | Full timed festival calendar |

Rebuild CSVs after a new export:

```bash
curl -sL -A "Mozilla/5.0" \
  "https://ars.electronica.art/negotiatinghumanity/hackathondata/" \
  -o "../DATA/raw/festival_2026.json"
python3 ../scripts/build_2026_databases.py
```

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export TELEGRAM_TOKEN="..."
# optional: WEBHOOK_SECRET, PEOPLE_CSV, MEET_SLOTS_CSV, SCHEDULE_CSV
uvicorn app:app --reload --port 8000
```

For webhook deploy, point Telegram to `POST /webhook`.

## Notes

- `ars_schedule.csv` is the **full** public calendar. For “Моё расписание”, trim rows or set `registration_status` to your team’s plan (as in 2025).
- Photos are empty in the official export — add URLs in `Photo` if you want cards with images.
- Conversation tips are hosq-oriented templates; edit freely.
