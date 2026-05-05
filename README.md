# Epstein API

Read-only Flask API for [epstein.photos](https://epstein.photos/) using the slimmed production SQLite database, deployed with Gunicorn. Lives next to `epstein-web` and `epstein-pipeline` repos.

## Routes

- `/photos` — Search photos either by person IDs or document prefix (EFTA / House Oversight stems), with face geometry per image.

- `/faces` — Returns faces and names used in *People* page (name, counts, best-face URL), with optional class filters.Victims and minors omitted.

- `/people` — Mapping of in-network `person_id` to display name.

## TODO

- Organization could probably be improved
- More tests
