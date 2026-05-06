# Epstein API

Read-only Flask API for [Decoherence Media's](https://decoherence.media/) [epstein.photos](https://epstein.photos/) project, using the slimmed production SQLite database, deployed with Gunicorn. Lives next to [`epstein-web`](https://github.com/decoherencemedia/epstein-web) and [`epstein-pipeline`](https://github.com/decoherencemedia/epstein-pipeline) repos.

For more information, read [our guide](https://decoherence.media/we-identified-more-than-400-people-in-photos-from-the-epstein-files/).

## Routes

- `/photos` — Search photos either by person IDs or document prefix (EFTA / House Oversight stems), with face geometry per image.

- `/faces` — Returns faces and names used in *People* page (name, counts, best-face URL), with optional class filters.Victims and minors omitted.

- `/people` — Mapping of in-network `person_id` to display name.

## TODO

- Organization could probably be improved
- More tests
