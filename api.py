import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, jsonify, request
from flask_caching import Cache
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default: repo scripts/faces.db when running from api/; override with EPSTEIN_SQLITE_PATH.
_DEFAULT_SQLITE = Path(__file__).resolve().parent.parent / "network"/ "scripts" / "faces.db"
SQLITE_PATH = os.environ.get("EPSTEIN_SQLITE_PATH", str(_DEFAULT_SQLITE))

cache = Cache(
    config={"CACHE_TYPE": "flask_caching.backends.filesystem", "CACHE_DIR": "/tmp"}
)


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

CORS(app)
cache.init_app(app)


@contextmanager
def get_db_connection():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@app.route("/")
def description():
    return {
        "message": "Welcome to the API of Epstein.Photos, which maps out the network of connections between people in the Epstein Library."
    }

def photos_for_all_person_ids(person_ids: list[str]) -> list[dict]:
    """
    Return one object per image: each image contains every requested person_id at least once.

    ``faces`` is a list of every face row in that image (all ``person_id`` values), including
    multiple boxes for the same person when they appear more than once. Each element is::

        {"face_id": str, "person_id": str, "left", "top", "width", "height"}

    Coordinates are Rekognition-normalized (0..1). Matches ``normalizeFaceEntries`` in
    ``site/pages/people-inner.html``.
    """
    unique = sorted(set(p for p in person_ids if p and str(p).strip()))
    n = len(unique)
    if n == 0:
        return []
    max_images = 1000

    placeholders_in = ",".join("?" * n)
    # Params: IN (n) for qualifying + HAVING (1) = n + 1
    sql = f"""
        WITH qualifying AS (
            SELECT f.image_name
            FROM faces AS f
            INNER JOIN images AS i ON i.image_name = f.image_name
            WHERE f.person_id IN ({placeholders_in})
                AND i.duplicate_of IS NULL
                AND COALESCE(i.is_explicit, 0) = 0
                AND COALESCE(i.contains_victim, 0) = 0
            GROUP BY f.image_name
            HAVING COUNT(DISTINCT f.person_id) = ?
        )
        SELECT
            f.image_name,
            f.face_id,
            f.person_id,
            f.left,
            f.top,
            f.width,
            f.height
        FROM faces AS f
        INNER JOIN images AS i ON i.image_name = f.image_name
        INNER JOIN qualifying AS q ON q.image_name = f.image_name
        WHERE f.person_id IS NOT NULL
            AND TRIM(f.person_id) != ''
            AND i.duplicate_of IS NULL
            AND COALESCE(i.is_explicit, 0) = 0
            AND COALESCE(i.contains_victim, 0) = 0
        ORDER BY f.image_name, f.person_id, f.face_id;
    """
    params = tuple(unique) + (n,)

    with get_db_connection() as conn:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()

    order: list[str] = []
    by_image: dict[str, dict] = {}
    for r in rows:
        webp = r["image_name"].replace(".jpg", ".webp")
        if webp not in by_image:
            if len(order) >= max_images:
                break
            by_image[webp] = {
                "image": webp,
                "faces": [],
            }
            order.append(webp)
        pid = r["person_id"]
        by_image[webp]["faces"].append(
            {
                "face_id": r["face_id"],
                "person_id": pid,
                "left": float(r["left"]),
                "top": float(r["top"]),
                "width": float(r["width"]),
                "height": float(r["height"]),
            }
        )

    return [by_image[k] for k in order]


def network_person_id_to_name() -> dict[str, str | None]:
    """
    person_id -> display name for people included in the network (include_in_network = 1).

    Victims have anonymized names in SQLite (``Victim 1``, …) from sheet sync, not real names.
    """
    sql = """
        SELECT person_id, name
        FROM people
        WHERE include_in_network = 1
    """
    with get_db_connection() as conn:
        cur = conn.execute(sql)
        rows = cur.fetchall()
    return {r["person_id"]: r["name"] for r in rows}


@app.route("/people")
@cache.cached(timeout=14400)
def get_network_people_names():
    """JSON object mapping person_id to display name (network members only)."""
    data = network_person_id_to_name()
    return jsonify(data=data)


@app.route("/photos")
@cache.cached(timeout=14400, query_string=True)
def get_photos_by_people():
    """
    Query: person_ids — comma-separated list, e.g. /photos?person_ids=person_1,person_2
    Returns images where every listed person appears on the same image (at least one face each).
    Each element is {"image": "<file>.webp", "faces": [ {...}, ... ]} — one object per face
    row in the DB for that image (including multiple appearances of the same person).
    """
    raw = request.args.get("person_ids", "").strip()
    if not raw:
        return jsonify(
            error="Bad Request",
            message="Missing person_ids query parameter (comma-separated).",
        ), 400

    person_ids = [p.strip() for p in raw.split(",") if p.strip()]
    if not person_ids:
        return jsonify(
            error="Bad Request",
            message="person_ids must contain at least one id.",
        ), 400

    data = photos_for_all_person_ids(person_ids)
    return jsonify(data=data)


def start():
    if not Path(SQLITE_PATH).is_file():
        logger.warning("SQLite file not found at %s — API will error on first request.", SQLITE_PATH)
    app.run()


if __name__ == "__main__":
    start()
