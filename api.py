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
    Return image rows for images that contain at least one face for every person_id
    (join faces + images; GROUP BY image; HAVING COUNT(DISTINCT person_id) = N).
    """
    unique = sorted(set(p for p in person_ids if p and str(p).strip()))
    n = len(unique)
    if n == 0:
        return []

    placeholders_in = ",".join("?" * n)
    # Second parameter to HAVING is n (must match distinct requested ids present on image).
    sql = f"""
        SELECT i.image_name
        FROM faces AS f
        INNER JOIN images AS i ON i.image_name = f.image_name
        WHERE f.person_id IN ({placeholders_in})
          AND i.duplicate_of IS NULL
        GROUP BY f.image_name
        HAVING COUNT(DISTINCT f.person_id) = ?
        ORDER BY f.image_name
    """
    params = tuple(unique) + (n,)

    with get_db_connection() as conn:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()

    return [r["image_name"].replace(".jpg", ".webp") for r in rows]


def network_person_id_to_name() -> dict[str, str | None]:
    """person_id -> name for people included in the network (include_in_network = 1)."""
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
