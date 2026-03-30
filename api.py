import logging
import os
import re
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

# Public URL prefix for best-face crops (13__optimize_node_faces.sh output); no trailing slash.
BEST_FACE_IMAGE_BASE = os.environ.get(
    "BEST_FACE_IMAGE_BASE", "/faces"
)

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


def _validate_document_prefix(raw: str) -> str | None:
    """
    Document stem prefix for image_name (e.g. EFTA00000005 matches EFTA00000005-00001.webp).
    Must be literal ``EFTA`` (case-insensitive) followed by exactly 8 digits. Normalized to ``EFTA`` + digits.
    """
    p = raw.strip()
    if not p:
        return None
    m = re.fullmatch(r"(?i)efta(\d{8})", p)
    if not m:
        return None
    return "EFTA" + m.group(1)


def photos_for_document_prefix(prefix: str) -> list[dict]:
    """
    Return one object per image whose ``image_name`` starts with ``prefix`` (same filters as
    ``photos_for_all_person_ids``: non-duplicate, non-explicit, no victim images).
    """
    unique_prefix = _validate_document_prefix(prefix)
    if unique_prefix is None:
        return []

    max_images = 1000
    pattern = unique_prefix + "%"

    sql = """
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
        WHERE f.image_name LIKE ?
            AND f.person_id IS NOT NULL
            AND TRIM(f.person_id) != ''
            AND i.duplicate_of IS NULL
            AND COALESCE(i.is_explicit, 0) = 0
            AND COALESCE(i.contains_victim, 0) = 0
        ORDER BY f.image_name, f.person_id, f.face_id;
    """

    with get_db_connection() as conn:
        cur = conn.execute(sql, (pattern,))
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


def _sanitize_label_for_filename(label: str) -> str:
    """Match ``scripts/12__export_node_faces.py`` filename stems."""
    s = "".join(c if c.isalnum() or c in "._- " else "" for c in label)
    return s.strip().replace(" ", "_").replace("/", "_") or "node"


def faces_list() -> list[dict]:
    """
    Network members only (``include_in_network = 1``), excluding victims (``is_victim = 0``),
    with photo counts and optional best-face image URL.

    Sort: (1) has ``best_face_id``, (2) has non-empty ``name``, (3) ``photo_count`` (desc),
    then ``person_id``.
    """
    sql = """
        SELECT
            p.person_id,
            p.name,
            p.best_face_id,
            (
                SELECT COUNT(DISTINCT f.image_name)
                FROM faces AS f
                INNER JOIN images AS i ON i.image_name = f.image_name
                WHERE f.person_id = p.person_id
                    AND i.duplicate_of IS NULL
                    AND COALESCE(i.is_explicit, 0) = 0
                    AND COALESCE(i.contains_victim, 0) = 0
            ) AS photo_count
        FROM people AS p
        WHERE p.include_in_network = 1
            AND COALESCE(p.is_victim, 0) = 0
        ORDER BY
            CASE
                WHEN p.best_face_id IS NOT NULL AND TRIM(COALESCE(p.best_face_id, '')) != ''
                THEN 1 ELSE 0
            END DESC,
            CASE
                WHEN p.name IS NOT NULL AND TRIM(COALESCE(p.name, '')) != ''
                THEN 1 ELSE 0
            END DESC,
            photo_count DESC,
            p.person_id
    """
    with get_db_connection() as conn:
        rows = conn.execute(sql).fetchall()

    base = BEST_FACE_IMAGE_BASE.rstrip("/")
    out: list[dict] = []
    for r in rows:
        pid = r["person_id"]
        name = r["name"]
        bf = r["best_face_id"]
        label = (name or "").strip() or pid
        safe = _sanitize_label_for_filename(label)
        image_url: str | None = None
        if bf is not None and str(bf).strip():
            image_url = f"{base}/{safe}_{str(bf).strip()}.webp"
        name_out: str | None = None
        if name is not None and str(name).strip():
            name_out = str(name).strip()
        out.append(
            {
                "person_id": pid,
                "image": image_url,
                "photo_count": int(r["photo_count"]),
                "name": name_out,
            }
        )
    return out


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


@app.route("/faces")
@cache.cached(timeout=14400)
def get_faces():
    """
    List of people with ``person_id``, ``image`` (best-face WebP URL or null),
    ``photo_count`` (distinct images in the public API sense), and ``name`` (or null).

    Only ``include_in_network = 1`` and non-victim rows. Order: best_face set, has name,
    higher ``photo_count``, then ``person_id``.
    """
    return jsonify(data=faces_list())


@app.route("/photos")
@cache.cached(timeout=14400, query_string=True)
def get_photos_by_people():
    """
    Query (one of):

    - person_ids — comma-separated list, e.g. /photos?person_ids=person_1,person_2
      Returns images where every listed person appears on the same image (at least one face each).

    - document_prefix — stem prefix, e.g. /photos?document_prefix=EFTA00000005
      Returns images whose filename starts with that prefix (e.g. EFTA00000005-00001.webp).

    Each element is {"image": "<file>.webp", "faces": [ {...}, ... ]} — one object per face
    row in the DB for that image (including multiple appearances of the same person).

    Mutually exclusive: pass **either** ``document_prefix`` **or** ``person_ids``, not both.
    """
    doc_raw = request.args.get("document_prefix", "").strip()
    raw = request.args.get("person_ids", "").strip()

    if doc_raw and raw:
        return jsonify(
            error="Bad Request",
            message="Use either document_prefix or person_ids, not both.",
        ), 400

    if doc_raw:
        normalized = _validate_document_prefix(doc_raw)
        if normalized is None:
            return jsonify(
                error="Bad Request",
                message="Invalid document_prefix: must be EFTA followed by exactly 8 digits (e.g. EFTA00000005).",
            ), 400
        data = photos_for_document_prefix(normalized)
        return jsonify(data=data)

    if not raw:
        return jsonify(
            error="Bad Request",
            message="Missing query: pass person_ids (comma-separated) or document_prefix (not both).",
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
