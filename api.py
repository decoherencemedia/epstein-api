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

# Default: ../faces.db (umbrella network/ next to epstein-api/ and epstein-pipeline/). Override with EPSTEIN_SQLITE_PATH.
_DEFAULT_SQLITE = Path(__file__).resolve().parent.parent / "faces.db"
SQLITE_PATH = os.environ.get("EPSTEIN_SQLITE_PATH", str(_DEFAULT_SQLITE))

# Public URL prefix for best-face crops (13__optimize_node_faces.sh output); no trailing slash.
BEST_FACE_IMAGE_BASE = os.environ.get(
    "BEST_FACE_IMAGE_BASE", "/faces"
)

# Distinct images per /photos response page (default limit; also the maximum allowed limit).
PHOTO_PAGE_SIZE = 1000

cache = Cache(
    config={"CACHE_TYPE": "flask_caching.backends.filesystem", "CACHE_DIR": "/tmp/flask"}
)


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

CORS(app)
cache.init_app(app)


def _parse_pagination_args() -> tuple[int, int]:
    limit_raw = request.args.get("limit", str(PHOTO_PAGE_SIZE))
    offset_raw = request.args.get("offset", "0")
    limit = int(limit_raw)
    offset = int(offset_raw)
    if limit < 1 or limit > PHOTO_PAGE_SIZE:
        raise ValueError(
            f"limit must be between 1 and {PHOTO_PAGE_SIZE} inclusive, got {limit!r}"
        )
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset!r}")
    return limit, offset


def _aggregate_faces_ordered(rows: list[sqlite3.Row], image_names_order: list[str]) -> list[dict]:
    """Build photo items in ``image_names_order`` (DB ``image_name`` values, typically .jpg)."""
    order_webp = [n.replace(".jpg", ".webp") for n in image_names_order]
    by_webp: dict[str, dict] = {}
    for w in order_webp:
        by_webp[w] = {"image": w, "faces": []}
    for r in rows:
        webp = r["image_name"].replace(".jpg", ".webp")
        if webp not in by_webp:
            raise RuntimeError(
                f"Face row image_name not in page list: {r['image_name']!r}"
            )
        by_webp[webp]["faces"].append(
            {
                "face_id": r["face_id"],
                "person_id": r["person_id"],
                "left": float(r["left"]),
                "top": float(r["top"]),
                "width": float(r["width"]),
                "height": float(r["height"]),
            }
        )
    out: list[dict] = []
    for w in order_webp:
        item = by_webp[w]
        if not item["faces"]:
            raise RuntimeError(f"No face rows aggregated for ordered image {w!r}")
        out.append(item)
    return out


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

def photos_for_all_person_ids(
    person_ids: list[str], *, limit: int, offset: int
) -> tuple[list[dict], int]:
    """
    Return one object per image: each image contains every requested person_id at least once.

    ``faces`` is a list of every face row in that image (all ``person_id`` values), including
    multiple boxes for the same person when they appear more than once. Each element is::

        {"face_id": str, "person_id": str, "left", "top", "width", "height"}

    Coordinates are Rekognition-normalized (0..1). Matches ``normalizeFaceEntries`` in
    ``site/pages/people-inner.html``.

    Paginates distinct ``image_name`` values; returns ``(page_items, total_count)``.
    """
    unique = sorted(set(p for p in person_ids if p and str(p).strip()))
    n = len(unique)
    if n == 0:
        return [], 0

    placeholders_in = ",".join("?" * n)
    qualifying_sql = f"""
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
    """
    base_params = tuple(unique) + (n,)

    count_sql = qualifying_sql + " SELECT COUNT(*) AS c FROM qualifying"
    names_sql = (
        qualifying_sql
        + """
        SELECT q.image_name
        FROM qualifying AS q
        ORDER BY q.image_name
        LIMIT ? OFFSET ?;
    """
    )

    with get_db_connection() as conn:
        total = int(conn.execute(count_sql, base_params).fetchone()["c"])
        name_rows = conn.execute(names_sql, base_params + (limit, offset)).fetchall()

    names = [r["image_name"] for r in name_rows]
    if not names:
        return [], total

    placeholders_names = ",".join("?" * len(names))
    faces_sql = f"""
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
        WHERE f.image_name IN ({placeholders_names})
            AND f.person_id IS NOT NULL
            AND TRIM(f.person_id) != ''
            AND i.duplicate_of IS NULL
            AND COALESCE(i.is_explicit, 0) = 0
            AND COALESCE(i.contains_victim, 0) = 0
        ORDER BY f.image_name, f.person_id, f.face_id;
    """
    with get_db_connection() as conn:
        rows = conn.execute(faces_sql, tuple(names)).fetchall()

    return _aggregate_faces_ordered(rows, names), total


DOCUMENT_ID_PREFIXES = ("EFTA", "BIRTHDAY_BOOK_", "HOUSE_OVERSIGHT_")


def is_valid_partial_document_query(s: str) -> bool:
    """
    True if ``s`` is empty, or could still become ``PREFIX`` + digits for one of
    ``DOCUMENT_ID_PREFIXES`` (case-insensitive). Mirrors ``epstein-web/site/js/shared.js``.
    """
    t = s.strip()
    if not t:
        return True
    u = t.upper()
    for p in DOCUMENT_ID_PREFIXES:
        if p.startswith(u):
            return True
        if u.startswith(p):
            rest = u[len(p) :]
            if rest == "" or rest.isdigit():
                return True
            return False
    return False


def _normalize_document_prefix(raw: str) -> str | None:
    """
    Non-empty, valid partial query → uppercase stem for ``LIKE prefix%``.
    """
    p = raw.strip()
    if not p:
        return None
    if not is_valid_partial_document_query(p):
        return None
    return p.upper()


def photos_for_document_prefix(
    prefix: str, *, limit: int, offset: int
) -> tuple[list[dict], int]:
    """
    Return one object per image whose ``image_name`` starts with ``prefix`` (same filters as
    ``photos_for_all_person_ids``: non-duplicate, non-explicit, no victim images).

    Paginates distinct ``image_name`` values; returns ``(page_items, total_count)``.
    """
    unique_prefix = _normalize_document_prefix(prefix)
    if unique_prefix is None:
        return [], 0

    pattern = unique_prefix + "%"

    base_from = """
        FROM faces AS f
        INNER JOIN images AS i ON i.image_name = f.image_name
        WHERE f.image_name LIKE ?
            AND (
                f.image_name LIKE 'EFTA%'
                OR f.image_name LIKE 'BIRTHDAY_BOOK_%'
                OR f.image_name LIKE 'HOUSE_OVERSIGHT_%'
            )
            AND f.person_id IS NOT NULL
            AND TRIM(f.person_id) != ''
            AND i.duplicate_of IS NULL
            AND COALESCE(i.is_explicit, 0) = 0
            AND COALESCE(i.contains_victim, 0) = 0
    """
    count_sql = "SELECT COUNT(DISTINCT f.image_name) AS c " + base_from
    names_sql = (
        "SELECT DISTINCT f.image_name "
        + base_from
        + " ORDER BY f.image_name LIMIT ? OFFSET ?"
    )

    with get_db_connection() as conn:
        total = int(conn.execute(count_sql, (pattern,)).fetchone()["c"])
        name_rows = conn.execute(names_sql, (pattern, limit, offset)).fetchall()

    names = [r["image_name"] for r in name_rows]
    if not names:
        return [], total

    placeholders = ",".join("?" * len(names))
    faces_sql = f"""
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
        WHERE f.image_name IN ({placeholders})
            AND f.person_id IS NOT NULL
            AND TRIM(f.person_id) != ''
            AND i.duplicate_of IS NULL
            AND COALESCE(i.is_explicit, 0) = 0
            AND COALESCE(i.contains_victim, 0) = 0
        ORDER BY f.image_name, f.person_id, f.face_id;
    """
    with get_db_connection() as conn:
        rows = conn.execute(faces_sql, tuple(names)).fetchall()

    return _aggregate_faces_ordered(rows, names), total


def _sanitize_label_for_filename(label: str) -> str:
    """Match ``scripts/12__export_node_faces.py`` filename stems."""
    s = "".join(c if c.isalnum() or c in "._- " else "" for c in label)
    return s.strip().replace(" ", "_").replace("/", "_") or "node"


def faces_list() -> list[dict]:
    """
    Network members only (``include_in_network = 1``), excluding victims (``is_victim = 0``),
    with photo counts and optional best-face image URL.

    ``photo_count`` is one aggregated pass over ``faces`` + ``images`` (``LEFT JOIN``), not a
    per-row correlated subquery.

    Sort: (1) has ``best_face_id``, (2) has non-empty ``name``, (3) ``photo_count`` (desc),
    then ``person_id``.
    """
    sql = """
        SELECT
            p.person_id,
            p.name,
            p.best_face_id,
            COALESCE(pc.cnt, 0) AS photo_count
        FROM people AS p
        LEFT JOIN (
            SELECT f.person_id AS pid, COUNT(DISTINCT f.image_name) AS cnt
            FROM faces AS f
            INNER JOIN images AS i ON i.image_name = f.image_name
            WHERE i.duplicate_of IS NULL
                AND COALESCE(i.is_explicit, 0) = 0
                AND COALESCE(i.contains_victim, 0) = 0
            GROUP BY f.person_id
        ) AS pc ON pc.pid = p.person_id
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

    - document_prefix — case-insensitive stem prefix, e.g. ``EFTA00000005``, ``efta001``,
      ``HOUSE_OVERSIGHT_12``. Returns images whose ``image_name`` matches ``prefix%`` within
      EFTA / BIRTHDAY_BOOK_ / HOUSE_OVERSIGHT families.

    Each element is {"image": "<file>.webp", "faces": [ {...}, ... ]} — one object per face
    row in the DB for that image (including multiple appearances of the same person).

    Optional pagination: ``limit`` (defaults to ``PHOTO_PAGE_SIZE``, cannot exceed it) and
    ``offset`` (default 0) apply to
    distinct images. The response includes ``total`` (full match count), ``limit``, and ``offset``.

    Mutually exclusive: pass **either** ``document_prefix`` **or** ``person_ids``, not both.
    """
    limit, offset = _parse_pagination_args()
    doc_raw = request.args.get("document_prefix", "").strip()
    raw = request.args.get("person_ids", "").strip()

    if doc_raw and raw:
        return jsonify(
            error="Bad Request",
            message="Use either document_prefix or person_ids, not both.",
        ), 400

    if doc_raw:
        normalized = _normalize_document_prefix(doc_raw)
        if normalized is None:
            return jsonify(
                error="Bad Request",
                message=(
                    "Invalid document_prefix: use a prefix of EFTA, BIRTHDAY_BOOK_, or HOUSE_OVERSIGHT_ "
                    "followed by digits (case-insensitive), e.g. EFTA00000005 or house_oversight_001."
                ),
            ), 400
        data, total = photos_for_document_prefix(normalized, limit=limit, offset=offset)
        return jsonify(data=data, total=total, limit=limit, offset=offset)

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

    data, total = photos_for_all_person_ids(person_ids, limit=limit, offset=offset)
    return jsonify(data=data, total=total, limit=limit, offset=offset)


def start():
    if not Path(SQLITE_PATH).is_file():
        logger.warning("SQLite file not found at %s — API will error on first request.", SQLITE_PATH)
    app.run()


if __name__ == "__main__":
    start()
