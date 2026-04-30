import hashlib
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, g, jsonify, request
from flask_caching import Cache
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default: ../faces_prod.db (API slice from ``update_materialized_content``). Production uses
# ``run.sh``: downloads ``DB_URL`` into ``EPSTEIN_SQLITE_PATH``. Override with EPSTEIN_SQLITE_PATH.
_DEFAULT_SQLITE = Path(__file__).resolve().parent.parent / "faces.db"
SQLITE_PATH = os.environ.get("EPSTEIN_SQLITE_PATH", str(_DEFAULT_SQLITE))

# Public URL prefix for best-face crops (13__optimize_node_faces.sh output); no trailing slash.
BEST_FACE_IMAGE_BASE = os.environ.get("BEST_FACE_IMAGE_BASE", "/faces")

# Distinct images per /photos response page (default limit; also the maximum allowed limit).
PHOTO_PAGE_SIZE = 1000

# Materialized (person_id, image_name) pairs for API-eligible images; built by
# epstein-pipeline/scripts/pipeline/update_materialized_content.py
PERSON_ELIGIBLE_IMAGES_TABLE = "person_eligible_images"

# GET /faces: include people by class (sheet ``people.include_in_network`` + name presence).
FACE_CLASSES_DEFAULT = ("named", "unknown")
FACE_CLASSES_ALLOWED = frozenset({"named", "unknown", "ignored", "unidentified"})
FACES_PAGE_SIZE = 1000

cache = Cache(config={"CACHE_TYPE": "flask_caching.backends.filesystem", "CACHE_DIR": "/tmp/flask"})


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
        raise ValueError(f"limit must be between 1 and {PHOTO_PAGE_SIZE} inclusive, got {limit!r}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset!r}")
    return limit, offset


def _parse_faces_pagination_args() -> tuple[int, int]:
    limit_raw = request.args.get("limit", str(FACES_PAGE_SIZE))
    offset_raw = request.args.get("offset", "0")
    limit = int(limit_raw)
    offset = int(offset_raw)
    if limit < 1 or limit > FACES_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {FACES_PAGE_SIZE} inclusive, got {limit!r}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset!r}")
    return limit, offset


def _aggregate_faces_ordered(
    rows: list[sqlite3.Row],
    image_names_order: list[str],
    *,
    google_drive_key_by_jpg: dict[str, str | None] | None = None,
) -> list[dict]:
    """Build photo items in ``image_names_order`` (DB ``image_name`` values, typically .jpg)."""
    order_webp = [n.replace(".jpg", ".webp") for n in image_names_order]
    by_webp: dict[str, dict] = {}
    for w in order_webp:
        by_webp[w] = {"image": w, "faces": []}
    for r in rows:
        webp = r["image_name"].replace(".jpg", ".webp")
        if webp not in by_webp:
            raise RuntimeError(f"Face row image_name not in page list: {r['image_name']!r}")
        ignored = bool(r["ignored"])
        by_webp[webp]["faces"].append(
            {
                "face_id": r["face_id"],
                "person_id": r["person_id"],
                "left": float(r["left"]),
                "top": float(r["top"]),
                "width": float(r["width"]),
                "height": float(r["height"]),
                "ignored": ignored,
            }
        )
    out: list[dict] = []
    for w in order_webp:
        item = by_webp[w]
        if not item["faces"]:
            raise RuntimeError(f"No face rows aggregated for ordered image {w!r}")
        payload: dict = {"image": w, "faces": item["faces"]}
        if google_drive_key_by_jpg is not None:
            jpg = w.replace(".webp", ".jpg")
            payload["google_drive_key"] = _clean_optional_text(google_drive_key_by_jpg.get(jpg))
        out.append(payload)
    return out


def _google_drive_keys_for_image_names(conn: sqlite3.Connection, image_names: list[str]) -> dict[str, str | None]:
    """Return ``image_name`` (as in DB, typically ``*.jpg``) -> optional Drive file id.

    Keys come from ``person_eligible_images.google_drive_key`` (pipeline DB and prod slice), filled
    by ``rebuild_person_eligible_images`` from ``images`` when ``has_face = 1``.
    """
    if not image_names:
        return {}
    cols = [row[1] for row in conn.execute("PRAGMA table_info(person_eligible_images)").fetchall()]
    if not cols:
        raise RuntimeError("person_eligible_images is missing or unreadable; run pipeline materialization.")
    if "google_drive_key" not in cols:
        raise RuntimeError(
            "person_eligible_images.google_drive_key column missing; run "
            "epstein-pipeline/scripts/pipeline/update_materialized_content.py "
            "(rebuild_person_eligible_images) against this SQLite file."
        )
    placeholders = ",".join("?" * len(image_names))
    q = f"""
        SELECT image_name, MAX(google_drive_key) AS google_drive_key
        FROM person_eligible_images
        WHERE image_name IN ({placeholders})
        GROUP BY image_name
    """
    found: dict[str, str | None] = dict.fromkeys(image_names)
    for r in conn.execute(q, tuple(image_names)).fetchall():
        found[str(r["image_name"])] = _clean_optional_text(r["google_drive_key"])
    return found


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


@contextmanager
def get_db_connection():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _require_person_eligible_images_table(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (PERSON_ELIGIBLE_IMAGES_TABLE,),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"Missing table {PERSON_ELIGIBLE_IMAGES_TABLE!r}. Run "
            "epstein-pipeline/scripts/pipeline/update_materialized_content.py "
            "against this database."
        )


def _normalize_photo_name_for_db(raw: str) -> str | None:
    """
    Map a ``photo`` query value to ``image_name`` as stored (always ``*.jpg`` in SQLite).

    The site may pass ``*.webp``; normalize to the DB basename with ``.jpg``.
    """
    s = raw.strip().replace("\\", "/").split("/")[-1].strip()
    if not s:
        return None
    lower = s.lower()
    if lower.endswith(".webp"):
        base = s[:-5]
        if not base:
            return None
        return base + ".jpg"
    if lower.endswith(".jpeg"):
        base = s[:-5]
        if not base:
            return None
        return base + ".jpg"
    if lower.endswith(".jpg"):
        return s
    return s


def _photo_rank_in_person_list(conn: sqlite3.Connection, unique: list[str], target_db_name: str) -> int | None:
    """0-based rank in ``ORDER BY image_name``, or ``None`` if ``target_db_name`` is not in the set."""
    pei = PERSON_ELIGIBLE_IMAGES_TABLE
    n = len(unique)
    branch = f"SELECT image_name FROM {pei} WHERE person_id = ?"
    if n == 1:
        pid = unique[0]
        row = conn.execute(
            f"SELECT 1 FROM {pei} WHERE person_id = ? AND image_name = ? LIMIT 1",
            (pid, target_db_name),
        ).fetchone()
        if not row:
            return None
        r = conn.execute(
            f"SELECT COUNT(*) AS c FROM {pei} WHERE person_id = ? AND image_name < ?",
            (pid, target_db_name),
        ).fetchone()
        return int(r["c"])
    intersect_body = " INTERSECT ".join([branch] * n)
    exists_sql = f"WITH q AS ({intersect_body}) SELECT 1 AS o FROM q WHERE image_name = ? LIMIT 1"
    row = conn.execute(exists_sql, tuple(unique) + (target_db_name,)).fetchone()
    if not row:
        return None
    count_sql = f"WITH q AS ({intersect_body}) SELECT COUNT(*) AS c FROM q WHERE image_name < ?"
    r = conn.execute(count_sql, tuple(unique) + (target_db_name,)).fetchone()
    return int(r["c"])


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _require_images_alias_for_efta_search(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "images")
    if not cols:
        raise RuntimeError("Missing table images (expected on API database for EFTA alias search).")
    if "alias" not in cols:
        raise RuntimeError(
            "images.alias column missing (rebuild faces_prod.db / run pipeline export after add_efta_alias.py)."
        )


def _document_faces_from_where(*, match_efta_alias: bool) -> str:
    """
    Shared FROM/WHERE for document-prefix queries over eligible faces.

    For EFTA prefixes, also match ``images.alias`` (slim table in ``faces_prod.db``), which enables
    searching synthesized IDs like ``EFTA00250540`` for a page image ``EFTA00250488-00052.jpg``.

    Join is on ``images.image_name`` (PK) and is guarded to EFTA-only prefixes so other document
    families keep the old fast path.
    """
    family_clause = """
            (
                f.image_name LIKE 'EFTA%'
                OR f.image_name LIKE 'BIRTHDAY_BOOK_%'
                OR f.image_name LIKE 'HOUSE_OVERSIGHT_%'
            )
    """
    if match_efta_alias:
        return f"""
        FROM faces AS f
        LEFT JOIN images AS i ON i.image_name = f.image_name
        WHERE f.is_eligible = 1
            AND {family_clause}
            AND (
                f.image_name LIKE ?
                OR (i.alias IS NOT NULL AND i.alias LIKE ?)
            )
        """
    return f"""
        FROM faces AS f
        WHERE f.is_eligible = 1
            AND {family_clause}
            AND f.image_name LIKE ?
    """


def _photo_rank_in_document_prefix(conn: sqlite3.Connection, pattern: str, target_db_name: str) -> int | None:
    """0-based rank among distinct ``image_name`` values for the document-prefix query, or ``None``."""
    match_efta_alias = pattern.upper().startswith("EFTA")
    if match_efta_alias:
        _require_images_alias_for_efta_search(conn)
    base_from = _document_faces_from_where(match_efta_alias=match_efta_alias)
    exists_sql = (
        "SELECT 1 FROM (SELECT DISTINCT f.image_name AS image_name "
        + base_from
        + ") AS u WHERE u.image_name = ? LIMIT 1"
    )
    if match_efta_alias:
        row = conn.execute(exists_sql, (pattern, pattern, target_db_name)).fetchone()
    else:
        row = conn.execute(exists_sql, (pattern, target_db_name)).fetchone()
    if not row:
        return None
    count_sql = (
        "SELECT COUNT(*) AS c FROM (SELECT DISTINCT f.image_name AS image_name "
        + base_from
        + " AND f.image_name < ?) AS t"
    )
    if match_efta_alias:
        r = conn.execute(count_sql, (pattern, pattern, target_db_name)).fetchone()
    else:
        r = conn.execute(count_sql, (pattern, target_db_name)).fetchone()
    return int(r["c"])


def resolve_effective_photo_offset(
    photo_raw: str | None,
    limit: int,
    request_offset: int,
    *,
    person_ids: list[str] | None = None,
    document_prefix_norm: str | None = None,
) -> int:
    """
    If ``photo`` is set and names a qualifying image, return the page offset (bucket start)
    that contains it. Otherwise return ``request_offset`` (clamped to non-negative).
    """
    req = max(0, request_offset)
    if not photo_raw or not str(photo_raw).strip():
        return req
    target = _normalize_photo_name_for_db(str(photo_raw))
    if not target:
        return req
    with get_db_connection() as conn:
        if person_ids is not None:
            unique = sorted(set(p for p in person_ids if p and str(p).strip()))
            if not unique:
                return req
            try:
                _require_person_eligible_images_table(conn)
            except RuntimeError:
                return req
            rank = _photo_rank_in_person_list(conn, unique, target)
        elif document_prefix_norm is not None:
            pattern = document_prefix_norm + "%"
            rank = _photo_rank_in_document_prefix(conn, pattern, target)
        else:
            return req
    if rank is None:
        return req
    return (rank // limit) * limit


def _get_photos_limit_and_effective_offset() -> tuple[int, int]:
    """
    Parse request args and return ``(limit, effective_offset)`` for the /photos query.

    Raises ``ValueError`` with a client-facing message on bad requests (including pagination).
    """
    limit, req_off = _parse_pagination_args()
    doc_raw = request.args.get("document_prefix", "").strip()
    raw = request.args.get("person_ids", "").strip()
    photo_raw = request.args.get("photo", "").strip()
    if doc_raw and raw:
        raise ValueError("Use either document_prefix or person_ids, not both.")
    if doc_raw:
        normalized = _normalize_document_prefix(doc_raw)
        if normalized is None:
            raise ValueError(
                "Invalid document_prefix: use a prefix of EFTA, BIRTHDAY_BOOK_, or HOUSE_OVERSIGHT_ "
                "followed by digits (case-insensitive). Extensions and hyphens in pasted file names "
                "are stripped, e.g. EFTA00000005, EFTA00000005.pdf, EFTA00000005-00006.webp."
            )
        eff = resolve_effective_photo_offset(photo_raw, limit, req_off, document_prefix_norm=normalized)
        return limit, eff
    if not raw:
        raise ValueError("Missing query: pass person_ids (comma-separated) or document_prefix (not both).")
    person_ids = [p.strip() for p in raw.split(",") if p.strip()]
    if not person_ids:
        raise ValueError("person_ids must contain at least one id.")
    eff = resolve_effective_photo_offset(photo_raw, limit, req_off, person_ids=person_ids)
    return limit, eff


def make_photos_cache_key() -> str:
    """
    Cache key depends only on resolved pagination (``person_ids`` or ``document_prefix`` +
    ``limit`` + effective offset), not on raw ``photo``, so ``?photo=`` and ``?offset=`` that
    resolve to the same page share an entry.
    """
    try:
        limit, eff = _get_photos_limit_and_effective_offset()
    except ValueError:
        return f"photos/e/{hashlib.md5(request.query_string).hexdigest()}"

    g.photos_limit = limit
    g.photos_effective_offset = eff
    doc_raw = request.args.get("document_prefix", "").strip()
    if doc_raw:
        normalized = _normalize_document_prefix(doc_raw)
        assert normalized is not None
        return f"v2/photos/doc/{normalized}/{limit}/{eff}"
    raw = request.args.get("person_ids", "").strip()
    person_ids = sorted([p.strip() for p in raw.split(",") if p.strip()])
    key_ids = ",".join(person_ids)
    return f"v2/photos/ppl/{key_ids}/{limit}/{eff}"


def _photos_cache_only_ok(rv: object) -> bool:
    if isinstance(rv, tuple) and len(rv) >= 2:
        return rv[1] == 200
    sc = getattr(rv, "status_code", 200)
    return sc == 200


@app.route("/")
def description():
    return {
        "message": (
            "Welcome to the API of Epstein.Photos, which maps out the network "
            "of connections between people in the Epstein Library."
        )
    }


def photos_for_all_person_ids(person_ids: list[str], *, limit: int, offset: int) -> tuple[list[dict], int]:
    """
    Return one object per image: each image contains every requested person_id at least once.

    ``faces`` is a list of every face row in that image (all ``person_id`` values), including
    multiple boxes for the same person when they appear more than once. Each element is::

        {"face_id": str, "person_id": str, "left", "top", "width", "height"}

    Coordinates are Rekognition-normalized (0..1). Matches ``normalizeFaceEntries`` in
    ``site/pages/search-inner.html``.

    Paginates distinct ``image_name`` values; returns ``(page_items, total_count)``.

    Qualifying image names come from the ``PERSON_ELIGIBLE_IMAGES_TABLE`` materialization
    (rebuilt by the pipeline), using intersection across selected people when
    ``len(person_ids) > 1``.
    """
    unique = sorted(set(p for p in person_ids if p and str(p).strip()))
    n = len(unique)
    if n == 0:
        return [], 0

    pei = PERSON_ELIGIBLE_IMAGES_TABLE
    branch = f"SELECT image_name FROM {pei} WHERE person_id = ?"

    with get_db_connection() as conn:
        _require_person_eligible_images_table(conn)

        if n == 1:
            pid = unique[0]
            count_sql = f"SELECT COUNT(*) AS c FROM {pei} WHERE person_id = ?"
            names_sql = f"SELECT image_name FROM {pei} WHERE person_id = ? ORDER BY image_name LIMIT ? OFFSET ?"
            total = int(conn.execute(count_sql, (pid,)).fetchone()["c"])
            name_rows = conn.execute(names_sql, (pid, limit, offset)).fetchall()
        else:
            intersect_body = " INTERSECT ".join([branch] * n)
            count_sql = f"WITH q AS ({intersect_body}) SELECT COUNT(*) AS c FROM q"
            names_sql = f"WITH q AS ({intersect_body}) SELECT image_name FROM q ORDER BY image_name LIMIT ? OFFSET ?"
            params = tuple(unique)
            total = int(conn.execute(count_sql, params).fetchone()["c"])
            name_rows = conn.execute(names_sql, params + (limit, offset)).fetchall()

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
                f.height,
                CASE WHEN COALESCE(p.include_in_network, 0) = -1 THEN 1 ELSE 0 END AS ignored
            FROM faces AS f
            LEFT JOIN people AS p ON p.person_id = f.person_id
            WHERE f.image_name IN ({placeholders_names})
                AND f.is_eligible = 1
            ORDER BY f.image_name, f.person_id, f.face_id;
        """
        rows = conn.execute(faces_sql, tuple(names)).fetchall()
        drive = _google_drive_keys_for_image_names(conn, names)

    return _aggregate_faces_ordered(rows, names, google_drive_key_by_jpg=drive), total


def person_metadata_for_person_id(person_id: str) -> dict | None:
    """
    Return metadata for one person_id from ``people``, or ``None`` when not found.

    Victims (``is_victim = 1``) never get metadata — same as not found — so the UI does not
    expose links or reference URLs for anonymized victim pages.

    Used by ``/photos`` when exactly one person_id is searched so UI can render
    single-person context links without additional API round-trips.
    """
    pid = str(person_id or "").strip()
    if not pid:
        return None
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT person_id, name, link, jmail_link, reference_image_url
            FROM people
            WHERE person_id = ?
              AND COALESCE(is_victim, 0) = 0
            LIMIT 1
            """,
            (pid,),
        ).fetchone()
    if row is None:
        return None

    return {
        "person_id": str(row["person_id"]),
        "name": _clean_optional_text(row["name"]),
        "link": _clean_optional_text(row["link"]),
        "jmail_link": _clean_optional_text(row["jmail_link"]),
        "reference_image_url": _clean_optional_text(row["reference_image_url"]),
    }


DOCUMENT_ID_PREFIXES = ("EFTA", "BIRTHDAY_BOOK_", "HOUSE_OVERSIGHT_")


def _strip_document_search_term(raw: str) -> str:
    """
    Same rules as ``SiteShared.stripDocumentSearchTerm``: remove extensions, hyphens, and periods
    used in pasted file names so ``LIKE prefix%`` matches DB ``image_name`` stems.
    """
    t = raw.strip()
    if not t:
        return ""
    u = t.upper()
    for p in DOCUMENT_ID_PREFIXES:
        if u.startswith(p):
            rest = u[len(p) :]
            rest = re.sub(r"\.[A-Za-z]*$", "", rest)
            rest = re.sub(r"[-.]", "", rest)
            return p + rest
    u = re.sub(r"\.(PDF|WEBP|JPEG|JPG)$", "", u, flags=re.IGNORECASE)
    return re.sub(r"[-.]", "", u)


def _is_valid_partial_document_query_normalized(u: str) -> bool:
    t = u.strip()
    if not t:
        return True
    up = t.upper()
    for p in DOCUMENT_ID_PREFIXES:
        if p.startswith(up):
            return True
        if up.startswith(p):
            rest = up[len(p) :]
            if rest == "" or rest.isdigit():
                return True
            return False
    return False


def is_valid_partial_document_query(s: str) -> bool:
    """
    True if ``s`` is empty, or could still become ``PREFIX`` + digits for one of
    ``DOCUMENT_ID_PREFIXES`` (case-insensitive). Mirrors ``epstein-web/site/js/shared.js``.
    """
    t = s.strip()
    if not t:
        return True
    st = _strip_document_search_term(t)
    if _is_valid_partial_document_query_normalized(st):
        return True
    u = t.upper()
    for p in DOCUMENT_ID_PREFIXES:
        if p.startswith(u):
            return True
        if u.startswith(p):
            rest = u[len(p) :]
            if rest == "" or rest.isdigit():
                return True
            if re.fullmatch(r"(\d+)(-\d+)*(\.[A-Z]{0,4})?", rest):
                return True
            return False
    return False


def _normalize_document_prefix(raw: str) -> str | None:
    """
    Non-empty, valid query → uppercase stem for ``LIKE prefix%`` (extensions / hyphens / periods stripped).
    """
    p = _strip_document_search_term(raw)
    if not p:
        return None
    if not _is_valid_partial_document_query_normalized(p):
        return None
    return p


def photos_for_document_prefix(prefix: str, *, limit: int, offset: int) -> tuple[list[dict], int]:
    """
    Return one object per image whose ``image_name`` starts with ``prefix`` (same filters as
    ``photos_for_all_person_ids``: non-duplicate, non-explicit, no victim images).

    Paginates distinct ``image_name`` values; returns ``(page_items, total_count)``.
    """
    unique_prefix = _normalize_document_prefix(prefix)
    if unique_prefix is None:
        return [], 0

    pattern = unique_prefix + "%"
    match_efta_alias = unique_prefix.startswith("EFTA")

    base_from = _document_faces_from_where(match_efta_alias=match_efta_alias)
    count_sql = "SELECT COUNT(DISTINCT f.image_name) AS c " + base_from
    names_sql = "SELECT DISTINCT f.image_name " + base_from + " ORDER BY f.image_name LIMIT ? OFFSET ?"

    with get_db_connection() as conn:
        if match_efta_alias:
            _require_images_alias_for_efta_search(conn)
            total = int(conn.execute(count_sql, (pattern, pattern)).fetchone()["c"])
            name_rows = conn.execute(names_sql, (pattern, pattern, limit, offset)).fetchall()
        else:
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
                f.height,
                CASE WHEN COALESCE(p.include_in_network, 0) = -1 THEN 1 ELSE 0 END AS ignored
            FROM faces AS f
            LEFT JOIN people AS p ON p.person_id = f.person_id
            WHERE f.image_name IN ({placeholders})
                AND f.is_eligible = 1
            ORDER BY f.image_name, f.person_id, f.face_id;
        """
        rows = conn.execute(faces_sql, tuple(names)).fetchall()
        drive = _google_drive_keys_for_image_names(conn, names)

    return _aggregate_faces_ordered(rows, names, google_drive_key_by_jpg=drive), total


def _sanitize_label_for_filename(label: str) -> str:
    """Match ``scripts/12__export_node_faces.py`` filename stems."""
    s = "".join(c if c.isalnum() or c in "._- " else "" for c in label)
    return s.strip().replace(" ", "_").replace("/", "_") or "node"


def parse_face_classes_query(raw: str) -> list[str]:
    """
    Parse a non-empty ``classes`` query value for ``GET /faces``.

    Comma-separated tokens: ``named``, ``unknown``, ``ignored``, ``unidentified``.
    """
    parts = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("classes must list at least one token when non-empty.")
    unknown_tokens = [p for p in parts if p not in FACE_CLASSES_ALLOWED]
    if unknown_tokens:
        raise ValueError(
            f"Invalid classes token(s): {unknown_tokens!r}. Allowed: {', '.join(sorted(FACE_CLASSES_ALLOWED))}."
        )
    return sorted(set(parts))


def resolve_face_classes_from_request() -> list[str]:
    """
    Classes list for ``GET /faces``.

    - Parameter **absent** → default ``named`` + ``unknown`` (historical /people behavior).
    - ``classes=`` present but empty → **no** classes (empty result set).
    - Otherwise parse comma-separated tokens.
    """
    if "classes" not in request.args:
        return list(FACE_CLASSES_DEFAULT)
    raw = request.args.get("classes", "")
    if not str(raw).strip():
        return []
    return parse_face_classes_query(raw)


def _faces_class_where_fragment(classes: list[str]) -> str:
    """Return ``( ... OR ... )`` SQL for the selected class tokens."""
    if not classes:
        return "(1 = 0)"
    parts: list[str] = []
    cs = set(classes)
    if "named" in cs:
        parts.append("(COALESCE(p.include_in_network, 0) = 1 AND TRIM(COALESCE(p.name, '')) != '')")
    if "unknown" in cs:
        parts.append("(COALESCE(p.include_in_network, 0) = 1 AND TRIM(COALESCE(p.name, '')) = '')")
    if "ignored" in cs:
        parts.append("(COALESCE(p.include_in_network, 0) = -1)")
    if "unidentified" in cs:
        parts.append("(COALESCE(p.include_in_network, 0) = 0)")
    if not parts:
        return "(1 = 0)"
    return "(" + " OR ".join(parts) + ")"


def faces_list(
    classes: list[str] | None = None, *, limit: int = FACES_PAGE_SIZE, offset: int = 0
) -> tuple[list[dict], int]:
    """
    People rows for the People gallery, with photo counts and optional best-face image URL.

    ``classes`` selects which ``people.include_in_network`` + name buckets to include
    (see ``parse_face_classes_query``). Victims (``is_victim = 1``) are always excluded.

    ``photo_count`` comes from ``people.valid_distinct_image_count`` (recomputed by pipeline).
    People are included only when ``people.has_non_minor_eligible_face = 1`` when that materialized
    column exists on this DB.

    Sort: (1) has ``best_face_id``, (2) has non-empty ``name``, (3) ``photo_count`` (desc),
    then ``person_id``.
    """
    if classes is None:
        classes = list(FACE_CLASSES_DEFAULT)
    class_sql = _faces_class_where_fragment(classes)
    with get_db_connection() as conn:
        people_cols = _table_columns(conn, "people")
        has_non_minor_filter = "has_non_minor_eligible_face" in people_cols
        non_minor_sql = (
            "AND COALESCE(p.has_non_minor_eligible_face, 0) = 1" if has_non_minor_filter else ""
        )
        sql = f"""
        SELECT
            p.person_id,
            p.name,
            p.best_face_id,
            p.valid_distinct_image_count AS photo_count
        FROM people AS p
        WHERE COALESCE(p.is_victim, 0) = 0
            AND {class_sql}
            {non_minor_sql}
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
        LIMIT ? OFFSET ?
    """
        count_sql = f"""
        SELECT COUNT(*) AS c
        FROM people AS p
        WHERE COALESCE(p.is_victim, 0) = 0
            AND {class_sql}
            {non_minor_sql}
    """
        total = int(conn.execute(count_sql).fetchone()["c"])
        rows = conn.execute(sql, (limit, offset)).fetchall()

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
    return out, total


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


def make_faces_cache_key() -> str:
    """Cache ``GET /faces`` per normalized ``classes`` + pagination args."""
    try:
        cls = resolve_face_classes_from_request()
    except ValueError:
        return "v2/faces/__invalid__"
    try:
        limit, offset = _parse_faces_pagination_args()
    except ValueError:
        return "v2/faces/__invalid_pagination__"
    if not cls:
        return f"v2/faces/__empty__/{limit}/{offset}"
    return "v2/faces/" + ",".join(cls) + f"/{limit}/{offset}"


@app.route("/faces")
@cache.cached(timeout=14400, make_cache_key=make_faces_cache_key)
def get_faces():
    """
    List of people with ``person_id``, ``image`` (best-face WebP URL or null),
    ``photo_count`` (distinct images in the public API sense), and ``name`` (or null).

    Non-victim rows only.     Optional query: ``classes`` — comma-separated list of which groups to **include**
    ``named``, ``unknown``, ``ignored``, ``unidentified``. Default: ``named,unknown``
    (in-network with and without names). Adding classes **widens** the result set versus default.

    Optional pagination: ``limit`` (default ``FACES_PAGE_SIZE``; max same) and ``offset`` (default 0).

    Order: best_face set, has name, higher ``photo_count``, then ``person_id``.
    """
    try:
        classes = resolve_face_classes_from_request()
        limit, offset = _parse_faces_pagination_args()
    except ValueError as e:
        return jsonify(error="Bad Request", message=str(e)), 400
    data, total = faces_list(classes, limit=limit, offset=offset)
    return jsonify(data=data, total=total, limit=limit, offset=offset)


@app.route("/photos")
@cache.cached(
    timeout=14400,
    make_cache_key=make_photos_cache_key,
    response_filter=_photos_cache_only_ok,
)
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
    ``offset`` (default 0) apply to distinct images. The response includes ``total`` (full match
    count), ``limit``, and ``offset`` (the effective offset used for this page).

    Optional ``photo`` — filename (or basename) of an image in the result set. When set, and the
    image qualifies for the query, pagination is adjusted so the returned page is the chunk
    (by ``limit``) that contains that image. If the image is not in the set, ``offset`` from the
    request is used (default 0). ``photo`` takes precedence over ``offset`` when the image is
    found.

    Mutually exclusive: pass **either** ``document_prefix`` **or** ``person_ids``, not both.
    """
    try:
        if hasattr(g, "photos_effective_offset"):
            limit, offset = g.photos_limit, g.photos_effective_offset
        else:
            limit, offset = _get_photos_limit_and_effective_offset()
    except ValueError as e:
        return jsonify(error="Bad Request", message=str(e)), 400

    doc_raw = request.args.get("document_prefix", "").strip()
    raw = request.args.get("person_ids", "").strip()
    if doc_raw:
        normalized = _normalize_document_prefix(doc_raw)
        if normalized is None:
            return jsonify(
                error="Bad Request",
                message=(
                    "Invalid document_prefix: use a prefix of EFTA, BIRTHDAY_BOOK_, or HOUSE_OVERSIGHT_ "
                    "followed by digits (case-insensitive). Extensions and hyphens in pasted file names "
                    "are stripped, e.g. EFTA00000005, EFTA00000005.pdf, EFTA00000005-00006.webp."
                ),
            ), 400
        data, total = photos_for_document_prefix(normalized, limit=limit, offset=offset)
        return jsonify(data=data, total=total, limit=limit, offset=offset)

    person_ids = [p.strip() for p in raw.split(",") if p.strip()]
    data, total = photos_for_all_person_ids(person_ids, limit=limit, offset=offset)
    out = {"data": data, "total": total, "limit": limit, "offset": offset}
    unique_person_ids = sorted(set(person_ids))
    if len(unique_person_ids) == 1:
        out["person_metadata"] = person_metadata_for_person_id(unique_person_ids[0])
    return jsonify(out)


def start():
    if not Path(SQLITE_PATH).is_file():
        logger.warning("SQLite file not found at %s — API will error on first request.", SQLITE_PATH)
    app.run()


if __name__ == "__main__":
    start()
