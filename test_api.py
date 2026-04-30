import sqlite3

import pytest

from api import (
    SQLITE_PATH,
    app,
    person_metadata_for_person_id,
)


def _first_victim_person_id() -> str | None:
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        row = conn.execute(
            "SELECT person_id FROM people WHERE COALESCE(is_victim, 0) = 1 LIMIT 1"
        ).fetchone()
        return str(row[0]) if row else None
    finally:
        conn.close()


def _person_eligible_images_present() -> bool:
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='person_eligible_images'"
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _has_non_minor_eligible_face(person_id: str) -> bool:
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM faces
            WHERE person_id = ?
              AND is_eligible = 1
              AND NOT (
                    age_range_low IS NOT NULL
                    AND age_range_high IS NOT NULL
                    AND (age_range_low < 18 OR age_range_high < 18)
              )
            LIMIT 1
            """,
            (person_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


_PHOTOS_SKIP = pytest.mark.skipif(
    not _person_eligible_images_present(),
    reason=(
        "Requires person_eligible_images table; run epstein-pipeline/scripts/pipeline/update_materialized_content.py"
    ),
)

_VICTIM_SKIP = pytest.mark.skipif(
    _first_victim_person_id() is None,
    reason="No is_victim=1 row in people",
)

IMAGE_COUNT = 364


@pytest.fixture(scope="module")
def client():
    with app.test_client() as client:
        yield client


def test_description(client):
    response = client.get("/")
    assert b"Library" in response.data


@_PHOTOS_SKIP
def test_photos_single(client):
    response = client.get("/photos?person_ids=person_49")
    assert response.status_code == 200
    body = response.json
    assert body["limit"] == 1000
    assert body["offset"] == 0
    assert len(body["data"]) == IMAGE_COUNT
    assert body["total"] == IMAGE_COUNT
    assert "EFTA00254398-00069.webp" in [image["image"] for image in body["data"]]


@_PHOTOS_SKIP
def test_photos_double(client):
    response = client.get("/photos?person_ids=person_49,person_1958")
    assert response.status_code == 200
    body = response.json
    assert body["total"] == len(body["data"])
    assert body["total"] == 4
    assert "EFTA00249026-00027.webp" in [image["image"] for image in body["data"]]


@_PHOTOS_SKIP
def test_photos_pagination_offset(client):
    r0 = client.get("/photos?person_ids=person_49&limit=5&offset=0")
    r1 = client.get("/photos?person_ids=person_49&limit=5&offset=5")
    assert r0.status_code == 200 and r1.status_code == 200
    a = r0.json
    b = r1.json
    assert a["total"] == b["total"]
    assert len(a["data"]) == 5
    assert len(b["data"]) == 5
    names0 = [x["image"] for x in a["data"]]
    names1 = [x["image"] for x in b["data"]]
    assert set(names0).isdisjoint(names1)


@_VICTIM_SKIP
def test_person_metadata_omits_victims():
    pid = _first_victim_person_id()
    assert pid is not None
    assert person_metadata_for_person_id(pid) is None


@_PHOTOS_SKIP
@_VICTIM_SKIP
def test_photos_single_victim_has_no_person_metadata(client):
    pid = _first_victim_person_id()
    response = client.get(f"/photos?person_ids={pid}")
    assert response.status_code == 200
    assert response.json.get("person_metadata") is None


def test_people(client):
    response = client.get("/people")
    assert response.status_code == 200
    assert response.json["data"]["person_49"] == "Jean-Luc Brunel"


def test_faces_default_matches_named_unknown_explicit(client):
    """Omitting ``classes`` defaults to named + unknown — same as ``?classes=named,unknown``."""
    r0 = client.get("/faces")
    r1 = client.get("/faces?classes=named,unknown")
    assert r0.status_code == 200 and r1.status_code == 200
    assert r0.json["data"] == r1.json["data"]


def test_faces_invalid_classes(client):
    r = client.get("/faces?classes=named,invalid_token")
    assert r.status_code == 400
    assert "Invalid classes token" in r.json["message"]


def test_faces_empty_classes(client):
    r = client.get("/faces?classes=")
    assert r.status_code == 200
    assert r.json["data"] == []


def test_faces_pagination_offset(client):
    r0 = client.get("/faces?limit=5&offset=0")
    r1 = client.get("/faces?limit=5&offset=5")
    assert r0.status_code == 200 and r1.status_code == 200
    a = r0.json
    b = r1.json
    assert a["limit"] == 5 and b["limit"] == 5
    assert a["offset"] == 0 and b["offset"] == 5
    assert isinstance(a["total"], int) and isinstance(b["total"], int)
    assert a["total"] == b["total"]
    assert len(a["data"]) == 5
    assert len(b["data"]) == 5
    ids0 = [x["person_id"] for x in a["data"]]
    ids1 = [x["person_id"] for x in b["data"]]
    assert set(ids0).isdisjoint(ids1)


def test_faces_rows_have_non_minor_eligible_face(client):
    r = client.get("/faces?classes=named,unknown,ignored,unidentified&limit=1000&offset=0")
    assert r.status_code == 200
    rows = r.json["data"]
    assert rows
    for row in rows:
        assert _has_non_minor_eligible_face(str(row["person_id"]))
