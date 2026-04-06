import pytest

from api import (
    app,
)

@pytest.fixture(scope="module")
def client():
    with app.test_client() as client:
        yield client

def test_description(client):
    response = client.get("/")
    assert b"Library" in response.data

def test_photos_single(client):
    response = client.get("/photos?person_ids=person_49")
    assert response.status_code == 200
    body = response.json
    assert body["limit"] == 1000
    assert body["offset"] == 0
    assert len(body["data"]) == 338
    assert body["total"] == 338
    assert "EFTA00254398-00069.webp" in [image["image"] for image in body["data"]]

def test_photos_double(client):
    response = client.get("/photos?person_ids=person_49,person_1958")
    assert response.status_code == 200
    body = response.json
    assert body["total"] == len(body["data"])
    assert body["total"] == 4
    assert "EFTA00249026-00027.webp" in [image["image"] for image in body["data"]]


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

def test_people(client):
    response = client.get("/people")
    assert response.status_code == 200
    assert response.json["data"]["person_49"] == "Jean-Luc Brunel"
