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
    assert len(response.json["data"]) == 332
    assert "EFTA00254398-00069.webp" in [image["image"] for image in response.json["data"]]

def test_photos_double(client):
    response = client.get("/photos?person_ids=person_49,person_1958")
    assert response.status_code == 200
    assert len(response.json["data"]) == 4
    assert "EFTA00249026-00027.webp" in [image["image"] for image in response.json["data"]]

def test_people(client):
    response = client.get("/people")
    assert response.status_code == 200
    assert response.json["data"]["person_49"] == "Jean-Luc Brunel"
