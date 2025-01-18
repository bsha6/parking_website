import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"license_types_list" in response.data  # Check if the template includes the license types

def test_how_it_works(client):
    response = client.get("/how_it_works")
    assert response.status_code == 200

def test_contact_page(client):
    response = client.get("/contact")
    assert response.status_code == 200

def test_next_steps(client):
    response = client.get("/next_steps")
    assert response.status_code == 200