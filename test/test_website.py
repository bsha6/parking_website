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

# TODO: add test for model recommended dispute
def test_submission(client):
    # Define input data for the test
    input_data = {
        "fine-amount": 65,
        "license_type": "COM",
        "vehicle_body_type": "4DSD",
        "vehicle-color": "WHITE",
        "vehicle-make": "DODGE",
        "violation_code": 23,
    }

    # Expected phrases in the output
    expected_phrases = [
        "You should dispute!",
        "Specifically, the model predicts a 75.4% chance of your appeal being granted",
        "15.9% chance of your fine being reduced",
        "8.6% chance of your appeal being denied"
    ]

    # Make a POST request to the /submission endpoint
    response = client.post("/submission", data=input_data, content_type="application/x-www-form-urlencoded")
    print(response.data.decode('utf-8'))

    # Assertions
    # assert response.status_code == 200  # Ensure the request was successful
    response_data = response.data.decode('utf-8')  # Decode response data for assertions
    print(response_data)

    # Check that each expected phrase is in the response
    for phrase in expected_phrases:
        assert phrase in response_data

# TODO: add test for model recommended not to dispute