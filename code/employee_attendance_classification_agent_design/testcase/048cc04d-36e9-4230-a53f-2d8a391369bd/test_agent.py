
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Assume the FastAPI app is defined in attendance_app.py as 'app'
# and AttendanceInput/AttendanceOutput are Pydantic models in attendance_app.py
# If not, adjust the import paths accordingly.
try:
    from attendance_app import app
except ImportError:
    # If the app is not available, define a dummy app for test discovery
    from fastapi import FastAPI
    app = FastAPI()

@pytest.fixture(scope="module")
def client():
    """
    Fixture to provide a FastAPI test client for functional API tests.
    """
    return TestClient(app)

@pytest.fixture
def valid_attendance_input():
    """
    Fixture to provide a valid attendance input payload.
    """
    return {
        "employee_id": "E001",
        "checkin_time": "09:00",
        "shift_start": "09:00",
        "leave_status": "None",
        "holiday_status": "None",
        "date": "2024-06-01"
    }

def test_classify_attendance_present_functional(client, valid_attendance_input):
    """
    Functional test: Validates that the /classify endpoint correctly classifies an employee as Present
    when check-in is on time, no leave, and not a holiday.
    """
    # Patch any database calls within the endpoint to avoid real DB access
    with patch("attendance_app.get_db_session") as mock_db:
        mock_db.return_value = MagicMock()
        response = client.post("/classify", json=valid_attendance_input)
    assert response.status_code == 200, "Expected HTTP 200 for valid input"
    data = response.json()
    assert data.get("success") is True, "Expected success=True in response"
    assert data.get("attendance_status") == "Present", "Expected attendance_status='Present'"
    assert "Present" in data.get("message", ""), "Expected message to contain 'Present'"

def test_classify_attendance_present_functional_db_unavailable(client, valid_attendance_input):
    """
    Functional test: Simulates database unavailable scenario for /classify endpoint.
    """
    # Simulate DB connection error
    with patch("attendance_app.get_db_session", side_effect=Exception("DB unavailable")):
        response = client.post("/classify", json=valid_attendance_input)
    # Depending on implementation, could be 500 or custom error code
    assert response.status_code in (500, 503), "Expected HTTP 500 or 503 when DB is unavailable"
    data = response.json()
    assert data.get("success") is False or "error" in data, "Expected failure response when DB is unavailable"

@pytest.mark.parametrize(
    "malformed_payload",
    [
        {},  # Completely empty
        {"employee_id": "E001"},  # Missing required fields
        {"employee_id": "E001", "checkin_time": "09:00"},  # Partially missing
        {"employee_id": None, "checkin_time": "09:00", "shift_start": "09:00", "leave_status": "None", "holiday_status": "None", "date": "2024-06-01"},  # Null employee_id
        {"employee_id": "E001", "checkin_time": "not-a-time", "shift_start": "09:00", "leave_status": "None", "holiday_status": "None", "date": "2024-06-01"},  # Invalid time format
    ]
)
def test_classify_attendance_present_functional_malformed_input(client, malformed_payload):
    """
    Functional test: Simulates malformed input data for /classify endpoint.
    """
    response = client.post("/classify", json=malformed_payload)
    # FastAPI returns 422 for validation errors
    assert response.status_code == 422, "Expected HTTP 422 for malformed input"
    data = response.json()
    assert "detail" in data, "Expected validation error details in response"

