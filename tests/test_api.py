from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_upload_endpoint():
    response = client.post(
        "/upload/",
        files={"file": ("test.csv", b"id,nome\n1,Ana", "text/csv")}
    )
    assert response.status_code == 200
    assert "rows" in response.json()
