import io
import pytest
from fastapi.testclient import TestClient
from main import app
from fpdf import FPDF # type: ignore
import respx
from httpx import Response

@pytest.fixture
def client():
    return TestClient(app)

stubbed_text = """This document outlines the design and implementation details of the Acme Microservice Framework.\n\nThe system follows a modular microservices architecture, comprising of independent services that communicate via defined APIs. Each service is containerized using Docker and deployed using Kubernetes.\n\nSupported communication protocols include gRPC and REST/JSON."""

@pytest.fixture
def create_test_pdf_fpdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=stubbed_text, ln=1, align="L")
        
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return io.BytesIO(pdf_bytes)

class TestMain:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World!"}

    def test_parse_doc_with_valid_pdf(self, client, create_test_pdf_fpdf):
        buffer = create_test_pdf_fpdf
        files = {"file": ("test.pdf", buffer, "application/pdf")}

        # Mock OpenAI API call
        with respx.mock:
            respx.post("https://api.openai.com/v1/chat/completions").mock(
                return_value=Response(200, json={
                    "choices": [{
                        "message": {
                            "tool_calls": [{
                                "function": {
                                    "arguments": '{"introduction": "This document outlines the design and implementation details of the Acme Microservice Framework.", "architecture_overview": "The system follows a modular microservices architecture, comprising of independent services that communicate via defined APIs. Each service is containerized using Docker and deployed using Kubernetes.", "communication_protocols": ["gRPC", "REST/JSON"]}'
                                }
                            }]
                        }
                    }]
                })
            )
            response = client.post("/parse_doc", files=files)
            assert response.status_code == 200
            parsed = response.json()["parsed_data"]
            assert parsed["introduction"] == "This document outlines the design and implementation details of the Acme Microservice Framework."
            assert parsed["architecture_overview"].startswith("The system follows a modular microservices architecture")
            assert parsed["communication_protocols"] == ["gRPC", "REST/JSON"]
