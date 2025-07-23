# tests/test_llm_real_call.py
import pytest
import os
from llm_service.LLMService import LLMService
import msgspec
from typing import Annotated, Literal, List
import PyPDF2
import io
from dotenv import load_dotenv

load_dotenv()

class DocumentExtraction(msgspec.Struct):
    introduction: Annotated[str, "An introduction extracted from the document"]
    architecture_overview: Annotated[str, "The architecture overview"]
    communication_protocols: List[Literal["gRPC", "https", "REST/JSON"]]

@pytest.mark.asyncio
async def test_llm_extraction_real_call():
    # Load real PDF
    pdf_path = "sample_document.pdf"
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(f.read()))
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

    # Call real LLM
    api_key = os.getenv("OPENAI_API_KEY")
    llm = LLMService(model="gpt-4o-mini", api_key=api_key)

    result = await llm.extraction(file_content=text, extraction_interface=DocumentExtraction)
    assert isinstance(result, DocumentExtraction)
    assert result.introduction  # not empty
    assert all(p in ['gRPC', 'https', 'REST/JSON'] for p in result.communication_protocols)
