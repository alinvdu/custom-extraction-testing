from typing import Annotated, List, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File
import PyPDF2
import io
import os
from llm_service.LLMService import LLMService
from dotenv import load_dotenv
import httpx
import msgspec
import logging
load_dotenv()

app = FastAPI()

llmService = LLMService(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@app.get("/")
async def root() -> dict:
    return {"message": "Hello World!"}

@app.post("/parse_doc")
async def parse_doc(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Malformed file uploaded.")
    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted for now!")
    file_content = await file.read()
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="File is not a valid PDF or is corrupted."
        )
    
    try:
        class DocumentExtraction(msgspec.Struct):
            introduction: Annotated[str, "An introduction extracted from the document"]
            architecture_overview: Annotated[str, "The architecture overview"]
            communication_protocols: List[Literal["gRPC", "https", "REST/JSON"]]

            def to_dict(self):
                return {f: getattr(self, f) for f in self.__struct_fields__}

        document_extraction: DocumentExtraction = await llmService.extraction(text, extraction_interface=DocumentExtraction)
    except httpx.HTTPStatusError as e:
        error_detail = f"LLM service returned an error: {e.response.text}"
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to LLM service: {e}")
    except Exception as e:
        logger.exception("Unexpected error during document extraction.")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    return {
        "parsed_data": document_extraction.to_dict()
    }
