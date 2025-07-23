# tests/test_llm_trulens_judge.py

import os
import io
import json
import pytest
import PyPDF2
import msgspec
import time

from typing import Annotated, Literal, List
from dotenv import load_dotenv

from llm_service.LLMService import LLMService

from trulens.providers.openai import OpenAI as FeedbackOpenAI
from trulens.core import Feedback, TruSession
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument

load_dotenv()

# --- Schema definition
class DocumentExtraction(msgspec.Struct):
    introduction: Annotated[str, "An introduction extracted from the document"]
    architecture_overview: Annotated[str, "The architecture overview"]
    communication_protocols: List[Literal["gRPC", "https", "REST/JSON"]]

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}


# --- Wrapper for TruApp
class ExtractorWrapper:
    def __init__(self):
        self.llm = LLMService(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    @instrument()
    def extract(self, text: str) -> str:
        import asyncio
        result: DocumentExtraction = asyncio.run(
            self.llm.extraction(file_content=text, extraction_interface=DocumentExtraction)
        )
        return json.dumps(result.to_dict())


# --- Test using TruApp correctly
def test_llm_extraction_with_relevance_judge_and_leaderboard():
    # Query leaderboard
    session = TruSession()
    # Load the PDF
    pdf_path = "./sample_document.pdf"
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(f.read()))
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

    # Feedback setup
    provider = FeedbackOpenAI()
    f_relevance = Feedback(provider.relevance_with_cot_reasons, name="Relevance").on_input().on_output()

    # Wrap the app
    wrapper = ExtractorWrapper()
    tru_app = TruApp(
        wrapper,
        app_id="llm-doc-parser",
        feedbacks=[f_relevance]
    )

    # Run with recording enabled
    with tru_app as recording:
        output = wrapper.extract(text)
        print('output is', output)

    time.sleep(2)

    leaderboard = session.get_leaderboard()
    print('leaderboard is', leaderboard)
    # Inspect last record
    latest = leaderboard.iloc[-1]
    score = latest["Relevance"]

    print("\n=== TruLens Evaluation ===")
    print("Output:", output)
    print("Relevance Score:", score)

    assert score >= 0.6
