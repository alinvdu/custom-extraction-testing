from typing import Any, Optional, Dict
import httpx
import msgspec
from msgspec import Struct, ValidationError
from llm_service.llmUtils import struct_to_openai_tool_schema

from typing import TypeVar

T = TypeVar('T', bound=Struct)

def compose_prompt_for_extraction_legacy(data: str, schema: Dict[str, Any], notes: Dict[str, str]) -> str:
    return f"""
        Extract the following fields from the data to match the following schema:
        {schema}

        Data:
        {data}

        Return the response as JSON following the Fields schema above and nothing else, here's the schema again for reference:
        Schema:
        {schema}

        Instructions:
        {notes}
    """

class LLMService:
    """
    Service class for interacting with the OpenAI language model API for document extraction tasks.

    This class is intended to be used as a backend utility for extracting daata from document text using an LLM. It manages API authentication, request formatting, and error handling.
    It also adds retry mechanism. To use it please specify a msgspec interface.

    Parameters:
        model (str, Required): The OpenAI model to use (e.g., 'gpt-4o-mini').
        api_key (str, Required): The OpenAI API key for authentication.
        temperature (float): Sampling temperature for the LLM (default: 0.7).
    """
    base_url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str, api_key: Optional[str], temperature: float = 0.7):
        if not api_key:
            raise Exception("Please define OPENAI_API_KEY inside your .env environment")
        self.model = model
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.temperature = temperature

    async def extraction(self, file_content: str, extraction_interface: type[T]) -> T:
        """
        Extracts data from the file_content, please provide a msgspec interface to adhere to.

        Parameters:
            file_content (str): The full text content of the document from which to extract the Introduction.

        Returns:
            str: The extracted Introduction section as returned by the LLM.

        Intended use:
            Call this method with the text of a document (e.g., after extracting text from a PDF) to obtain the Introduction section using the configured LLM model.
        """
        tool_schema = struct_to_openai_tool_schema(extraction_interface)

        # schema, notes = struct_to_llm_schema(extraction_interface)
        # composed_prompt = compose_prompt_for_extraction(file_content, schema=schema, notes=notes)

        data = {
            "model": self.model,
            "tools": [tool_schema],
            "tool_choice": "auto",
            "messages": [{
                "role": "system",
                "content": "You are an Expert Document Extractor"
            },
            {
                "role": "user",
                "content": file_content,
            }]
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_url, headers=self.headers, json=data, timeout=30.0)
                response.raise_for_status()
                tool_response = response.json()["choices"][0]["message"].get("tool_calls", [])[0]
                arguments = tool_response["function"]["arguments"]
                
                try:
                    res = msgspec.json.decode(arguments, type=extraction_interface)
                except ValidationError as e:
                    raise Exception(f"Validation failed with error: {e}")
                except Exception as e:
                    raise Exception(f"Unexpected error processing response: {e}")

                return res

        except httpx.HTTPStatusError as exc:
            print(f"HTTP exception occurred: {exc}")
            print(f"Response content: {exc.response.text}")
            raise

        # define the output schema as msgspec object

        # create the prompt for extraction

        # 