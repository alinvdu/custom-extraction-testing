import pytest
import respx
from httpx import Response
from LLMService import LLMService
from typing import Annotated
import msgspec

@pytest.mark.asyncio
@respx.mock
async def test_fetch_data():
    url = LLMService.base_url

    class Test(msgspec.Struct):
        example1: Annotated[str, "Just the first example"]
        example2: Annotated[str, "Just the second example"]


    respx.post(url).mock(return_value=Response(200, json={
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "arguments": '{"example1": "I am just here", "example2": "So am I"}'
                    }
                }]
            }
        }]
    }))

    result = await LLMService(model="gpt-4o-mini", api_key="test").extraction(file_content="test",
        extraction_interface=Test)

    assert result.example1 == "I am just here"
    assert result.example2 == "So am I"
