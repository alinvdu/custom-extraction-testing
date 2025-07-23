from typing import Dict, get_type_hints, get_args, get_origin, Annotated, Literal, List, Any, Tuple
import re
import inspect


def struct_to_llm_schema(cls) -> Tuple[Dict[str, Any], Dict[str, str]]:
    type_hints = get_type_hints(cls, include_extras=True)
    schema: Dict[str, Any] = {}
    notes: Dict[str, str] = {}

    for field_name, annotated_type in type_hints.items():
        if get_origin(annotated_type) is Annotated:
            base_type, *annotations = get_args(annotated_type)
            description = annotations[0] if annotations else ""
        else:
            base_type = annotated_type
            description = ""

        origin = get_origin(base_type)
        args = get_args(base_type)

        field_type: str | List = ""

        # Determine the schema type
        if origin is Literal:
            field_type = list(args)
        elif origin in (list, List) and get_origin(args[0]) is Literal:
            field_type = list(get_args(args[0]))
        elif base_type is str:
            field_type = "string"
        elif base_type is int:
            field_type = "integer"
        elif base_type is float:
            field_type = "float"
        elif base_type is bool:
            field_type = "boolean"
        else:
            field_type = str(base_type)

        schema[field_name] = field_type

        # Construct notes with more info (type + description)
        if description:
            if isinstance(field_type, list):
                # Mention the list of choices
                notes[field_name] = f"{description} Choose one or more from: {', '.join(map(str, field_type))}."
            else:
                notes[field_name] = description
        else:
            if isinstance(field_type, list):
                notes[field_name] = f"Choose one or more from: {', '.join(map(str, field_type))}."

    return schema, notes

def struct_to_openai_tool_schema(cls) -> Dict[str, Any]:
    """
    Converts a msgspec.Struct class to OpenAI tool/function schema format.

    Returns:
        A dict compatible with OpenAI's "tools" format.
    """
    type_hints = get_type_hints(cls, include_extras=True)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for field_name, annotated_type in type_hints.items():
        if get_origin(annotated_type) is Annotated:
            base_type, *annotations = get_args(annotated_type)
            description = annotations[0] if annotations else ""
        else:
            base_type = annotated_type
            description = ""

        origin = get_origin(base_type)
        args = get_args(base_type)

        field_schema: Dict[str, Any] = {}

        if origin is Literal:
            field_schema = {
                "type": "string",
                "enum": list(args)
            }
        elif origin in (list, List):
            item_origin = get_origin(args[0])
            if item_origin is Literal:
                field_schema = {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(get_args(args[0]))
                    }
                }
            else:
                field_schema = {
                    "type": "array",
                    "items": {"type": "string"}  # fallback
                }
        elif base_type is str:
            field_schema = {"type": "string"}
        elif base_type is int:
            field_schema = {"type": "integer"}
        elif base_type is float:
            field_schema = {"type": "number"}
        elif base_type is bool:
            field_schema = {"type": "boolean"}
        else:
            field_schema = {"type": "string"}  # fallback for unknowns

        if description:
            field_schema["description"] = description

        properties[field_name] = field_schema
        required.append(field_name)

    # Generate a descriptive name and docstring
    fn_name = cls.__name__.lower()
    doc = inspect.getdoc(cls) or f"Extract structured {fn_name} data from document text."

    return {
        "type": "function",
        "function": {
            "name": fn_name,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

def strip_json_markers(data: str) -> str:
    """
    Remove JSON code block markers from a string if present.
    
    Args:
        data: The input string that might be wrapped in ```json markers
        
    Returns:
        The input string with ```json markers removed if they were present
    """
    if not isinstance(data, str):
        return data
        
    # Remove ```json at the start and ``` at the end if both are present
    pattern = r'^\s*```(?:json\n)?(.*?)\s*```\s*$'
    match = re.search(pattern, data, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return data
