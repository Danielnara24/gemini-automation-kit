import os
import sys
import mimetypes
import enum
import pathlib
from typing import Any, List, Optional, Union, Dict

try:
    import google.genai as genai
    from google.genai import types
except ImportError:
    print("The 'google-genai' package is not installed.")
    print("Please install it by running the following command in your terminal:")
    print("pip install -q -U google-genai")
    sys.exit(1)

try:
    from pydantic import BaseModel
except ImportError:
    print("The 'pydantic' package is not installed.")
    print("It is needed for the structured response feature.")
    print("Please install it by running the following command in your terminal:")
    print("pip install -q -U pydantic")
    sys.exit(1)


def check_api_key():
    """
    Checks if the Gemini API key is set in the environment variables.

    Returns:
        bool: True if the key is found, False otherwise.
    """
    if 'GEMINI_API_KEY' in os.environ:
        return True
    else:
        print("Gemini API key is not set in environment variables.")
        print("Please set your 'GEMINI_API_KEY'.")
        return False

def add_citations(response: types.GenerateContentResponse) -> str:
    """
    Processes a Gemini response to add inline citations and a formatted source list.
    """
    try:
        metadata = response.candidates[0].grounding_metadata
        if not metadata:
            return response.text
    except (IndexError, AttributeError):
        return response.text

    text = response.text
    supports = metadata.grounding_supports
    chunks = metadata.grounding_chunks

    if not supports or not chunks:
        return text

    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        chunk_indices = sorted(list(set(support.grounding_chunk_indices)))

        if chunk_indices:
            citation_links = []
            for i in chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = "".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    if chunks:
        source_list_header = "\n\n---\n**Sources:**\n"
        source_list = []
        for i, chunk in enumerate(chunks):
            title = chunk.web.title or "Source"
            uri = chunk.web.uri
            source_list.append(f"{i + 1}. [{title}]({uri})")

        text += source_list_header + "\n".join(source_list)

    return text

def _process_media_attachments(
    media_paths: List[str], 
    media_resolution: Optional[Dict[str, str]] = None
) -> Union[List[types.Part], str]:
    """
    Helper function to process a list of file paths into Gemini API Part objects.
    Detects Images, Videos, and PDFs.
    
    Returns:
        List[types.Part]: A list of prepared parts.
        str: An error message string if a file cannot be processed.
    """
    if not media_paths:
        return []

    parts = []
    video_count = 0

    for path in media_paths:
        if not path:
            continue
            
        file_path = pathlib.Path(path)
        if not file_path.exists():
            return f"Error: File not found at '{path}'"

        # Guess MIME type
        mime_type, _ = mimetypes.guess_type(path)
        
        if not mime_type:
            return f"Error: Could not determine MIME type for '{path}'."

        # Read file bytes
        try:
            file_bytes = file_path.read_bytes()
        except Exception as e:
            return f"Error reading file '{path}': {e}"

        # Categorize and create Part
        if mime_type.startswith("image/"):
            # Supported: png, jpeg, webp, heic, heif
            parts.append(types.Part(
                inline_data=types.Blob(data=file_bytes, mime_type=mime_type),
                media_resolution=media_resolution
            ))
        elif mime_type.startswith("video/"):
            video_count += 1
            parts.append(types.Part(
                inline_data=types.Blob(data=file_bytes, mime_type=mime_type),
                media_resolution=media_resolution
            ))
        elif mime_type == "application/pdf":
            parts.append(types.Part(
                inline_data=types.Blob(data=file_bytes, mime_type=mime_type),
                media_resolution=media_resolution
            ))
        else:
            return f"Error: Unsupported file type '{mime_type}' for file '{path}'. Supported types are Images, Videos, and PDFs."

    if video_count > 1:
        print(f"Warning: {video_count} videos detected. While supported by newer models, it is generally recommended to use fewer videos per request for optimal results.")

    return parts

def prompt_gemini(
    model: str = "gemini-2.5-flash",
    prompt: str = "",
    media_attachments: List[str] = None,
    thinking: bool = True,
    temperature: float = 1.0,
    google_search: bool = False,
    code_execution: bool = False,
    url_context: bool = False
):
    """
    Generates content using a Gemini LLM, with optional multimodal inputs (Images, Video, PDF).

    Args:
        model (str): The name of the Gemini model to use.
        prompt (str): The text prompt to send to the model.
        media_attachments (List[str], optional): A list of file paths to local images, videos, 
                                                 or PDFs to include in the prompt. Defaults to None.
        thinking (bool, optional): Enables or disables the thinking feature. Defaults to True.
        temperature (float, optional): Controls randomness. Defaults to 1.0.
        google_search (bool, optional): Enables grounding with Google Search. Defaults to False.
        code_execution (bool, optional): Enables the code execution tool. Defaults to False.
        url_context (bool, optional): Enables the URL context tool. Defaults to False.

    Returns:
        tuple (str, int): A tuple containing the generated text response and the number of input tokens.
                          Returns (error_message, 0) if an error occurs.
    """
    try:
        client = genai.Client()
        # Standard models use thinking_budget
        thinking_budget = -1 if thinking else 0

        # Prepare tools
        tools = []
        if google_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if code_execution:
            tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
        if url_context:
            tools.append(types.Tool(url_context={}))

        config = types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            tools=tools if tools else None
        )

        # Process Media Attachments
        media_parts = []
        if media_attachments:
            result = _process_media_attachments(media_attachments)
            if isinstance(result, str): # Error message
                return result, 0
            media_parts = result

        # Construct Contents (Media should come before text)
        text_part = types.Part(text=prompt)
        contents = media_parts + [text_part]

        # Call API
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        input_token_count = response.usage_metadata.prompt_token_count
        full_response = ""

        try:
            has_grounding_metadata = (
                google_search
                and response.candidates
                and hasattr(response.candidates[0], 'grounding_metadata')
                and response.candidates[0].grounding_metadata
            )

            if has_grounding_metadata:
                full_response = add_citations(response)
                code_parts = []
                for part in response.candidates[0].content.parts:
                    if part.executable_code is not None:
                        code_parts.append(f"\n\n```python\n{part.executable_code.code}\n```")
                    if part.code_execution_result is not None:
                        code_parts.append(f"\n**Execution Result:**\n```\n{part.code_execution_result.output}\n```")
                if code_parts:
                    full_response += "".join(code_parts)
            else:
                response_parts = []
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        response_parts.append(part.text)
                    if part.executable_code is not None:
                        response_parts.append(f"\n```python\n{part.executable_code.code}\n```")
                    if part.code_execution_result is not None:
                        response_parts.append(f"\n**Execution Result:**\n```\n{part.code_execution_result.output}\n```")
                full_response = "".join(response_parts)

            if not full_response and hasattr(response, 'text'):
                full_response = response.text

        except (IndexError, ValueError):
            try:
                full_response = response.text
            except (IndexError, ValueError):
                full_response = "Error: The response was empty or blocked. No content generated."

        return full_response, input_token_count

    except Exception as e:
        return f"An error occurred during content generation: {e}", 0

def prompt_gemini_structured(
    model: str = "gemini-2.5-flash",
    prompt: str = "",
    response_schema: Any = None,
    media_attachments: List[str] = None,
    thinking: bool = True,
    temperature: float = 1.0
):
    """
    Generates structured content (JSON/Enum) using a Gemini LLM, with optional multimodal inputs.

    Args:
        model (str): The name of the Gemini model to use.
        prompt (str): The text prompt.
        response_schema (Any): The schema for the structured output (Pydantic model or Enum).
        media_attachments (List[str], optional): A list of file paths to local images, videos, 
                                                 or PDFs. Defaults to None.
        thinking (bool, optional): Enables or disables the thinking feature. Defaults to True.
        temperature (float, optional): Creativity allowed. Defaults to 1.0.

    Returns:
        tuple (Any, int): A tuple containing the structured response and input tokens.
    """
    try:
        client = genai.Client()
        thinking_budget = -1 if thinking else 0

        mime_type = "application/json"
        if isinstance(response_schema, type) and issubclass(response_schema, enum.Enum):
            mime_type = "text/x.enum"

        config = {
            "temperature": temperature,
            "thinking_config": {"thinking_budget": thinking_budget},
            "response_mime_type": mime_type,
            "response_schema": response_schema,
        }

        # Process Media Attachments
        media_parts = []
        if media_attachments:
            result = _process_media_attachments(media_attachments)
            if isinstance(result, str): # Error message
                return result, 0
            media_parts = result

        # Construct Contents
        text_part = types.Part(text=prompt)
        contents = media_parts + [text_part]

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        input_token_count = response.usage_metadata.prompt_token_count
        return response.parsed, input_token_count

    except Exception as e:
        return f"An error occurred during structured content generation: {e}", 0

def prompt_gemini_3(
    model: str = "gemini-3-pro-preview",
    prompt: str = "",
    response_schema: Any = None,
    media_attachments: List[str] = None,
    thinking_level: str = "high", 
    media_resolution: str = "medium",
    temperature: float = 1.0,
    google_search: bool = False,
    code_execution: bool = False,
    url_context: bool = False
):
    """
    A specialized wrapper for the Gemini 3 model family (e.g., gemini-3-pro-preview).

    Args:
        model (str): Defaults to "gemini-3-pro-preview".
        prompt (str): The text prompt.
        response_schema (Any, optional): Structured output schema.
        media_attachments (List[str], optional): List of file paths (Images, Videos, PDFs).
        thinking_level (str): "low" (faster) or "high" (deep reasoning). Defaults to "high".
        media_resolution (str): "low", "medium", or "high". Applies to images, videos and PDFs.
        temperature (float): Defaults to 1.0.
        google_search (bool): Enable Google Search grounding.
        code_execution (bool): Enable Python code execution.
        url_context (bool): Enable URL reading.

    Returns:
        tuple (Any, int): (Response, Token_Count).
    """
    try:
        # Gemini 3 features require v1alpha
        client = genai.Client(http_options={'api_version': 'v1alpha'})

        # --- Tools ---
        tools = []
        if google_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if code_execution:
            tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
        if url_context:
            tools.append(types.Tool(url_context={}))

        # --- Media Resolution ---
        res_map = {
            "low": "media_resolution_low",
            "medium": "media_resolution_medium",
            "high": "media_resolution_high"
        }
        selected_resolution = res_map.get(media_resolution.lower(), "media_resolution_medium")
        resolution_config = {"level": selected_resolution}

        # --- Content Processing ---
        parts = []
        
        # Add Media Attachments with Resolution Config
        if media_attachments:
            result = _process_media_attachments(media_attachments, media_resolution=resolution_config)
            if isinstance(result, str): # Error message
                return result, 0
            parts.extend(result)

        # Add Text
        parts.append(types.Part(text=prompt))

        # --- Thinking Config ---
        # Map user input to "LOW" or "HIGH". Default is "HIGH".
        valid_levels = ["low", "high"]
        selected_thinking_level = thinking_level.lower()
        if selected_thinking_level not in valid_levels:
            selected_thinking_level = "high"
        
        # Note: Do not mix thinking_budget with thinking_level.
        # include_thoughts=True makes the thoughts visible in response (optional but useful).
        thinking_config = types.ThinkingConfig(
            thinking_level=selected_thinking_level.upper(),
            include_thoughts=True
        )

        # --- MIME Type & Schema ---
        response_mime_type = "text/plain"
        if response_schema:
            if isinstance(response_schema, type) and issubclass(response_schema, enum.Enum):
                response_mime_type = "text/x.enum"
            else:
                response_mime_type = "application/json"

        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=thinking_config,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            tools=tools if tools else None
        )

        # --- Generation ---
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(parts=parts)],
            config=generation_config
        )

        input_token_count = response.usage_metadata.prompt_token_count

        # --- Output Handling ---
        if response_schema:
            return response.parsed, input_token_count

        full_response = ""
        has_grounding = (
            google_search
            and response.candidates
            and hasattr(response.candidates[0], 'grounding_metadata')
            and response.candidates[0].grounding_metadata
        )

        if has_grounding:
            full_response = add_citations(response)
        else:
            full_response = response.text

        return full_response, input_token_count

    except Exception as e:
        return f"Gemini 3 Error: {e}", 0