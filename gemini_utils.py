import os
import sys
import mimetypes
import enum
import pathlib
from typing import Any

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

    If the key is not found, it prints detailed instructions on how to set it
    on Windows.

    Returns:
        bool: True if the key is found, False otherwise.
    """
    # The library checks for GEMINI_API_KEY.
    if 'GEMINI_API_KEY' in os.environ:
        return True
    else:
        print("Gemini API key is not set in environment variables.")
        print("Please set your 'GEMINI_API_KEY'.")
        return False

def add_citations(response: types.GenerateContentResponse) -> str:
    """
    Processes a Gemini response to add inline citations and a formatted source list.

    This function takes the full response object, checks for grounding metadata from
    a Google Search, and inserts markdown-style links to the sources directly into
    the text. It also appends a clean, numbered list of all sources at the end.

    Args:
        response (types.GenerateContentResponse): The complete response object
                                                  from the Gemini API call.

    Returns:
        str: The response text with inline citations and a source list added.
             If no grounding metadata is found, returns the original response text.
    """
    # Check if grounding metadata exists. If not, return the plain text.
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

    # Sort supports by end_index in descending order to avoid shifting issues
    # when inserting citations into the text.
    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        # Get unique, sorted chunk indices for this segment of text.
        chunk_indices = sorted(list(set(support.grounding_chunk_indices)))

        if chunk_indices:
            citation_links = []
            for i in chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    # Create a markdown link like `[1](url)`.
                    citation_links.append(f"[{i + 1}]({uri})")

            # Join multiple links for the same text segment, e.g., `[1](url)[2](url)`.
            citation_string = "".join(citation_links)
            # Insert the citation string at the end of the cited segment.
            text = text[:end_index] + citation_string + text[end_index:]

    # Append a formatted list of all sources at the end of the response.
    if chunks:
        source_list_header = "\n\n---\n**Sources:**\n"
        source_list = []
        for i, chunk in enumerate(chunks):
            title = chunk.web.title or "Source"
            uri = chunk.web.uri
            source_list.append(f"{i + 1}. [{title}]({uri})")

        text += source_list_header + "\n".join(source_list)

    return text

def prompt_gemini(
    model: str = "gemini-2.5-flash",
    prompt: str = "",
    video_attachment: str = None,
    pdf_attachment: str = None,
    thinking: bool = True,
    temperature: float = 1.0,
    google_search: bool = False,
    code_execution: bool = False,
    url_context: bool = False
):
    """
    Generates content using a Gemini LLM, with optional video or PDF input.

    Args:
        model (str): The name of the Gemini model to use (e.g., "gemini-2.5-flash").
        prompt (str): The text prompt to send to the model.
        video_attachment (str, optional): The file path to a local video. If provided,
                                           the video will be included in the prompt.
                                           The video file should be less than 20MB for inline upload.
                                           Defaults to None.
        pdf_attachment (str, optional): The file path to a local PDF file. If provided,
                                        the PDF will be included in the prompt.
                                        For files under 20MB. Defaults to None.
        thinking (bool, optional): Enables or disables the thinking feature.
                                   True for dynamic thinking, False to disable.
                                   Defaults to True.
        temperature (float, optional): Controls randomness. Lower for more deterministic
                                       responses. Value is between 0.0 and 2.0.
                                       Defaults to 1.0.
        google_search (bool, optional): If True, enables grounding with Google Search to
                                        provide more accurate and up-to-date responses
                                        with source citations. Defaults to False.
        code_execution (bool, optional): If True, enables the code execution tool, allowing
                                         the model to generate and run Python code to
                                         answer the prompt. Defaults to False.
        url_context (bool, optional): If True, enables the URL context tool, allowing
                                      the model to access and process content from
                                      URLs provided in the prompt. Defaults to False.

    Returns:
        tuple (str, int): A tuple containing the generated text response and the number of input tokens.
                          Returns (error_message, 0) if an error occurs.
    """
    try:
        # The client will automatically pick up the API key from the environment.
        client = genai.Client()

        # Set the thinking budget based on the 'thinking' parameter.
        # -1 enables dynamic thinking.
        # 0 disables thinking.
        thinking_budget = -1 if thinking else 0

        # Prepare tools if any are enabled.
        tools = []
        if google_search:
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            tools.append(grounding_tool)
        if code_execution:
            code_execution_tool = types.Tool(
                code_execution=types.ToolCodeExecution()
            )
            tools.append(code_execution_tool)
        if url_context:
            url_context_tool = types.Tool(
                url_context={}
            )
            tools.append(url_context_tool)

        # Create the generation configuration object.
        config = types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            tools=tools if tools else None
        )

        # Prepare the contents for the API call.
        if video_attachment:
            try:
                # Dynamically determine the MIME type of the video file.
                mime_type, _ = mimetypes.guess_type(video_attachment)
                if not mime_type or not mime_type.startswith('video/'):
                    return f"Error: Could not determine a video MIME type for '{video_attachment}'. Please use a standard video file.", 0

                video_bytes = open(video_attachment, 'rb').read()
                video_part = types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type=mime_type)
                )
                text_part = types.Part(text=prompt)
                # For multimodal prompts, the media should come before the text.
                contents = [video_part, text_part]
            except FileNotFoundError:
                return f"Error: Video file not found at '{video_attachment}'", 0
            except Exception as e:
                return f"An error occurred while processing the video file: {e}", 0
        elif pdf_attachment:
            try:
                pdf_path = pathlib.Path(pdf_attachment)
                pdf_part = types.Part.from_bytes(
                    data=pdf_path.read_bytes(),
                    mime_type='application/pdf'
                )
                text_part = types.Part(text=prompt)
                # For multimodal prompts, the media should come before the text.
                contents = [pdf_part, text_part]
            except FileNotFoundError:
                return f"Error: PDF file not found at '{pdf_attachment}'", 0
            except Exception as e:
                return f"An error occurred while processing the PDF file: {e}", 0
        else:
            # For text-only prompts.
            contents = prompt

        # Call the API to generate content.
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        # Extract the input token count from the usage metadata.
        input_token_count = response.usage_metadata.prompt_token_count

        # Process the response.
        full_response = ""
        try:
            # Check if grounding was used and if metadata is present to add citations.
            has_grounding_metadata = (
                google_search
                and response.candidates
                and hasattr(response.candidates[0], 'grounding_metadata')
                and response.candidates[0].grounding_metadata
            )

            if has_grounding_metadata:
                # If grounding metadata exists, process it to add inline citations and a source list.
                full_response = add_citations(response)

                # Append any code execution parts, as they are not included in `response.text`.
                code_parts = []
                for part in response.candidates[0].content.parts:
                    if part.executable_code is not None:
                        code_parts.append(f"\n\n```python\n{part.executable_code.code}\n```")
                    if part.code_execution_result is not None:
                        code_parts.append(f"\n**Execution Result:**\n```\n{part.code_execution_result.output}\n```")
                if code_parts:
                    full_response += "".join(code_parts)
            else:
                # If no grounding, process all parts normally with improved formatting.
                response_parts = []
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        response_parts.append(part.text)
                    if part.executable_code is not None:
                        response_parts.append(f"\n```python\n{part.executable_code.code}\n```")
                    if part.code_execution_result is not None:
                        response_parts.append(f"\n**Execution Result:**\n```\n{part.code_execution_result.output}\n```")
                full_response = "".join(response_parts)

            # Fallback to response.text if the loop produced no content.
            if not full_response and hasattr(response, 'text'):
                full_response = response.text

        except (IndexError, ValueError):
            # This can happen if the response is blocked or malformed (e.g., no candidates).
            # Fallback to response.text which might contain info on why it was blocked.
            try:
                full_response = response.text
            except (IndexError, ValueError):
                full_response = "Error: The response was empty or blocked. No content generated."

        return full_response, input_token_count

    except Exception as e:
        # Catches potential errors like invalid API keys, network issues, etc.
        return f"An error occurred during content generation: {e}", 0

def prompt_gemini_structured(
    model: str,
    prompt: str,
    response_schema: Any,
    video_attachment: str = None,
    pdf_attachment: str = None,
    thinking: bool = True,
    temperature: float = 1.0
):
    """
    Generates structured content (JSON/Enum) using a Gemini LLM, with a required response schema.

    This function constrains the model's output to a specific structure,
    such as JSON or a single choice from a list (Enum). Instead of free-form
    text, it returns a parsed Python object that matches the provided schema.

    Usage Examples:
        1. JSON Output using Pydantic:
            from pydantic import BaseModel

            class Book(BaseModel):
                title: str
                author: str

            prompt = "List the book 'Dune'"
            schema = Book
            book_object, tokens = prompt_gemini_structured(
                model="gemini-2.5-flash",
                prompt=prompt,
                response_schema=schema
            )
            # book_object will be an instance of the Book class.
            print(book_object.title)

        2. Enum Output:
            import enum

            class Sentiment(enum.Enum):
                POSITIVE = "Positive"
                NEGATIVE = "Negative"
                NEUTRAL = "Neutral"

            prompt = "Analyze the sentiment of this sentence: 'I love sunny days.'"
            schema = Sentiment
            sentiment_enum, tokens = prompt_gemini_structured(
                model="gemini-2.5-flash",
                prompt=prompt,
                response_schema=schema
            )
            # sentiment_enum will be a member of the Sentiment enum, e.g., Sentiment.POSITIVE
            print(sentiment_enum)

    Args:
        model (str): The name of the Gemini model to use (e.g., "gemini-2.5-flash").
        prompt (str): The text prompt to send to the model.
        response_schema (Any): The schema for the structured output. This must be a
                               type annotation. For JSON output, use a Pydantic
                               `BaseModel` or a `list` of a `BaseModel` (e.g., `MyModel`
                               or `list[MyModel]`). For Enum output, use an `enum.Enum`
                               subclass (e.g., `MyEnum`).
        video_attachment (str, optional): The file path to a local video. If provided,
                                           the video will be included in the prompt.
                                           The video file should be less than 20MB for inline upload.
                                           Defaults to None.
        pdf_attachment (str, optional): The file path to a local PDF file. If provided,
                                        the PDF will be included in the prompt.
                                        For files under 20MB. Defaults to None.
        thinking (bool, optional): Enables or disables the thinking feature. True for
                                   dynamic thinking, False to disable. Defaults to True.
        temperature (float, optional): Creativity allowed in responses. Value is between 0.0 and 2.0.
                                       Defaults to 1.0.

    Returns:
        tuple (Any, int): A tuple containing the structured, parsed response and the
                          number of input tokens. The type of the response object will
                          match the `response_schema`. Returns a `(str, int)` tuple
                          with an error message and 0 tokens if an error occurs.
    """
    try:
        # The client will automatically pick up the API key from the environment.
        client = genai.Client()

        # Set the thinking budget based on the 'thinking' parameter.
        thinking_budget = -1 if thinking else 0

        # Determine the appropriate MIME type based on the response_schema provided.
        # Assumes JSON for Pydantic models/lists, and text/x.enum for enums.
        mime_type = "application/json"
        if isinstance(response_schema, type) and issubclass(response_schema, enum.Enum):
            mime_type = "text/x.enum"

        # Create the generation configuration dictionary.
        config = {
            "temperature": temperature,
            "thinking_config": {"thinking_budget": thinking_budget},
            "response_mime_type": mime_type,
            "response_schema": response_schema,
        }

        # Prepare the contents for the API call.
        if video_attachment:
            try:
                # Dynamically determine the MIME type of the video file.
                video_mime_type, _ = mimetypes.guess_type(video_attachment)
                if not video_mime_type or not video_mime_type.startswith('video/'):
                    return f"Error: Could not determine a video MIME type for '{video_attachment}'. Please use a standard video file.", 0

                video_bytes = open(video_attachment, 'rb').read()
                video_part = types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type=video_mime_type)
                )
                text_part = types.Part(text=prompt)
                contents = [video_part, text_part]
            except FileNotFoundError:
                return f"Error: Video file not found at '{video_attachment}'", 0
            except Exception as e:
                return f"An error occurred while processing the video file: {e}", 0
        elif pdf_attachment:
            try:
                pdf_path = pathlib.Path(pdf_attachment)
                pdf_part = types.Part.from_bytes(
                    data=pdf_path.read_bytes(),
                    mime_type='application/pdf'
                )
                text_part = types.Part(text=prompt)
                # For multimodal prompts, the media should come before the text.
                contents = [pdf_part, text_part]
            except FileNotFoundError:
                return f"Error: PDF file not found at '{pdf_attachment}'", 0
            except Exception as e:
                return f"An error occurred while processing the PDF file: {e}", 0
        else:
            # For text-only prompts.
            contents = prompt

        # Call the API to generate content.
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        # Extract the input token count from the usage metadata.
        input_token_count = response.usage_metadata.prompt_token_count

        # Return the parsed, structured object instead of the raw text.
        return response.parsed, input_token_count

    except Exception as e:
        # Catches potential errors like invalid API keys, network issues, etc.
        return f"An error occurred during structured content generation: {e}", 0
