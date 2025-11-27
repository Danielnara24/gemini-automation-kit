import os
import sys
import mimetypes
import enum
import pathlib
import hashlib
import base64
import time
import itertools
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

def get_remote_file_name(client: genai.Client, file_path: str) -> str | None:
    """
    Checks if a local file is already uploaded to Gemini by comparing SHA-256 hashes.
    
    Args:
        client: The initialized genai.Client object.
        file_path: The local path to the file.
        
    Returns:
        str: The remote file name (e.g., 'files/abc123xyz') if found.
        None: If the file is not found on the server.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    except FileNotFoundError:
        return None

    # Google stores the Hex Digest encoded in Base64
    local_hash_b64 = base64.b64encode(sha256_hash.hexdigest().encode('utf-8')).decode('utf-8')

    try:
        # iterate through remote files looking for a match
        for remote_file in client.files.list():
            if remote_file.sha256_hash == local_hash_b64:
                return remote_file.name
    except Exception:
        # If listing fails (e.g. permission or network), assume not found
        return None
            
    return None

def _process_media_attachments(
    client: genai.Client, 
    media_paths: List[str], 
    inline_limit_mb: float = 10.0,
    media_resolution: Optional[Dict[str, str]] = None
) -> Union[List[types.Part], str]:
    """
    Processes media files. Checks for existing remote files, calculates optimal 
    inline vs upload separation to keep inline data under the limit, uploads 
    necessary files, and returns a list of prepared Gemini Parts.
    """
    if not media_paths:
        return []

    final_parts = []
    
    # Candidates for local processing: (path, size_bytes, mime_type)
    local_candidates = []
    
    # 1. First Pass: Validation, Mime Detection, and Remote Check
    for path in media_paths:
        if not path:
            continue
        
        file_path = pathlib.Path(path)
        if not file_path.exists():
            return f"Error: File not found at '{path}'"

        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            return f"Error: Could not determine MIME type for '{path}'."
        
        # Check supported types
        if not (mime_type.startswith("image/") or mime_type.startswith("video/") or mime_type == "application/pdf"):
             return f"Error: Unsupported file type '{mime_type}' for file '{path}'."

        # Check if already uploaded
        remote_name = get_remote_file_name(client, path)
        if remote_name:
            # It's already there, just add the reference
            final_parts.append(types.Part(
                file_data=types.FileData(file_uri=f"https://generativelanguage.googleapis.com/v1beta/{remote_name}", mime_type=mime_type),
                media_resolution=media_resolution
            ))
        else:
            # It's a candidate for local logic (inline vs upload)
            file_size = file_path.stat().st_size
            local_candidates.append({
                "path": path,
                "size": file_size,
                "mime": mime_type,
                "path_obj": file_path
            })

    if not local_candidates:
        return final_parts

    # 2. Optimization Logic: Subset Sum Problem
    # We want to maximize the size of files kept inline such that Sum(inline) <= Limit
    limit_bytes = inline_limit_mb * 1024 * 1024
    
    inline_files = []
    upload_files = []

    # Calculate total size
    total_candidate_size = sum(c["size"] for c in local_candidates)

    if total_candidate_size <= limit_bytes:
        # All fit inline
        inline_files = local_candidates
    else:
        # Find optimal subset for inline
        # Generate all combinations. 
        # Safety limit: if too many files, fallback to simple greedy (smallest first) to avoid hanging
        if len(local_candidates) > 20:
            # Greedy approach
            sorted_candidates = sorted(local_candidates, key=lambda x: x["size"])
            current_sum = 0
            for c in sorted_candidates:
                if current_sum + c["size"] <= limit_bytes:
                    inline_files.append(c)
                    current_sum += c["size"]
                else:
                    upload_files.append(c)
        else:
            # Optimal combinations approach
            best_sum = 0
            best_combination = []
            
            # Helper to get indices
            indices = range(len(local_candidates))
            
            for r in range(len(local_candidates) + 1):
                for subset_indices in itertools.combinations(indices, r):
                    current_subset = [local_candidates[i] for i in subset_indices]
                    current_size = sum(x["size"] for x in current_subset)
                    
                    if current_size <= limit_bytes:
                        if current_size > best_sum:
                            best_sum = current_size
                            best_combination = current_subset
            
            inline_files = best_combination
            # The rest go to upload
            inline_paths = set(x["path"] for x in inline_files)
            upload_files = [x for x in local_candidates if x["path"] not in inline_paths]

    # 3. Process Inline Files
    for item in inline_files:
        try:
            file_bytes = item["path_obj"].read_bytes()
            final_parts.append(types.Part(
                inline_data=types.Blob(data=file_bytes, mime_type=item["mime"]),
                media_resolution=media_resolution
            ))
        except Exception as e:
            return f"Error reading file '{item['path']}': {e}"

    # 4. Process Upload Files
    for item in upload_files:
        print(f"Uploading '{item['path']}' ({item['size']/1024/1024:.2f} MB)...")
        try:
            uploaded_file = client.files.upload(file=item["path"])
            
            # Wait for ACTIVE state
            while True:
                myfile = client.files.get(name=uploaded_file.name)
                if myfile.state.name == "ACTIVE":
                    break
                elif myfile.state.name == "FAILED":
                    return f"Error: File processing failed for '{item['path']}' on Google's side."
                time.sleep(2)
            
            final_parts.append(types.Part(
                file_data=types.FileData(file_uri=myfile.uri, mime_type=item["mime"]),
                media_resolution=media_resolution
            ))
            print(f"Upload complete: {item['path']}")
            
        except Exception as e:
            return f"Error uploading file '{item['path']}': {e}"

    return final_parts

def delete_all_uploads():
    client = genai.Client()
    
    print("Checking for uploaded files...")
    
    # client.files.list() returns an iterator of all files
    files = list(client.files.list())
    
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files. Starting deletion...")

    for f in files:
        print(f"Deleting {f.name}...")
        try:
            client.files.delete(name=f.name)
        except Exception as e:
            print(f"Failed to delete {f.name}: {e}")

    print("Cleanup complete.")

def prompt_gemini(
    model: str = "gemini-2.5-flash",
    prompt: str = "",
    media_attachments: List[str] = None,
    inline_data_limit: float = 10.0,
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
        inline_data_limit (float): Limit in MB for inline data before forcing upload. Defaults to 10.0.
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

        # Process Media Attachments using the new uploader
        media_parts = []
        if media_attachments:
            result = _process_media_attachments(client, media_attachments, inline_limit_mb=inline_data_limit)
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
    inline_data_limit: float = 10.0,
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
        inline_data_limit (float): Limit in MB for inline data before forcing upload. Defaults to 10.0.
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
            result = _process_media_attachments(client, media_attachments, inline_limit_mb=inline_data_limit)
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
    inline_data_limit: float = 10.0,
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
        inline_data_limit (float): Limit in MB for inline data before forcing upload. Defaults to 10.0.
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
            # Pass the v1alpha client and resolution config to the uploader
            result = _process_media_attachments(
                client, 
                media_attachments, 
                inline_limit_mb=inline_data_limit,
                media_resolution=resolution_config
            )
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