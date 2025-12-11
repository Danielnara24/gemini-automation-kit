import requests
from PIL import Image, ImageDraw, ImageFont, ImageColor
import io
import json
import os
import sys
import logging
import mimetypes
import enum
import pathlib
import hashlib
import base64
import time
import itertools
import google.genai as genai
from google.genai import types
from pydantic import BaseModel
from typing import Any, List, Optional, Union, Dict

# Initialize logger for this module
logger = logging.getLogger(__name__)

def check_api_key():
    """
    Checks if the Gemini API key is set in the environment variables.

    Returns:
        bool: True if the key is found, False otherwise.
    """
    if 'GEMINI_API_KEY' in os.environ:
        return True
    else:
        logger.warning("Gemini API key is not set in environment variables. Please set your 'GEMINI_API_KEY'.")
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

def _parse_video_timestamp(value: Union[str, int, float, None]) -> Optional[str]:
    """
    Parses timestamps (int, 'MM:SS', 'HH:MM:SS') into '123s' format.
    """
    if value is None:
        return None
    
    # If it's already a number, return as seconds string
    if isinstance(value, (int, float)):
        return f"{int(value)}s"
    
    val_str = str(value).strip()
    
    # If it contains colon, parse as time format
    if ":" in val_str:
        try:
            parts = val_str.split(":")
            seconds = 0
            for part in parts:
                seconds = seconds * 60 + float(part)
            return f"{int(seconds)}s"
        except ValueError:
            return None

    # If it ends with 's', assume valid (e.g. "40s")
    if val_str.lower().endswith("s"):
        return val_str
    
    # If string is just a number
    if val_str.isdigit():
        return f"{val_str}s"
        
    return None

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
    media_paths: List[Union[str, Dict[str, Any]]], 
    inline_limit_mb: float = 20.0,
    media_resolution: Optional[Dict[str, str]] = None
) -> Union[List[types.Part], str]:
    """
    Processes media files (Local paths, YouTube URLs, and YouTube Dicts with timestamps).
    Supports Images, Videos, Audio, and PDFs.
    """
    if not media_paths:
        return []

    final_parts = []
    
    # Candidates for local processing: (path, size_bytes, mime_type)
    local_candidates = []
    
    for item in media_paths:
        if not item:
            continue
        
        # --- 1. Normalize Input (Distinguish between URL/Dict and Local Path) ---
        target_path_or_url = ""
        video_meta = None
        
        if isinstance(item, dict):
            target_path_or_url = item.get("url", "")
            # Process timestamps if present
            start = _parse_video_timestamp(item.get("start"))
            end = _parse_video_timestamp(item.get("end"))
            
            if start or end:
                video_meta = types.VideoMetadata(
                    start_offset=start,
                    end_offset=end
                )
        else:
            target_path_or_url = str(item)

        # --- 2. Check for YouTube / Web URL ---
        # Simple check for YouTube URLs to handle them separately from local files
        if "youtube.com" in target_path_or_url or "youtu.be" in target_path_or_url:
            part_args = {
                "file_data": types.FileData(file_uri=target_path_or_url),
                "media_resolution": media_resolution
            }
            if video_meta:
                part_args["video_metadata"] = video_meta
                
            final_parts.append(types.Part(**part_args))
            continue
            
        # --- 3. Local File Processing ---
        path = target_path_or_url
        file_path = pathlib.Path(path)
        
        if not file_path.exists():
            msg = f"Error: File not found at '{path}'"
            logger.error(msg)
            return msg

        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            msg = f"Error: Could not determine MIME type for '{path}'."
            logger.error(msg)
            return msg
        
        # Supported types
        is_supported = (
            mime_type.startswith("image/") or 
            mime_type.startswith("video/") or 
            mime_type.startswith("audio/") or 
            mime_type == "application/pdf"
        )
        
        if not is_supported:
             msg = f"Error: Unsupported file type '{mime_type}' for file '{path}'."
             logger.error(msg)
             return msg

        # Check if already uploaded
        remote_name = get_remote_file_name(client, path)
        if remote_name:
            logger.info(f"File '{path}' found remotely as '{remote_name}'. Using existing file.")
            file_uri = f"https://generativelanguage.googleapis.com/v1beta/{remote_name}"
            final_parts.append(types.Part(
                file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
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

    # --- 4. Optimization Logic (Local Files Only) ---
    limit_bytes = inline_limit_mb * 1024 * 1024
    inline_files = []
    upload_files = []

    total_candidate_size = sum(c["size"] for c in local_candidates)

    if total_candidate_size <= limit_bytes:
        inline_files = local_candidates
    else:
        # Greedy fallback for speed if many files, otherwise combinations
        if len(local_candidates) > 20:
            sorted_candidates = sorted(local_candidates, key=lambda x: x["size"])
            current_sum = 0
            for c in sorted_candidates:
                if current_sum + c["size"] <= limit_bytes:
                    inline_files.append(c)
                    current_sum += c["size"]
                else:
                    upload_files.append(c)
        else:
            best_sum = 0
            best_combination = []
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
            inline_paths = set(x["path"] for x in inline_files)
            upload_files = [x for x in local_candidates if x["path"] not in inline_paths]

    # --- 5. Process Inline Files ---
    for item in inline_files:
        try:
            file_bytes = item["path_obj"].read_bytes()
            final_parts.append(types.Part(
                inline_data=types.Blob(data=file_bytes, mime_type=item["mime"]),
                media_resolution=media_resolution
            ))
        except Exception as e:
            msg = f"Error reading file '{item['path']}': {e}"
            logger.error(msg)
            return msg

    # --- 6. Process Upload Files ---
    for item in upload_files:
        logger.info(f"Uploading '{item['path']}' ({item['size']/1024/1024:.2f} MB)...")
        try:
            uploaded_file = client.files.upload(file=item["path"])
            
            # Wait for ACTIVE state
            while True:
                myfile = client.files.get(name=uploaded_file.name)
                if myfile.state.name == "ACTIVE":
                    break
                elif myfile.state.name == "FAILED":
                    msg = f"Error: File processing failed for '{item['path']}' on Google's side."
                    logger.error(msg)
                    return msg
                time.sleep(1)
            
            file_uri = f"https://generativelanguage.googleapis.com/v1beta/{myfile.name}"
            
            final_parts.append(types.Part(
                file_data=types.FileData(file_uri=file_uri, mime_type=item["mime"]),
                media_resolution=media_resolution
            ))
            logger.info(f"Upload complete: {item['path']}")
            
        except Exception as e:
            msg = f"Error uploading file '{item['path']}': {e}"
            logger.error(msg)
            return msg

    return final_parts

def delete_all_uploads():
    client = genai.Client()
    
    logger.info("Checking for uploaded files...")
    
    # client.files.list() returns an iterator of all files
    try:
        files = list(client.files.list())
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return
    
    if not files:
        logger.info("No files found.")
        return

    logger.info(f"Found {len(files)} files. Starting deletion...")

    for f in files:
        logger.info(f"Deleting {f.name}...")
        try:
            client.files.delete(name=f.name)
        except Exception as e:
            logger.error(f"Failed to delete {f.name}: {e}")

    logger.info("Cleanup complete.")

def prompt_gemini(
    model: str = "gemini-2.5-flash",
    prompt: str = "",
    media_attachments: List[str] = None,
    response_schema: Any = None,
    upload_threshold_mb: float = 20.0,
    thinking: bool = True,
    temperature: float = 1.0,
    google_search: bool = False,
    code_execution: bool = False,
    url_context: bool = False,
    max_retries: int = 0
):
    """
    Generates content using a Gemini LLM, supports structured output (JSON/Enum) and multimodal inputs.

    Args:
        model (str): The name of the Gemini model to use.
        prompt (str): The text prompt to send to the model.
        media_attachments (List[str], optional): A list of file paths (audio, images, videos, PDFs).
        response_schema (Any, optional): Schema for structured output (Pydantic model or Enum).
                                         NOTE: Cannot be used with tools (search/code) on Gemini 2.5.
        upload_threshold_mb (float): Limit in MB for inline data before forcing upload. Defaults to 20.0.
        thinking (bool, optional): Enables or disables the thinking feature. Defaults to True.
        temperature (float, optional): Controls randomness. Defaults to 1.0.
        google_search (bool, optional): Enables grounding with Google Search. Defaults to False.
        code_execution (bool, optional): Enables the code execution tool. Defaults to False.
        url_context (bool, optional): Enables the URL context tool. Defaults to False.
        max_retries (int, optional): Number of times to retry the API call if it fails. Defaults to 0.

    Returns:
        tuple (str | Any, int): A tuple containing the response (text or parsed object) and input token count.
    """
    try:
        # --- VALIDATION CHECK ---
        # Disable tools if response_schema is provided
        if response_schema and (google_search or code_execution or url_context):
            logger.warning("Warning: response_schema cannot be used with tools (google_search, code_execution, url_context) on Gemini 2.5. Disabling all tools.")
            google_search = False
            code_execution = False
            url_context = False


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

        # Determine MIME type for response
        response_mime_type = "text/plain"
        if response_schema:
            if isinstance(response_schema, type) and issubclass(response_schema, enum.Enum):
                response_mime_type = "text/x.enum"
            else:
                response_mime_type = "application/json"

        config = types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            tools=tools if tools else None,
            response_mime_type=response_mime_type,
            response_schema=response_schema
        )

        # Process Media Attachments using the new uploader
        media_parts = []
        if media_attachments:
            result = _process_media_attachments(client, media_attachments, inline_limit_mb=upload_threshold_mb)
            if isinstance(result, str): # Error message
                return result, 0
            media_parts = result

        # Construct Contents (Media should come before text)
        text_part = types.Part(text=prompt)
        contents = media_parts + [text_part]

        # Call API with retry logic
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                break
            except Exception as e:
                if attempt == max_retries:
                    raise e
                if "GenerateRequestsPerMinute" in str(e):
                    logger.warning(f"RPM limit reached (Attempt {attempt + 1}). Sleeping for 50 seconds...")
                    time.sleep(50)
                else:
                    time.sleep(1)

        input_token_count = response.usage_metadata.prompt_token_count
        
        if response_schema:
            return response.parsed, input_token_count

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

def prompt_gemini_3(
    model: str = "gemini-3-pro-preview",
    prompt: str = "",
    response_schema: Any = None,
    media_attachments: List[str] = None,
    upload_threshold_mb: float = 20.0,
    thinking_level: str = "high", 
    media_resolution: str = "medium",
    temperature: float = 1.0,
    google_search: bool = False,
    code_execution: bool = False,
    url_context: bool = False,
    max_retries: int = 0
):
    """
    A specialized wrapper for the Gemini 3 model family (e.g., gemini-3-pro-preview).

    Args:
        model (str): Defaults to "gemini-3-pro-preview".
        prompt (str): The text prompt.
        response_schema (Any, optional): Structured output schema.
        media_attachments (List[str], optional): List of file paths (Audio, Images, Videos, PDFs).
        upload_threshold_mb (float): Limit in MB for inline data before forcing upload. Defaults to 20.0.
        thinking_level (str): "low" (faster) or "high" (deep reasoning). Defaults to "high".
        media_resolution (str): "low", "medium", or "high". Applies to images, videos and PDFs.
        temperature (float): Defaults to 1.0.
        google_search (bool): Enable Google Search grounding.
        code_execution (bool): Enable Python code execution.
        url_context (bool): Enable URL reading.
        max_retries (int, optional): Number of times to retry the API call if it fails. Defaults to 0.

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
                inline_limit_mb=upload_threshold_mb,
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

        # Call API with retry logic
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[types.Content(parts=parts)],
                    config=generation_config
                )
                # If successful, break the retry loop
                break
            except Exception as e:
                # If this was the last attempt, raise the exception to be handled by the outer block
                if attempt == max_retries:
                    raise e
                
                # Check for RPM error
                if "GenerateRequestsPerMinute" in str(e):
                    logger.warning(f"RPM limit reached (Attempt {attempt + 1}). Sleeping for 50 seconds...")
                    time.sleep(50)
                else:
                    time.sleep(1)

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
        return f"An error occurred during content generation: {e}", 0
    
# --- Helper Functions for Detection ---

def _clean_json_markdown(text: str) -> str:
    """Removes markdown code fencing from JSON strings."""
    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]
    return text.strip()

def _visualize_2d(image_path_or_url: str, bounding_boxes: List[Dict[str, Any]]) -> Image.Image:
    """
    Draws bounding boxes and labels on the image. 
    Handles both local paths and URLs for the source image.
    """
    try:
        # Load Image
        if str(image_path_or_url).startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url, stream=True)
            response.raise_for_status()
            im = Image.open(io.BytesIO(response.content))
        else:
            im = Image.open(image_path_or_url)
        
        # Ensure image is in a mode that supports color drawing
        if im.mode != 'RGB':
            im = im.convert('RGB')

        draw = ImageDraw.Draw(im)
        width, height = im.size
        
        # distinct colors for different objects
        colors = [
            'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 
            'cyan', 'magenta', 'lime', 'teal', 'coral'
        ]

        # Use default font or try to load a better one if available
        try:
            # Try loading a standard font, fallback to default
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        for i, box_data in enumerate(bounding_boxes):
            box = box_data.get("box_2d")
            label = box_data.get("label", "Object")
            
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box
            
            abs_y1 = int(ymin / 1000 * height)
            abs_x1 = int(xmin / 1000 * width)
            abs_y2 = int(ymax / 1000 * height)
            abs_x2 = int(xmax / 1000 * width)

            # Draw Rectangle
            color = colors[i % len(colors)]
            draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3)
            
            # Draw Label with background
            text_bbox = draw.textbbox((abs_x1, abs_y1), label, font=font)
            # offset label slightly above or inside box
            text_loc = (abs_x1, max(0, abs_y1 - 20))
            
            # optional: draw text background for readability
            draw.rectangle(draw.textbbox(text_loc, label, font=font), fill=color)
            draw.text(text_loc, label, fill="black" if color in ['yellow', 'lime', 'cyan'] else "white", font=font)

        return im
    except Exception as e:
        logger.error(f"Failed to visualize detection: {e}")
        return None

# --- Main Detect Function ---

def detect_2d(
    model: str = "gemini-2.5-flash-lite",
    prompt: str = "Detect all items.",
    image_path: str = None,
    visual: bool = False,
    temperature: float = 0.5,
    max_retries: int = 0
) -> tuple[Union[List[Dict[str, Any]], str], Optional[Image.Image]]:
    """
    Performs 2D object detection on an image using Gemini.
    
    Args:
        model (str): The model to use.
        prompt (str): Instructions on what to detect.
        image_path (str): Local path or URL to the image.
        visual (bool): If True, returns a PIL Image with bounding boxes drawn.
        temperature (float): Model temperature. Docs suggest ~0.5 for detection.
        max_retries (int): Retry attempts.

    Returns:
        tuple: (JSON Data [List of Dicts], PIL Image [or None])
    """
    if not image_path:
        return "Error: image_path is required.", None

    # Handle client initialization based on model family
    if "gemini-3" in model:
         client = genai.Client(http_options={'api_version': 'v1alpha'})
    else:
         client = genai.Client()

    # 1. System Instructions (Exact string requested)
    system_instruction = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        "label" should be each item's unique characteristics (colors, size, position, adjectives, etc..).
    """

    # 2. Configure Thinking based on model
    if model == "gemini-2.5-pro":
        thinking_config = types.ThinkingConfig(include_thoughts=False)
    elif model == "gemini-3-pro-preview":
        thinking_config = types.ThinkingConfig(thinking_level="low")
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=0)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=temperature,
        system_instruction=system_instruction,
        thinking_config=thinking_config
    )

    # 3. Process Media
    media_parts = _process_media_attachments(client, [image_path], inline_limit_mb=20.0)
    if isinstance(media_parts, str): 
        return f"Media Error: {media_parts}", None

    # 4. Generate Content
    contents = media_parts + [types.Part(text=prompt)]
    
    response_text = ""
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            response_text = response.text
            break
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Detection failed: {e}")
                return f"Error: {e}", None
            time.sleep(1)

    # 5. Parse Output
    try:
        clean_json = _clean_json_markdown(response_text)
        json_data = json.loads(clean_json)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response: {response_text}")
        return f"Error: Could not parse model response. Raw: {response_text}", None

    # 6. Visualization
    annotated_image = None
    if visual:
        annotated_image = _visualize_2d(image_path, json_data)

    return json_data, annotated_image