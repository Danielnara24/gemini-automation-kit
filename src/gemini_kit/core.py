import os
import sys
import logging
import mimetypes
import enum
import pathlib
import hashlib
import base64
import time
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import io
import json
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
    
# --- Helper Functions for Vision Understanding ---

def _extract_json_from_markdown(text: str) -> str:
    """Extracts JSON string from Markdown fencing."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("```json"):
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            return json_output.strip()
    # If no fencing, assume raw text is JSON
    return text.strip()

def _get_color(index: int) -> str:
    """Returns a distinct color based on index."""
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 
        'cyan', 'magenta', 'lime', 'teal', 'coral', 'gold'
    ]
    return colors[index % len(colors)]

def _project_3d_center_to_2d(
    box_3d: List[float], 
    img_width: int, 
    img_height: int, 
    fov: float = 60.0
) -> Optional[tuple[int, int]]:
    """
    Approximates the 2D pixel projection of a 3D box center.
    Assumes standard camera intrinsics with the provided FOV.
    """
    try:
        x_metric, y_metric, z_metric = box_3d[0], box_3d[1], box_3d[2]
        
        # Avoid division by zero or negative Z (behind camera)
        if z_metric <= 0:
            return None

        # Calculate focal length based on FOV
        focal_length = img_width / (2 * np.tan(np.radians(fov) / 2))
        
        cx, cy = img_width / 2, img_height / 2
        
        # Simple perspective projection
        x_px = int((x_metric / z_metric) * focal_length + cx)
        y_px = int((y_metric / z_metric) * focal_length + cy)
        
        return (x_px, y_px)
    except Exception:
        return None

def vision_understanding(
    media_path: str,
    prompt: str,
    task: str = "detect_2d",
    model: str = "gemini-2.5-flash",
    visual: bool = False,
    segmentation_threshold: int = 127,
    max_retries: int = 1
) -> tuple[Any, Optional[Image.Image]]:
    """
    Performs specialized computer vision tasks using Gemini.

    Args:
        media_path (str): Path to the local image file.
        prompt (str): Text description of what to find (e.g., "the cat", "all bottles").
        task (str): One of 'detect_2d', 'detect_3d', 'pointing', 'segmentation'.
        model (str): Gemini model ID.
        visual (bool): If True, returns an annotated PIL Image.
        segmentation_threshold (int): Threshold (0-255) for binary mask generation.
        max_retries (int): Retry attempts.

    Returns:
        tuple: (JSON Data (List/Dict), Annotated PIL Image (or None))
    """
    
    valid_tasks = ['detect_2d', 'detect_3d', 'pointing', 'segmentation']
    if task not in valid_tasks:
        raise ValueError(f"Invalid task '{task}'. Must be one of {valid_tasks}")

    # Load Image for processing dimensions and visualization later
    try:
        pil_image = Image.open(media_path)
        img_width, img_height = pil_image.size
    except Exception as e:
        logger.error(f"Could not load image at {media_path}: {e}")
        return {"error": str(e)}, None

    client = genai.Client()

    # --- Construct Task-Specific System Instructions ---
    # These prompts are engineered based on Google's Cookbook examples for consistency.
    
    base_instruction = ""
    
    if task == "detect_2d":
        base_instruction = """
            Return bounding boxes as a JSON array with labels. Never return masks or code fencing. MAXIMUM 25 OBJECTS.
            If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
            Output format: list of objects with keys 'label' and 'box_2d'.
            box_2d is [ymin, xmin, ymax, xmax] normalized to 0-1000.
        """
    elif task == "pointing":
        base_instruction = """
            Point to the items described. MAXIMUM 25 OBJECTS.
            The answer should follow the json format:
            [{\"point\": [y, x], \"label\": \"your_label\"}, ...]
            The points are in [y, x] format normalized to 0-1000.
        """
    elif task == "detect_3d":
        base_instruction = """
            You are an expert in 3D spatial understanding. 
            Detect the 3D bounding boxes of the requested items. NO MORE THAN 10 ITEMS.
            Output a strict JSON list where each entry contains:
            1. 'label': The object name.
            2. 'box_3d': An array of exactly 9 numbers representing [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw].
        """
    elif task == "segmentation":
        base_instruction = """
            Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d' ([ymin, xmin, ymax, xmax] 0-1000),
            the segmentation mask in key 'mask' (base64 png), and the text label in the key 'label'.
            MAXIMUM 25 OBJECTS.
        """

    # --- API Call ---
    config = types.GenerateContentConfig(
        system_instruction=base_instruction,
        temperature=0.5, # Specific temp rec for vision tasks
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Must be 0 for spatial tasks
    )

    # We send the PIL image directly (Client handles conversion) or use the existing uploader
    # Using the direct PIL object is faster for single images
    contents = [prompt, pil_image]

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
                logger.error(f"Vision task failed: {e}")
                return {"error": str(e)}, None
            time.sleep(2)

    # --- Parse JSON ---
    try:
        clean_json = _extract_json_from_markdown(response_text)
        data = json.loads(clean_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON response from Gemini.")
        return {"error": "Invalid JSON response", "raw": response_text}, None

    if not visual:
        return data, None

    # --- Visualization Logic ---
    annotated_img = pil_image.copy()
    draw = ImageDraw.Draw(annotated_img, "RGBA") # RGBA for transparency support
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for i, item in enumerate(data):
        color_name = _get_color(i)
        color_rgb = ImageColor.getrgb(color_name)
        label = item.get("label", "Object")

        # 1. Detect 2D
        if task == "detect_2d" and "box_2d" in item:
            # box_2d is [ymin, xmin, ymax, xmax] 0-1000
            box = item["box_2d"]
            ymin, xmin, ymax, xmax = box
            
            abs_y1 = int(ymin / 1000 * img_height)
            abs_x1 = int(xmin / 1000 * img_width)
            abs_y2 = int(ymax / 1000 * img_height)
            abs_x2 = int(xmax / 1000 * img_width)
            
            draw.rectangle([(abs_x1, abs_y1), (abs_x2, abs_y2)], outline=color_rgb, width=3)
            draw.text((abs_x1 + 5, abs_y1 + 5), label, fill=color_rgb, font=font)

        # 2. Pointing
        elif task == "pointing" and "point" in item:
            # point is [y, x] 0-1000
            pt = item["point"]
            y_norm, x_norm = pt[0], pt[1]
            
            abs_x = int(x_norm / 1000 * img_width)
            abs_y = int(y_norm / 1000 * img_height)
            
            r = 5 # radius
            draw.ellipse([(abs_x - r, abs_y - r), (abs_x + r, abs_y + r)], fill=color_rgb, outline="white")
            draw.text((abs_x + 8, abs_y - 8), label, fill=color_rgb, font=font)

        # 3. Detect 3D (Approximation)
        elif task == "detect_3d" and "box_3d" in item:
            # We only draw the center point as requested
            center_pt = _project_3d_center_to_2d(item["box_3d"], img_width, img_height)
            
            if center_pt:
                abs_x, abs_y = center_pt
                r = 6
                # Draw a different shape (e.g. rectangle) to distinguish from 2D pointing
                draw.rectangle([(abs_x - r, abs_y - r), (abs_x + r, abs_y + r)], fill=color_rgb, outline="white")
                draw.text((abs_x + 8, abs_y - 8), f"{label} (3D Center)", fill=color_rgb, font=font)

        # 4. Segmentation
        elif task == "segmentation" and "mask" in item and "box_2d" in item:
            try:
                # box_2d logic (same as above)
                box = item["box_2d"]
                ymin, xmin, ymax, xmax = box
                abs_y1 = int(ymin / 1000 * img_height)
                abs_x1 = int(xmin / 1000 * img_width)
                abs_y2 = int(ymax / 1000 * img_height)
                abs_x2 = int(xmax / 1000 * img_width)
                
                box_w = abs_x2 - abs_x1
                box_h = abs_y2 - abs_y1
                
                if box_w <= 0 or box_h <= 0: continue

                # Decode Mask
                b64_str = item["mask"].replace("data:image/png;base64,", "")
                mask_bytes = base64.b64decode(b64_str)
                mask_img = Image.open(io.BytesIO(mask_bytes))
                
                # Resize mask to bounding box size
                mask_img = mask_img.resize((box_w, box_h), Image.Resampling.BILINEAR)
                mask_arr = np.array(mask_img)
                
                # Create Colored Overlay
                # New image size of the full image
                overlay = Image.new('RGBA', (img_width, img_height), (0,0,0,0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # We need to construct a numpy array for the specific ROI to colorize it efficiently
                # Or simply iterate (slower but safer without heavier deps). 
                # Let's use a simpler PIL approach: Create a solid color block, apply mask as alpha
                
                color_block = Image.new('RGBA', (box_w, box_h), color_rgb + (150,)) # 150 is alpha transparency
                
                # Use the decoded mask (L mode usually) as the alpha mask for the color block
                # We typically need to threshold the mask from the API
                mask_l = mask_img.convert('L')
                # Binarize based on threshold
                mask_l = mask_l.point(lambda p: 255 if p > segmentation_threshold else 0)
                
                color_block.putalpha(mask_l)
                
                # Paste onto the full size transparent overlay
                overlay.paste(color_block, (abs_x1, abs_y1), color_block)
                
                # Composite overlay onto main image
                annotated_img = Image.alpha_composite(annotated_img.convert('RGBA'), overlay)
                
                # Re-init draw object because annotated_img changed
                draw = ImageDraw.Draw(annotated_img)
                
                # Draw bounding box on top
                draw.rectangle([(abs_x1, abs_y1), (abs_x2, abs_y2)], outline=color_rgb, width=2)
                draw.text((abs_x1, abs_y1 - 15), label, fill=color_rgb, font=font)

            except Exception as e:
                logger.warning(f"Failed to process mask for {label}: {e}")

    return data, annotated_img

image_path = "/home/daniel/Downloads/cupcakes.jpeg" # Replace with your image

# Example 1: Object Detection (2D)
print("--- Running 2D Detection ---")
json_data, vis_image = vision_understanding(
    media_path=image_path,
    prompt="Detect all cupcakes in the image. Label them according to their frosting color.",
    task="segmentation",
    visual=True
)

print("JSON Output:", json.dumps(json_data, indent=2))
if vis_image:
    vis_image.save("result_2d.png")
    print("Saved result_2d.png")