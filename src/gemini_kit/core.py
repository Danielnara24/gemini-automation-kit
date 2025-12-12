import requests
from PIL import Image
import json
import os
import logging
import enum
import time
import google.genai as genai
from google.genai import types
from typing import Any, List, Optional, Union, Dict

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Import helper functions
from .utils import (
    _add_citations,
    _process_media_attachments,
    _clean_json_markdown,
    _visualize_2d,
    _visualize_segmentation,
    _save_segmentation_artifacts,
    _visualize_points,
    _generate_processed_filename
)

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
                    logger.warning(f"API error: {e}. Retrying...")
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
                full_response = _add_citations(response)
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
                    logger.warning(f"API error: {e}. Retrying...")
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
            full_response = _add_citations(response)
        else:
            full_response = response.text

        return full_response, input_token_count

    except Exception as e:
        return f"An error occurred during content generation: {e}", 0
    
def detect_2d(
    model: str = "gemini-2.5-flash-lite",
    prompt: str = "Detect all items.",
    image_path: str = None,
    visual: bool = False,
    output_path: str = None,
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
    try:
        # Check if input is a URL
        if str(image_path).startswith(('http://', 'https://')):
            img_resp = requests.get(image_path)
            img_resp.raise_for_status()
            
            # Attempt to get mime type from headers, fallback to jpeg
            mime_type = img_resp.headers.get('Content-Type', 'image/jpeg')
            
            media_parts = [types.Part.from_bytes(data=img_resp.content, mime_type=mime_type)]
        else:
            # Fallback to standard processing for local files/YouTube
            media_parts = _process_media_attachments(client, [image_path], inline_limit_mb=20.0)
            if isinstance(media_parts, str): 
                return f"Media Error: {media_parts}", None
    except Exception as e:
        return f"Media Error: {str(e)}", None

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
            logger.warning(f"API error: {e}. Retrying...")
            time.sleep(1)

    # 5. Parse Output
    try:
        clean_json = _clean_json_markdown(response_text)
        json_data = json.loads(clean_json)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response: {response_text}")
        return f"Error: Could not parse model response. Raw: {response_text}", None

    # 6. Visualization and Saving
    annotated_image = None
    
    # We allow generation if visual is requested OR if we need to save to disk
    if visual or output_path:
        annotated_image = _visualize_2d(image_path, json_data)

    if output_path and annotated_image:
        try:
            os.makedirs(output_path, exist_ok=True)
            filename = _generate_processed_filename(image_path)
            save_path = os.path.join(output_path, filename)
            annotated_image.save(save_path)
            logger.info(f"Saved 2D detection image to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save output image: {e}")

    # Only return the image object if visual was explicitly True
    return json_data, annotated_image if visual else None

def segmentation(
    model: str = "gemini-2.5-flash",
    prompt: str = "Segment all items.",
    image_path: str = None,
    visual: bool = False,
    output_path: str = None,
    temperature: float = 0.5,
    max_retries: int = 0
) -> tuple[Union[List[Dict[str, Any]], str], Optional[Image.Image]]:
    """
    Performs object segmentation on an image using Gemini.
    
    Args:
        model (str): The model to use (must be Gemini 2.5 or newer).
        prompt (str): Instructions on what to segment.
        image_path (str): Local path or URL to the image.
        visual (bool): If True, returns a PIL Image with masks and boxes drawn.
        output_path (str): If provided, saves individual masks and overlays to this directory.
        temperature (float): Model temperature.
        max_retries (int): Retry attempts.

    Returns:
        tuple: (JSON Data [List of Dicts], PIL Image [or None])
    """
    if not image_path:
        return "Error: image_path is required.", None

    # Handle client initialization
    if "gemini-3" in model:
         client = genai.Client(http_options={'api_version': 'v1alpha'})
    else:
         client = genai.Client()

    # 1. System Instructions (Strict format for segmentation)
    system_instruction = """
    Output a JSON list of segmentation masks where each entry contains:
    1. The 2D bounding box in the key "box_2d" (normalized 0-1000 [ymin, xmin, ymax, xmax]).
    2. The segmentation mask in key "mask" (Base64 encoded PNG).
    3. The text label in the key "label". 
    Use descriptive labels. Do not return markdown code fencing.
    """

    # 2. Configure Thinking (Critical for valid JSON/Masks)
    if "gemini-2.5-pro" in model:
        # 2.5 Pro uses include_thoughts
        thinking_config = types.ThinkingConfig(include_thoughts=False)
    elif "gemini-3-pro-preview" in model:
        return "Error: gemini-3-pro-preview does not support segmentation. Try other models instead.", None
    else:
        # Standard Flash models use budget
        thinking_config = types.ThinkingConfig(thinking_budget=0)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=temperature,
        system_instruction=system_instruction,
        thinking_config=thinking_config
    )

    # 3. Process Media
    try:
        if str(image_path).startswith(('http://', 'https://')):
            img_resp = requests.get(image_path)
            img_resp.raise_for_status()
            img_data = img_resp.content
            mime_type = "image/jpeg" # Default assumption
        else:
            with open(image_path, 'rb') as f:
                img_data = f.read()
            mime_type = "image/jpeg" # Default assumption
            
        image_part = types.Part.from_bytes(data=img_data, mime_type=mime_type)
    except Exception as e:
        return f"Media Error: {str(e)}", None

    # 4. Generate Content
    contents = [image_part, prompt]
    
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
                logger.error(f"Segmentation failed: {e}")
                return f"Error: {e}", None
            logger.warning(f"API error: {e}. Retrying...")
            time.sleep(1)

    # 5. Parse Output
    try:
        clean_json = _clean_json_markdown(response_text)
        json_data = json.loads(clean_json)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response: {response_text}")
        return f"Error: Could not parse model response. Raw: {response_text}", None

    # 6. Visualization & Saving
    annotated_image = None
    
    # Generate the main combined image if requested or if saving
    if visual or output_path:
        annotated_image = _visualize_segmentation(image_path, json_data)

    if output_path:
        try:
            # Create main output directory
            os.makedirs(output_path, exist_ok=True)

            # 1. Save individual artifacts to 'masks_and_overlays' subfolder
            artifacts_dir = os.path.join(output_path, "masks_and_overlays")
            _save_segmentation_artifacts(image_path, json_data, artifacts_dir)

            # 2. Save the combined annotated image to output_path
            if annotated_image:
                filename = _generate_processed_filename(image_path)
                save_path = os.path.join(output_path, filename)
                annotated_image.save(save_path)
                logger.info(f"Saved combined segmentation image to: {save_path}")
                
        except Exception as e:
            logger.error(f"Failed to save segmentation outputs: {e}")

    return json_data, annotated_image if visual else None

def pointing(
    model: str = "gemini-2.5-flash",
    prompt: str = "Point to the main items in this image.",
    image_path: str = None,
    context: List[Any] = None,
    visual: bool = False,
    output_path: str = None,
    temperature: float = 0.5,
    max_retries: int = 0
) -> tuple[Union[List[Dict[str, Any]], str], Optional[Image.Image]]:
    """
    Performs pointing or multiview correspondence.
    
    Args:
        model (str): The model to use.
        prompt (str): Instructions on what to point to.
        image_path (str): Local path or URL to the image.
        context: A list of previous images (paths/urls) or text/JSON strings to provide history.
        visual (bool): If True, returns a PIL Image with points drawn.
        temperature (float): Model temperature (docs suggest >0, e.g., 0.5).
        max_retries (int): Retry attempts.

    Returns:
        tuple: (JSON Data [List of Dicts], PIL Image [or None])
    """
    if not image_path:
        return "Error: image_path is required.", None

    if "gemini-3" in model:
         client = genai.Client(http_options={'api_version': 'v1alpha'})
    else:
         client = genai.Client()

    # 1. System Instructions
    # We relax the instructions slightly to allow "in_frame" boolean for multiview
    system_instruction = """
    Output a JSON list of objects.
    Format: [{"point": [y, x], "label": "string", "in_frame": boolean (optional)}, ...]
    1. "point": [y, x] normalized 0-1000.
    2. "in_frame": Set to false if the object from context is not visible.
    3. Do not return markdown code fencing.
    """

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=temperature,
        system_instruction=system_instruction
    )

    # 2. Process Media Helper
    def load_media(path):
        if str(path).startswith(('http://', 'https://')):
            resp = requests.get(path)
            resp.raise_for_status()
            return types.Part.from_bytes(data=resp.content, mime_type="image/jpeg")
        elif os.path.exists(path):
            with open(path, 'rb') as f:
                return types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
        return None

    # 3. Build Content List
    contents = []

    # Add Context (Previous images or text)
    if context:
        for item in context:
            # If it looks like an image path/url, load it
            if isinstance(item, str) and (item.endswith(('.jpg', '.png', '.jpeg')) or item.startswith('http')):
                 part = load_media(item)
                 if part: contents.append(part)
                 else: contents.append(item) # Treat as text if load fails
            # If it looks like JSON string or plain text
            else:
                contents.append(str(item))

    # Add Prompt
    contents.append(prompt)

    # Add Target Image
    target_image_part = load_media(image_path)
    if not target_image_part: return f"Media Error: Could not load {image_path}", None
    contents.append(target_image_part)
    
    # 4. Generate Content
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
                logger.error(f"Pointing failed: {e}")
                return f"Error: {e}", None
            logger.warning(f"API error: {e}. Retrying...")
            time.sleep(1)

    # 5. Parse Output
    try:
        clean_json = _clean_json_markdown(response_text)
        json_data = json.loads(clean_json)
        if isinstance(json_data, dict): json_data = [json_data]
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {response_text}")
        return f"Error: Invalid JSON. Raw: {response_text}", None

    # 6. Visualization and Saving
    annotated_image = None
    
    if visual or output_path:
        annotated_image = _visualize_points(image_path, json_data)
        
    if output_path and annotated_image:
        try:
            os.makedirs(output_path, exist_ok=True)
            filename = _generate_processed_filename(image_path)
            save_path = os.path.join(output_path, filename)
            annotated_image.save(save_path)
            logger.info(f"Saved pointing image to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save output image: {e}")

    return json_data, annotated_image if visual else None
