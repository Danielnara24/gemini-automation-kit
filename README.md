# Gemini API Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/gemini-api-toolkit)](https://pypi.org/project/gemini-api-toolkit/)
![Maintenance](https://img.shields.io/badge/Maintained%3F-Yes-brightgreen.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Danielnara24/gemini-api-toolkit/pulls)

Python wrapper for the [Google Gemini API](https://ai.google.dev/gemini-api/docs) (using the official `google-genai` SDK).

## Comparison: Manually vs. Toolkit

The following example demonstrates a complex multimodal workflow:
Processing images and a PDF to generate an insurance claim decision using the `gemini-3-pro-preview` model.

**Raw SDK (Manual Implementation):**
```python
import time
import mimetypes
import pathlib
from google import genai
from google.genai import types

# 1. Setup Client (Requires v1alpha for specific Gemini 3 features)
client = genai.Client(http_options={'api_version': 'v1alpha'})

# 2. Handle Inputs
media_files = ["front_bumper.jpg", "side_panel.jpg", "police_report.pdf"]
parts = []

for path_str in media_files:
    path = pathlib.Path(path_str)
    mime_type, _ = mimetypes.guess_type(path)
    
    if mime_type == "application/pdf":
        print(f"Uploading {path}...")
        with open(path, "rb") as f:
            # Upload via Files API
            uploaded_file = client.files.upload(file=f, config={'mime_type': mime_type})
        
        # Poll for processing
        while True:
            file_meta = client.files.get(name=uploaded_file.name)
            if file_meta.state.name == "ACTIVE":
                print("File Active.")
                break
            elif file_meta.state.name == "FAILED":
                raise Exception("File upload failed")
            time.sleep(2)
            
        parts.append(types.Part(
            file_data=types.FileData(file_uri=uploaded_file.uri, mime_type=mime_type),
            media_resolution={"level": "media_resolution_high"}
        ))
    else:
        # Send Images Inline
        parts.append(types.Part(
            inline_data=types.Blob(
                data=path.read_bytes(), 
                mime_type=mime_type
            ),
            media_resolution={"level": "media_resolution_high"}
        ))

# 3. Add Prompt
parts.append(types.Part(text="Analyze these images and the report. Determine if the insurance claim should be approved and explain why."))

# 4. Configure Gemini 3 Specifics
generate_config = types.GenerateContentConfig(
    temperature=1.0, # Recommended default for Gem 3
    thinking_config=types.ThinkingConfig(
        thinking_level="HIGH",
        include_thoughts=True
    )
)

# 5. Generate
print("Generating...")
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents=[types.Content(parts=parts)],
    config=generate_config
)

print(response.text)
```


**With Gemini API Toolkit:**
```python
from gemini_kit import prompt_gemini_3

# 1. Define Inputs
media = ["front_bumper.jpg", "side_panel.jpg", "police_report.pdf"]
prompt_text = "Analyze these images and the report. Determine if the insurance claim should be approved and explain why."

# 2. Call the Function
result, tokens = prompt_gemini_3(
    model="gemini-3-pro-preview",
    prompt=prompt_text,
    media_attachments=media,
    media_resolution="high", 
    thinking_level="high"
)

print(result)
```

## Installation

**Using Pip:**
```bash
pip install gemini-api-toolkit
```

**From Source:**
```bash
git clone https://github.com/Danielnara24/gemini-api-toolkit.git
cd gemini-api-toolkit
pip install -e .
```

## Usage

### 1. Basic Text & Google Search
```python
from gemini_kit import prompt_gemini

prompt = "What are the latest specs of the Steam Deck OLED vs the ROG Ally X?"

response, tokens = prompt_gemini(
    model="gemini-2.5-flash",
    prompt=prompt,
    google_search=True,  # Enables Search Tool
    thinking=True        # Enables Thinking
)

print(response)
```

### 2. Mixed Media (Video, PDF, Images, Audio)
Pass local file paths or YouTube URLs. The kit handles upload/inline logic automatically.
```python
from gemini_kit import prompt_gemini

files = ["./downloads/tutorial.mp4", "./documents/specification.pdf"]

response, tokens = prompt_gemini(
    model="gemini-2.5-pro",
    prompt="Compare the specifications in the PDF with the device shown in the video.",
    media_attachments=files
)

print(response)
```

### 3. Structured Output (Pydantic)
Enforce a JSON schema on the output. 
*Note: In Gemini 2.5, you cannot combine Structured Output with Tools (Search/Code).*

```python
from pydantic import BaseModel
from gemini_kit import prompt_gemini

class MovieIdea(BaseModel):
    title: str
    logline: str
    estimated_budget: int

response_obj, tokens = prompt_gemini(
    model="gemini-2.5-flash",
    prompt="Generate a movie idea about a robot learning to paint.",
    response_schema=MovieIdea
)

# Returns a MovieIdea object directly
print(f"Title: {response_obj.title}")
print(f"Budget: ${response_obj.estimated_budget}")
```

### 4. Gemini 3: Search + Code + JSON
```python
from pydantic import BaseModel
from gemini_kit import prompt_gemini_3

class CryptoRatio(BaseModel):
    btc_price: float
    eth_price: float
    ratio: float
    summary: str

response_obj, tokens = prompt_gemini_3(
    prompt="Find current BTC and ETH prices and calculate the ETH/BTC ratio.",
    response_schema=CryptoRatio, 
    google_search=True,
    code_execution=True,
    thinking_level="high"
)

print(f"Ratio: {response_obj.ratio} | Summary: {response_obj.summary}")
```

### 5. Cleanup
Free up server storage space (deletes files uploaded via Files API).
```python
from gemini_kit import delete_all_uploads

delete_all_uploads()
```

> [!TIP]
> The `examples/` folder in this repository contains scripts demonstrating specific use cases.

## Arguments for prompting functions

*   **model:** The name of the Gemini model to use (e.g., "gemini-2.5-flash", "gemini-3-pro-preview").
*   **prompt:** The text instruction sent to the model.
*   **response_schema:** Pydantic model or Enum class to enforce structured JSON output. (Note: In `prompt_gemini`, this disables tools).
*   **media_attachments:** List of file paths (audio, images, videos, PDFs) or YouTube URLs to analyze.
*   **upload_threshold_mb:** Files larger than this (in MB) are uploaded via Files API; smaller are sent inline.
*   **thinking_level:** Controls reasoning depth for Gemini 3 ("low" or "high").
*   **thinking:** Boolean to enable/disable the thinking process for Gemini 2.5.
*   **media_resolution:** Sets token usage/quality for inputs ("low", "medium", "high") for Gemini 3.
*   **temperature:** Controls output randomness (0.0 to 2.0).
*   **google_search:** Boolean to enable Grounding with Google Search.
*   **code_execution:** Boolean to enable the Python code interpreter tool.
*   **url_context:** Boolean to enable the model to read/process content from URLs in the prompt.
*   **max_retries:** Number of times to retry the API call if it fails. 0 by default.

## Spatial Understanding

The toolkit provides dedicated functions for 2D detection (bounding boxes), pointing, and segmentation generation. These functions return both the raw JSON data and a visualized Pillow image.

### 1. 2D Object Detection
Detect objects with bounding boxes using any Gemini model.

```python
from gemini_kit import detect_2d

# Returns JSON data and a PIL Image with drawn boxes
json_data, visual_image = detect_2d(
    model="gemini-2.5-pro", 
    prompt="Detect all faces in the image. Label what they are wearing.",
    image_path="street.jpg",
    visual=True
)

visual_image.show()
```

![2D Detection Example](https://github.com/user-attachments/assets/40c440ac-abf1-4265-bb31-b43686007f2a)

### 2. Pointing
Identify the precise location of objects (y, x coordinates).

```python
from gemini_kit import pointing

json_data, visual_image = pointing(
    model="gemini-3-pro-preview", 
    prompt="Label each part of the motherboard in the image.",
    image_path="motherboard.png",
    visual=True
)

visual_image.show()
```

![Pointing Example](https://github.com/user-attachments/assets/7085ab92-d46d-4ef8-ad7f-5f7803ada0ad)

### 3. Segmentation
Generate pixel-level masks for objects.  
*Note: Only supported on Gemini 2.5 models.*

```python
from gemini_kit import segmentation

# visual=True returns a combined overlay image
# output_path saves individual mask files to disk
json_data, visual_image = segmentation(
    model="gemini-2.5-pro", 
    prompt="Segment all cupcakes in the image, label 'sprinkles' or 'no sprinkles'",
    image_path="cupcakes.jpeg",
    visual=True,
    output_path="output_samples" 
)

visual_image.show()
```

![Segmentation Example](https://github.com/user-attachments/assets/a43a0cdd-4b21-4427-ab4a-71189d6f7628)

## Arguments for Spatial Understanding Functions

*   **model:** The name of the Gemini model to use (e.g., "gemini-2.5-flash", "gemini-3-pro-preview").
*   **prompt:** The text instruction sent to the model.
*   **image_path:** Local path or URL of the image to use.
*   **visual:** If True, returns a PIL Image with the visualization.
*   **output_path:** The path to save PIL images or masks and overlays. Won't save if not specified.
*   **temperature:** Controls output randomness (0.0 to 2.0). 0.5 Recommended.
*   **max_retries:** Number of times to retry the API call if it fails. 0 by default.

## Disclaimer

> This is an unofficial open-source utility and is **not affiliated with, endorsed by, or connected to Google**.
The code is provided "as is" to help developers interact with the Gemini API more easily. Users are responsible for their own API usage, costs, and adherence to Google's Terms of Service.