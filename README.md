# Gemini API Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/gemini-api-toolkit)](https://pypi.org/project/gemini-api-toolkit/)
![Maintenance](https://img.shields.io/badge/Maintained%3F-Yes-brightgreen.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

Python wrapper for the [Google Gemini API](https://ai.google.dev/gemini-api/docs) (using the official `google-genai` SDK).

## Comparison: Manually vs. Toolkit

The following example demonstrates a complex multimodal workflow:
Processing images and a PDF to generate a structured insurance claim decision using the `gemini-3-pro-preview` model.

**Raw SDK (Manual Implementation):**
```python
import time
import mimetypes
import pathlib
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Define Structure
class ClaimDecision(BaseModel):
    decision: str = Field(description="Approved or Denied")
    estimated_cost: float
    reasoning: str
    requires_human_review: bool

# 2. Setup Client
client = genai.Client(http_options={'api_version': 'v1alpha'})

# 3. Handle Inputs
media_files = ["front_bumper.jpg", "side_panel.jpg", "police_report.pdf"]
parts = []

for path_str in media_files:
    path = pathlib.Path(path_str)
    mime_type, _ = mimetypes.guess_type(path)
    
    if mime_type == "application/pdf":
        print(f"Uploading {path}...")
        with open(path, "rb") as f:
            # Upload
            uploaded_file = client.files.upload(file=f, config={'mime_type': mime_type})
        
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
            media_resolution={"level": "media_resolution_high"} # PDF spec from docs
        ))
    else:
        # Inline Image
        parts.append(types.Part(
            inline_data=types.Blob(
                data=path.read_bytes(), 
                mime_type=mime_type
            ),
            media_resolution={"level": "media_resolution_high"} # Image spec from docs
        ))

# 4. Add Prompt
parts.append(types.Part(text="Analyze these images and the report to determine if the insurance claim should be approved."))

# 5. Configure Gemini 3 Specifics
generate_config = types.GenerateContentConfig(
    temperature=1.0, # Recommended default for Gem 3
    response_mime_type="application/json",
    response_schema=ClaimDecision,
    thinking_config=types.ThinkingConfig(
        thinking_level="HIGH",
        include_thoughts=True
    )
)

# 6. Generate
print("Generating...")
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents=[types.Content(parts=parts)],
    config=generate_config
)

# 7. Parse Result
if response.parsed:
    print(response.parsed)
else:
    print(response.text)
```


**With Gemini API Toolkit:**
```python
from pydantic import BaseModel, Field
from gemini_kit import prompt_gemini_3

# 1. Define Structure
class ClaimDecision(BaseModel):
    decision: str = Field(description="Approved or Denied")
    estimated_cost: float
    reasoning: str
    requires_human_review: bool

# 2. Define Inputs
media = ["front_bumper.jpg", "side_panel.jpg", "police_report.pdf"]
prompt_text = "Analyze these images and the report to determine if the insurance claim should be approved."

# 3. Call the Function
result, tokens = prompt_gemini_3(
    model="gemini-3-pro-preview",
    prompt=prompt_text,
    media_attachments=media,
    response_schema=ClaimDecision,
    media_resolution="high", 
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

### 2. Mixed Media (Video, PDF, Images)
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

### 3. Gemini 3: Search + Code + JSON
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

### 4. Cleanup
Free up server storage space (deletes files uploaded via Files API).
```python
from gemini_kit import delete_all_uploads

delete_all_uploads()
```

## Arguments

*   **model:** The name of the Gemini model to use (e.g., "gemini-2.5-flash", "gemini-3-pro-preview").
*   **prompt:** The text instruction sent to the model.
*   **response_schema:** Pydantic model or Enum class to enforce structured JSON output.
*   **media_attachments:** List of file paths (images, videos, PDFs) or YouTube URLs to analyze.
*   **upload_threshold_mb:** Files larger than this (in MB) are uploaded via Files API; smaller are sent inline.
*   **thinking_level:** Controls reasoning depth for Gemini 3 ("low" or "high").
*   **thinking:** Boolean to enable/disable the thinking process for Gemini 2.5.
*   **media_resolution:** Sets token usage/quality for inputs ("low", "medium", "high") for Gemini 3.
*   **temperature:** Controls output randomness (0.0 to 2.0).
*   **google_search:** Boolean to enable Grounding with Google Search.
*   **code_execution:** Boolean to enable the Python code interpreter tool.
*   **url_context:** Boolean to enable the model to read/process content from URLs in the prompt.

## Disclaimer

> This is an unofficial open-source utility and is **not affiliated with, endorsed by, or connected to Google**.
The code is provided "as is" to help developers interact with the Gemini API more easily. Users are responsible for their own API usage, costs, and adherence to Google's Terms of Service.