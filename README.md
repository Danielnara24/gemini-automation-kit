# Gemini Automation Kit

A robust, lightweight Python wrapper for the [Google Gemini API](https://ai.google.dev/gemini-api/docs) (using the official `google-genai` SDK). 

This kit simplifies building automation tools by abstracting the complexity of managing multimodal inputs, tools, and response formatting. It provides a clean interface for handling videos, PDFs, "Thinking" models, and structured JSON responses.

**Fully compatible with the Gemini 3 model family.**

## Features & Core Functions

This kit provides **three specialized functions**, each tailored to specific model generations and capabilities.

### 1. Multimodal & Structured Data (Gemini 3)
Use **`prompt_gemini_3`** for the `gemini-3-pro-preview` family. This function unlocks the newest API capabilities:
*   **Combined Modes:** Supports **Tools (Search/Code) and Structured Outputs** in a single request.
*   **Thinking Level:** Granular control (`"low"` for speed, `"high"` for deep reasoning).
*   **Media Resolution:** Control token usage (`"low"`, `"medium"`, `"high"`) for Images, PDFs, and Videos to balance cost vs. detail.

### 2. Standard Text & Multimodal (Gemini 2.5)
Use **`prompt_gemini`** for `gemini-2.5-flash` and `gemini-2.5-pro`.
*   **General Purpose:** Ideal for chat, standard text generation, and file analysis.
*   **Unified Tooling:** Use Google Search, Code Execution, and URL Context simultaneously.
*   **Simple Thinking:** Supports boolean (`True`/`False`) configuration.

### 3. Strict Structured Data (Gemini 2.5)
Use **`prompt_gemini_structured`** for `gemini-2.5-flash` and `gemini-2.5-pro`.
*   **Strict Schemas:** Enforce output formats (JSON or Enums) using Pydantic models.
*   *Limitation:* Due to API constraints on older models, this function **cannot** use Tools (Search/Code) simultaneously.

---

## Automatic Media Management

All 3 functions accept a `media_attachments` argument (a list of file paths).

Simply pass a list of local file paths. The kit automatically detects MIME types for **Videos**, **Images**, and **PDFs**.
```python
media_attachments=["image.png", "video.mp4", "document.pdf"]
```

### Inline vs. Files API
The Gemini API accepts media in two ways: **Inline** (base64 data sent directly in the prompt) or via the **Files API** (uploaded to Google's Servers).

The API has a limit on how large the "Inline" data can be. This limit fluctuates based on API load. To prevent errors, this kit uses a smart helper function (`_process_media_attachments`) to manage this for you:

1.  **Default Threshold:** The script uses a default `upload_threshold_mb` of **20MB**.
2.  **The Logic:** 
    *   If your total media size is **under** 20MB, it sends the data inline (faster, no upload).
    *   If your total media size is **over** 20MB, it automatically uploads files to the Google Files API and links them in the request.
    *   It will only upload files that are keeping the inline data size > 20MB. The time it takes to upload a file will vary depending on the file's size.

**Important Configuration:**
You can adjust the `upload_threshold_mb` argument in any prompt function.
*   **Lower it** if you receive API errors regarding "Inline data limit".
*   **Raising it** too high increases the risk of the API rejecting the request due to the inline data limit.

### Files API & Storage
When files exceed the threshold, they are stored using Google's infrastructure. Per the documentation:

> “You can use the Files API to upload and interact with media files. The Files API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours.”

### Smart Caching (Deduplication)
To save bandwidth and time, the kit calculates the **SHA-256 hash** of every local file before uploading.
*   If a file with the same hash already exists on Google's servers, the wrapper **skips the upload** and uses the existing reference.
*   This works even if you change the local file path or filename; as long as the content is identical, it won't be re-uploaded.

### Cleanup
The kit includes a utility function, `delete_all_uploads()`, which queries the Files API and deletes all active files. This is useful for freeing up space or cleaning up after a testing session.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Danielnara24/gemini-automation-kit.git
    cd gemini-automation-kit
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

You must set your Gemini API key as an environment variable before running the scripts.

**Mac/Linux:**
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

## Usage

Import the utility functions into your own Python scripts.

> **Note:** For comprehensive examples covering more use cases, refer to `gemini_2_5_examples.py` and `gemini_3_examples.py` included in this repository.

### 1. Basic Text & Google Search
Use `prompt_gemini` for standard interactions.

```python
from gemini_utils import prompt_gemini

prompt = "What are the latest specs of the Steam Deck OLED vs the ROG Ally X?"

response, tokens = prompt_gemini(
    model="gemini-2.5-flash",
    prompt=prompt,
    google_search=True,  # Enables Search Tool
    thinking=True        # Enables Thinking
)

print(response)
# Output will include inline citations [1] and a source list.
```

### 2. Mixed Media (Video, PDF, Images)
Pass any combination of supported files. The kit handles the upload/inline logic automatically.

```python
from gemini_utils import prompt_gemini

# Local files
files = [
    "./downloads/tutorial.mp4", 
    "./documents/specification.pdf"
]

response, tokens = prompt_gemini(
    model="gemini-2.5-pro",
    prompt="Compare the specifications in the PDF with the device shown in the video.",
    media_attachments=files,
    upload_threshold_mb=15.0 # Force upload if files > 15MB
)

print(response)
```

### 3. Gemini 3: Complex Workflow (Search + Code + JSON)
The `prompt_gemini_3` function allows you to combine tools (Search/Code) with Structured Outputs.

```python
from pydantic import BaseModel
from gemini_utils import prompt_gemini_3

# 1. Define the desired output structure
class CryptoRatio(BaseModel):
    btc_price: float
    eth_price: float
    ratio: float
    summary: str

# 2. Run the agent
# The model will: 
#   A. Search Google for current prices
#   B. Write/Run Python code to calculate the exact ratio
#   C. Return the result as a strict JSON object
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
Free up server storage space.

```python
from gemini_utils import delete_all_uploads

delete_all_uploads()
```

## Disclaimer

This is an unofficial open-source utility and is **not affiliated with, endorsed by, or connected to Google**.

The code is provided "as is" to help developers interact with the Gemini API more easily. Users are responsible for their own API usage, costs, and adherence to Google's Terms of Service.

## API Limits & Pricing

This tool relies on the Google Gemini API. Usage is subject to the rate limits and pricing of the tier you are using.

*   **Rate Limits:** Please consult the official documentation for the most current rate limits (RPM/TPM/RPD):  
    [**Google Gemini API Rate Limits**](https://ai.google.dev/gemini-api/docs/rate-limits)

*   **Pricing:** For details on costs associated with the paid tier:  
    [**Google Gemini API Pricing**](https://ai.google.dev/gemini-api/docs/pricing)

## License

[MIT](https://choosealicense.com/licenses/mit/)