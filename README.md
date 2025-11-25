# Gemini Automation Kit

A robust, lightweight Python wrapper for the [Google Gemini API](https://ai.google.dev/) (using the official `google-genai` SDK). 

This kit simplifies building automation tools by abstracting the complexity of managing multimodal inputs, tools, and response formatting. It provides a clean interface for handling videos, PDFs, "Thinking" models, and—crucially—parsing structured JSON responses.

## Features

*   **Multimodal Simplified:** Pass local file paths for **Videos** or **PDFs** directly into the prompt function. The script automatically handles MIME types and file data.
*   **Unified Tooling:** The main `prompt_gemini` function allows you to use **Google Search**, **Code Execution**, and **URL Context** simultaneously. The model can look up info, read a URL, and write/run code to answer a single prompt.
*   **Structured Outputs:** A dedicated function (`prompt_gemini_structured`) enforces strict output formats (JSON or Enums) using Pydantic models. 
    *   *Note: This is separated into its own function because the Gemini API does not currently support using Tools (like Search/Code) and strict JSON Schemas in the same request.*
*   **Automatic Citations:** Automatically parses Google Search grounding metadata to insert inline markdown citations (e.g., `[1](url)`) and a formatted source list at the end of the text response.
*   **Thinking Configuration:** Native support for enabling/disabling the "Thinking" process (for reasoning models like `gemini-2.5-pro`).

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
export GOOGLE_API_KEY="your_actual_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_API_KEY="your_actual_api_key_here"
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your_actual_api_key_here"
```

## API Limits & Pricing

This tool relies on the Google Gemini API. Usage is subject to the rate limits and pricing of the tier you are using.

*   **Rate Limits:** Please consult the official documentation for the most current rate limits (RPM/TPM/RPD):  
    [**Google Gemini API Rate Limits**](https://ai.google.dev/gemini-api/docs/rate-limits)

*   **Pricing:** For details on costs associated with the paid tier:  
    [**Google Gemini API Pricing**](https://ai.google.dev/gemini-api/docs/pricing)

## Usage

Import the utility functions into your own Python scripts:

### 1. Basic Text & Google Search
Use `prompt_gemini` for standard interactions. Set `google_search=True` to enable live web grounding. The script will automatically format the citations.

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

### 2. Multimodal: Video & PDF
You don't need to manually upload files via the API; just pass the local file path.

**Video Example:**
```python
video_path = "./downloads/tutorial.mp4"

response, tokens = prompt_gemini(
    model="gemini-2.5-flash",
    prompt="Summarize the steps shown in this video and extract the code used.",
    video_attachment=video_path,
    code_execution=True # Allows the model to run code
)
```

**PDF Example:**
```python
pdf_path = "./documents/financial_report.pdf"

response, tokens = prompt_gemini(
    model="gemini-2.5-pro",
    prompt="Analyze the risk factors mentioned in this document.",
    pdf_attachment=pdf_path
)
```

### 3. Structured Output (JSON)
Use `prompt_gemini_structured` when you need the LLM to return a specific data structure for your code to use. This uses Pydantic to define the schema.

```python
from pydantic import BaseModel
from gemini_utils import prompt_gemini_structured

# 1. Define your schema
class Movie(BaseModel):
    title: str
    director: str
    year: int
    genre: str

# 2. Call the function with the schema
response_data, tokens = prompt_gemini_structured(
    model="gemini-2.5-flash",
    prompt="List 3 classic 80s sci-fi movies",
    response_schema=list[Movie] # We expect a list of movies
)

# 3. Use the data as real Python objects
for movie in response_data:
    print(f"{movie.title} was directed by {movie.director}")
```

## Disclaimer

This is an unofficial open-source utility and is **not affiliated with, endorsed by, or connected to Google**.

The code is provided "as is" to help developers interact with the Gemini API more easily. Users are responsible for their own API usage, costs, and adherence to Google's Terms of Service.

## License

[MIT](https://choosealicense.com/licenses/mit/)