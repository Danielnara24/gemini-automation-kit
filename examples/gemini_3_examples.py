import os
from typing import List
from pydantic import BaseModel, Field
from gemini_kit import prompt_gemini_3, delete_all_uploads, check_api_key
import logging
# Configure logging to see info messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# Pydantic Schemas for Structured Examples
# ==========================================

class CityStats(BaseModel):
    city_name: str
    population: int
    mayor: str
    top_attractions: List[str]

class CalculationResult(BaseModel):
    operation_description: str
    steps_taken: List[str]
    final_numeric_result: float

class VideoEvent(BaseModel):
    timestamp: str
    description: str
    significance: str

class VideoAnalysis(BaseModel):
    main_topic: str
    events: List[VideoEvent]
    conclusion: str

# ==========================================
# Use Case Examples
# ==========================================

def example_url_context_low_thinking():
    """
    1. Simple text use case with URL context. Low thinking level.
    """
    print("--- Running Example 1: URL Context + Low Thinking ---")
    
    # URL Context allows the model to 'read' the page. 
    # Low thinking is faster, suitable for summarization tasks where deep reasoning isn't required.
    prompt = "Read this website and provide a 3-bullet point summary of the top news stories: https://news.google.com"
    
    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt=prompt,
        url_context=True,
        thinking_level="low" 
    )
    
    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_search_structured():
    """
    2. Simple structured output case with google search.
    """
    print("--- Running Example 2: Search + Structured Output ---")
    
    # The model searches Google for real-time info, then forces the output into the Pydantic JSON schema.
    prompt = "Find the latest statistics for Tokyo, Japan."
    
    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt=prompt,
        google_search=True,
        response_schema=CityStats
    )
    
    print(f"Input Tokens: {tokens}")
    print(f"City: {response.city_name}")
    print(f"Population: {response.population}")
    print(f"Mayor: {response.mayor}")
    print(f"Attractions: {', '.join(response.top_attractions)}")


def example_code_structured():
    """
    3. Simple structured output case with code execution.
    """
    print("--- Running Example 3: Code Execution + Structured Output ---")
    
    # The model writes and runs Python code to solve the math, then formats the result.
    prompt = "Calculate the sum of the first 50 prime numbers."
    
    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt=prompt,
        code_execution=True,
        response_schema=CalculationResult
    )
    
    print(f"Input Tokens: {tokens}")
    print(f"Operation: {response.operation_description}")
    print(f"Result: {response.final_numeric_result}")


def example_search_code_low_thinking():
    """
    4. Google search example with code execution. Low thinking level.
    """
    print("--- Running Example 4: Search + Code + Low Thinking ---")
    
    # Combines finding data (Search) and processing it (Code), but uses Low thinking for speed.
    prompt = (
        "Search for the height of the Empire State Building and the Eiffel Tower in meters. "
        "Then write code to calculate the difference in height."
    )
    
    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt=prompt,
        google_search=True,
        code_execution=True,
        thinking_level="low"
    )
    
    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_video_structured_placeholder():
    """
    5. Video path placeholder example with structured output.
    """
    print("--- Running Example 5: Video + Structured Output ---")
    
    # REPLACE WITH YOUR ACTUAL FILE PATH
    video_path = "media_samples/meeting_recording.mp4"
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}. Please update the path.")
        return

    prompt = "Analyze this video and extract key events."

    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt=prompt,
        media_attachments=[video_path],
        response_schema=VideoAnalysis,
        media_resolution="high"
    )

    print(f"Input Tokens: {tokens}")
    print(f"Topic: {response.main_topic}")
    for event in response.events:
        print(f"[{event.timestamp}] {event.description}")


def example_multi_pdf_low_res():
    """
    6. 3 Pdf paths placeholders, prompting to make a summary of all 3. Low media resolution.
    """
    print("--- Running Example 6: Multi-PDF Summary (Low Res) ---")
    
    # REPLACE WITH YOUR ACTUAL FILE PATHS
    pdf_paths = [
        "media_samples/doc1.pdf",
        "media_samples/doc2.pdf",
        "media_samples/doc3.pdf"
    ]

    missing = [p for p in pdf_paths if not os.path.exists(p)]
    if missing:
        print(f"Files not found: {missing}. Please update paths.")
        return

    # 'Low' resolution tokens are cheaper/faster, good for text-heavy PDFs where 
    # visual fidelity (charts/images) isn't the priority.
    prompt = "Summarize the connection between these three documents."

    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt=prompt,
        media_attachments=pdf_paths,
        media_resolution="low"
    )

    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_image_sequence():
    """
    7. 5 image paths placeholders, prompting to explain the image sequence.
    """
    print("--- Running Example 7: 5 Image Sequence ---")
    
    # REPLACE WITH YOUR ACTUAL FILE PATHS
    image_paths = [
        "media_samples/seq_1.jpg",
        "media_samples/seq_2.jpg",
        "media_samples/seq_3.jpg",
        "media_samples/seq_4.jpg",
        "media_samples/seq_5.jpg"
    ]

    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        print(f"Files not found: {missing}. Please update paths.")
        return

    prompt = "Describe the narrative flow shown in these 5 images."

    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt=prompt,
        media_attachments=image_paths
        # Default resolution is "medium" if not specified
    )

    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_mixed_media_low_res():
    """
    8. Video, PDF and 2 image placeholders, prompting to explain how they relate. Low media resolution.
    """
    print("--- Running Example 8: Mixed Media Context (Low Res) ---")
    
    # REPLACE WITH YOUR ACTUAL FILE PATHS
    media_files = [
        "media_samples/video.mp4",
        "media_samples/file.pdf",
        "media_samples/image.png",
        "media_samples/image.png"
    ]

    missing = [p for p in media_files if not os.path.exists(p)]
    if missing:
        print(f"Files not found: {missing}. Please update paths.")
        return

    prompt = "Explain how the images and the PDF content align with the content in the video."

    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt=prompt,
        media_attachments=media_files,
        media_resolution="low" 
    )

    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_youtube_simple_url():
    """
    9. Simple YouTube URL.
    """
    print("--- Running Example 9: YouTube URL ---")
    
    media_files = ["https://youtu.be/chr2I7CZTfk?si=wMwiIRcj0bP6uStj"]

    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt="Make a summary of this video",
        media_attachments=media_files
    )

    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_youtube_timestamps_summary():
    """
    10. Two YouTube videos: One using "MM:SS" timestamps, one using seconds (int/str).
    """
    print("--- Running Example 10: YouTube Segments (Mixed Timestamps) ---")
    
    # We define specific segments of interest.
    # Video 1: Uses MM:SS format.
    # Video 2: Uses raw seconds (integer).
    
    media_files = [
        {
            "url": "https://youtu.be/6Q-ESEmDf4Q?si=04n8U1IQ00LAS7xi",
            "start": 4,   # 4 seconds (0:04 mark)
            "end": 200    # 200 seconds (3:20 mark)
        },
        {
            "url": "https://youtu.be/aR20FWCCjAs?si=_qs59P-bdOCmrRJX",
            "start": "18:49",
            "end": "25:13"
        }
    ]

    prompt = "Compare the topics discussed in these two specific video segments."

    response, tokens = prompt_gemini_3(
        model="gemini-3-pro-preview",
        prompt=prompt,
        media_attachments=media_files,
        thinking_level="low" # Keeping it fast for this example
    )

    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


# ==========================================
# Main Execution Block
# ==========================================

if __name__ == "__main__":
    # Ensure API Key is set
    if check_api_key():
        
        # Uncomment the function you wish to run:

        # example_url_context_low_thinking()
        # example_search_structured()
        # example_code_structured()
        # example_search_code_low_thinking()
        # example_youtube_simple_url()
        # example_youtube_timestamps_summary()

        # NOTE: For the following, ensure you create the files or update paths:
        # example_video_structured_placeholder()
        # example_multi_pdf_low_res()
        # example_image_sequence()
        # example_mixed_media_low_res()

        # Utility to clean up cloud storage after testing heavy media
        # delete_all_uploads()
        pass