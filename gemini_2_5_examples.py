import os
from typing import List, Optional
from pydantic import BaseModel, Field
from gemini_utils import prompt_gemini, prompt_gemini_structured, delete_all_uploads, check_api_key

# ==========================================
# Pydantic Schemas for Structured Examples
# ==========================================

class Character(BaseModel):
    name: str
    role: str
    description: str

class MovieAnalysis(BaseModel):
    title: str
    genre: str
    characters: List[Character]
    estimated_budget_range: str
    marketing_slogan: str

class VideoSegment(BaseModel):
    timestamp_start: str
    timestamp_end: str
    activity_description: str
    mood: str

class VideoLog(BaseModel):
    summary: str
    segments: List[VideoSegment]

# ==========================================
# Use Case Examples
# ==========================================

def example_url_context():
    """
    1. Simple text use case with URL context. 
    Model: gemini-2.5-flash
    """
    print("--- Running Example 1: URL Context (2.5 Flash) ---")
    
    # Note: URL context allows the model to read the content of the provided link in the prompt.
    prompt = "Read this website and provide a 3-bullet point summary of the top news stories: https://news.google.com"
    
    response, tokens = prompt_gemini(
        model="gemini-2.5-flash",
        prompt=prompt,
        url_context=True,
    )
    
    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_structured_no_tools():
    """
    2. Simple structured output case with no tools. 
    Model: gemini-2.5-pro
    """
    print("--- Running Example 2: Structured Output (2.5 Pro) ---")
    
    prompt = "Invent a sci-fi movie concept about time travel in the year 3000."
    
    response, tokens = prompt_gemini_structured(
        model="gemini-2.5-pro",
        prompt=prompt,
        response_schema=MovieAnalysis,
    )
    
    print(f"Input Tokens: {tokens}")
    # Response is already a parsed Pydantic object
    print(f"Movie Title: {response.title}")
    print(f"Slogan: {response.marketing_slogan}")
    print(f"Characters: {len(response.characters)}")
    for char in response.characters:
        print(f" - {char.name}: {char.role}")


def example_search_and_code():
    """
    3. Google search example with code execution. 
    Model: gemini-2.5-flash
    """
    print("--- Running Example 3: Search + Code Execution (2.5 Flash) ---")
    
    # This requires the model to search for current data, then write python code to calculate the answer.
    prompt = (
        "Search for the population of the top 3 most populous cities in Japan as of 2023/2024. "
        "Then, use code to calculate the average population of these 3 cities."
    )
    
    response, tokens = prompt_gemini(
        model="gemini-2.5-flash",
        prompt=prompt,
        google_search=True,
        code_execution=True
    )
    
    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_video_structured():
    """
    4. Video path placeholder example with structured output. 
    Model: gemini-2.5-pro
    """
    print("--- Running Example 4: Video Analysis (2.5 Pro) ---")
    
    # REPLACE WITH YOUR ACTUAL FILE PATH
    video_path = "media_samples/sample_video.mp4" 
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}. Please update the path in the function.")
        return

    prompt = "Analyze this video and log the specific events happening."

    response, tokens = prompt_gemini_structured(
        model="gemini-2.5-pro",
        prompt=prompt,
        media_attachments=[video_path],
        response_schema=VideoLog
    )

    print(f"Input Tokens: {tokens}")
    print(f"Video Summary: {response.summary}")
    for seg in response.segments:
        print(f"[{seg.timestamp_start} - {seg.timestamp_end}] {seg.activity_description} ({seg.mood})")


def example_multi_pdf_summary():
    """
    5. 3 Pdf paths placeholders, prompting to make a summary of all 3. 
    Model: gemini-2.5-pro
    """
    print("--- Running Example 5: Multi-PDF Summary (2.5 Pro) ---")
    
    # REPLACE WITH YOUR ACTUAL FILE PATHS
    pdf_paths = [
        "media_samples/report_q1.pdf",
        "media_samples/report_q2.pdf",
        "media_samples/report_q3.pdf"
    ]

    # Verify existence
    missing = [p for p in pdf_paths if not os.path.exists(p)]
    if missing:
        print(f"Files not found: {missing}. Please update paths in the function.")
        return

    prompt = "Read these three documents. Provide a combined summary of all 3 combined."

    response, tokens = prompt_gemini(
        model="gemini-2.5-pro",
        prompt=prompt,
        media_attachments=pdf_paths
    )

    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_image_sequence():
    """
    6. 5 image paths placeholders, prompting to explain the image sequence. 
    Model: gemini-2.5-flash
    """
    print("--- Running Example 6: Image Sequence Story (2.5 Flash) ---")
    
    # REPLACE WITH YOUR ACTUAL FILE PATHS
    image_paths = [
        "media_samples/frame_01.jpg",
        "media_samples/frame_02.jpg",
        "media_samples/frame_03.jpg",
        "media_samples/frame_04.jpg",
        "media_samples/frame_05.jpg"
    ]
    
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        print(f"Files not found: {missing}. Please update paths in the function.")
        return

    prompt = "These images represent a sequence of events. Narrate the story unfolding across these 5 frames."

    response, tokens = prompt_gemini(
        model="gemini-2.5-flash",
        prompt=prompt,
        media_attachments=image_paths
    )

    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response}")


def example_mixed_media():
    """
    7. Video, PDF and 2 image placeholders, prompting to explain how they relate. 
    Model: gemini-2.5-pro
    """
    print("--- Running Example 7: Mixed Media Context (2.5 Pro) ---")
    
    # REPLACE WITH YOUR ACTUAL FILE PATHS
    media_files = [
        "media_samples/video.mp4",    # Video
        "media_samples/file.pdf",     # PDF
        "media_samples/image.png",    # Image 1
        "media_samples/image.png"     # Image 2
    ]

    missing = [p for p in media_files if not os.path.exists(p)]
    if missing:
        print(f"Files not found: {missing}. Please update paths in the function.")
        return

    prompt = (
        "I have attached a video, a PDF, and two images. "
        "Explain how the video relates to the data in the images and the text in the PDF."
    )

    response, tokens = prompt_gemini(
        model="gemini-2.5-pro",
        prompt=prompt,
        media_attachments=media_files
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
        
        # example_url_context()
        # example_structured_no_tools()
        # example_search_and_code()
        
        # NOTE: For the following, ensure you create the files or update paths:
        # example_video_structured()
        # example_multi_pdf_summary()
        # example_image_sequence()
        # example_mixed_media()
        
        # Utility to clean up cloud storage after testing heavy media
        # delete_all_uploads()
        pass