import enum
from pydantic import BaseModel
from gemini_utils import prompt_gemini, prompt_gemini_structured, prompt_gemini_3, check_api_key

# --- Setup ---
# Ensure your API key is set in your environment variables.
if not check_api_key():
    print("Stopping execution due to missing API key.")
    exit()

# --- 1. Text Test with URL Context ---
def run_text_test():
    test_model = "gemini-2.5-flash" 
    test_prompt = "What does this source say https://store.steampowered.com/sale/steammachine? Compare with the latest xbox."
    
    print(f"\n--- Running Text Test ---")
    response_text, tokens = prompt_gemini(
        model=test_model,
        prompt=test_prompt,
        thinking=True,
        url_context=True
    )
    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response_text}")

# --- 2. Video Test (Multimodal + Code Execution) ---
def run_video_test():
    # UPDATE THIS PATH TO A REAL FILE ON YOUR SYSTEM
    video_path = './my_video.mp4' 
    
    print(f"\n--- Running Video Test ---")
    try:
        response_text, tokens = prompt_gemini(
            model="gemini-2.5-flash",
            prompt="Summarize this video, look up console specs, and multiply 2587*8493 using code.",
            video_attachment=video_path,
            google_search=True,
            code_execution=True
        )
        print(f"Input Tokens: {tokens}")
        print(f"Response:\n{response_text}")
    except Exception as e:
        print(f"Skipping video test (File likely not found): {e}")

# --- 3. Structured Output (JSON) ---
class Movie(BaseModel):
    title: str
    director: str
    year: int

def run_json_test():
    print("\n--- Running Structured Output (JSON) Test ---")
    schema = list[Movie]
    response, tokens = prompt_gemini_structured(
        model="gemini-2.5-flash",
        prompt="List three classic sci-fi movies.",
        response_schema=schema,
    )
    print(f"Input Tokens: {tokens}")
    # Response is a Python object (list of Movie)
    for movie in response:
        print(f"- {movie.title} ({movie.year}) by {movie.director}")

# --- 4. Gemini 3: High-Res Media & Thinking ---
def run_gemini_3_test():
    print(f"\n--- Running Gemini 3 Pro (High Res PDF + Thinking) ---")
    
    # Example: Reading a dense PDF with High resolution
    # UPDATE PATH
    pdf_path = '/home/daniel/Downloads/Fuentes/exploration-of-tpus-for-ai-applications-3xh6xnkech.pdf' 
    
    try:
        response_text, tokens = prompt_gemini_3(
            prompt="Make a summary of this pdf",
            pdf_attachment=pdf_path,
            thinking_level="high",      # Uses dynamic high thinking
            media_resolution="high"     # Uses ~560 tokens per page for max OCR detail
        )
        print(f"Input Tokens: {tokens}")
        print(f"Response:\n{response_text}")
    except Exception as e:
        print(f"Skipping Gemini 3 test: {e}")

# --- 5. Gemini 3: Tools + Structured Output (Combined) ---
# Previous models could not easily do Google Search AND Strict JSON at the same time.
class Laptop(BaseModel):
    model_name: str
    price: str
    review_score: float

def run_gemini_3_structured_tool_test():
    print(f"\n--- Running Gemini 3 Pro (Search + JSON) ---")
    
    # We ask it to Google Search for CURRENT info, but return a strict JSON list
    prompt = "Search for the top 3 rated gaming laptops released in late 2024/early 2025."
    
    response_data, tokens = prompt_gemini_3(
        prompt=prompt,
        response_schema=list[Laptop],
        google_search=True, # Tool enabled
        thinking_level="low" # Faster response
    )
    
    print(f"Input Tokens: {tokens}")
    
    if isinstance(response_data, list):
        for laptop in response_data:
            print(f"Found: {laptop.model_name} | Price: {laptop.price} | Score: {laptop.review_score}")
    else:
        print(response_data) # Error string

if __name__ == "__main__":
    # run_text_test()
    # run_json_test()
    # run_video_test() # Update path first
    # run_gemini_3_test() # Update path first
    run_gemini_3_structured_tool_test()