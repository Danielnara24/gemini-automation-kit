import enum
from pydantic import BaseModel
from gemini_utils import prompt_gemini, prompt_gemini_structured, check_api_key

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

if __name__ == "__main__":
    run_text_test()
    # run_json_test()
    # run_video_test() # Uncomment to run if you have a file