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
        google_search=True,
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

# --- 4. Gemini 3: High-Res Media ---
def run_gemini_3_test():
    print(f"\n--- Running Gemini 3 Pro (High Res PDF + Thinking) ---")
    
    # Example: Reading a dense PDF with High resolution
    # UPDATE PATH
    pdf_path = '/path/to/file.pdf' 
    
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

# --- 5. Gemini 3: Search + Code ---
def run_gemini_3_search_code_test():
    print(f"\n--- Running Gemini 3 Pro (Search + Code Only) ---")
    
    # The Prompt requires:
    # 1. Google Search: To find current population data (factual retrieval).
    # 2. Code Execution: To perform exact math (percentage difference) rather than hallucinating numbers.
    prompt = (
        "Search for the current metro area populations of Tokyo and Delhi. "
        "Then, use Python code to calculate the difference and determine exactly what "
        "percentage larger the more populous city is compared to the smaller one."
    )
    
    # We do NOT pass a response_schema, so we get a text response.
    # The function will automatically format the search citations and code blocks.
    response_text, tokens = prompt_gemini_3(
        prompt=prompt,
        google_search=True,
        code_execution=True,
        thinking_level="high" # 'High' is recommended for multi-step tool orchestration
    )
    
    print(f"Input Tokens: {tokens}")
    print(f"Response:\n{response_text}")

# --- 6. Gemini 3: Tools + Structured Output (Combined) ---
# Previous models could not do Google Search AND Strict JSON at the same time.
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

# --- 7. Gemini 3: Complex Agentic Task (Search + Code + JSON) ---
class FinancialAnalysis(BaseModel):
    asset_a: str
    asset_b: str
    price_a: float
    price_b: float
    # The model will compute this using Python and populate the field
    calculated_ratio: float 
    recommendation: str

def run_gemini_3_complex_test():
    print(f"\n--- Running Gemini 3 Complex Agentic Test ---")
    print("Goal: Search live crypto prices, use Code to calculate a ratio, and return JSON.")
    
    # This prompt requires:
    # 1. Google Search: To find the *current* prices (since knowledge cutoff is Jan 2025).
    # 2. Code Execution: To perform the division accurately (LLMs are bad at raw math).
    # 3. Structured Output: To map the result to the FinancialAnalysis class.
    prompt = (
        "Find the current price of Bitcoin (BTC) and Ethereum (ETH) in USD. "
        "Use code to calculate the ETH/BTC ratio (Price of ETH divided by Price of BTC). "
        "Return the prices, the calculated ratio, and a brief recommendation text."
    )
    
    try:
        # We assume prompt_gemini_3 is imported from gemini_utils
        result, tokens = prompt_gemini_3(
            prompt=prompt,
            response_schema=FinancialAnalysis, # Structured output format
            google_search=True,   # Get Data
            code_execution=True,  # Process Data
            thinking_level="high"
        )
        
        print(f"Input Tokens: {tokens}")
        print("Structured Result:")
        print(f"  - BTC: ${result.price_a}")
        print(f"  - ETH: ${result.price_b}")
        print(f"  - Ratio (ETH/BTC): {result.calculated_ratio:.5f}")
        print(f"  - Note: {result.recommendation}")
        
    except Exception as e:
        print(f"Complex test failed: {e}")

if __name__ == "__main__":
    run_text_test()
    # run_json_test()
    # run_video_test() # Update path first
    # run_gemini_3_test() # Update path first
    # run_gemini_3_search_code_test()
    # run_gemini_3_structured_tool_test()
    # run_gemini_3_complex_test()