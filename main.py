import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Use Field for potential future validation
from supabase import create_client, Client
from dotenv import load_dotenv
import json # To safely parse Gemini's potential JSON output
import logging # For better error tracking

# --- Configuration & Initialization ---

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get API keys and Supabase details from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Basic validation to ensure environment variables are set
if not all([GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    logger.error("Missing one or more environment variables (GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_KEY)")
    # In a real app, you might raise a more specific startup error
    # For simplicity here, we'll let it fail later if keys are missing during client init

# Initialize Google Generative AI client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using a model optimized for speed and cost for this task initially
    # Ensure this model supports JSON output or adjust parsing logic if needed
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or 'gemini-1.5-pro-latest' for more power
    logger.info("Google Generative AI client configured successfully.")
except Exception as e:
    logger.error(f"Error configuring Google Generative AI client: {e}")
    # Handle error appropriately, maybe exit or raise

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Supabase client: {e}")
    # Handle error appropriately

# Initialize FastAPI app
app = FastAPI(
    title="CoupleScan API",
    description="Analyzes relationship chat logs using AI.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Allows requests from your frontend (running on Vercel or locally)
# Adjust origins if needed for production
origins = [
    "http://localhost",          # Allow local development (if frontend runs on default port)
    "http://localhost:3000",     # Common local dev port for frameworks like React/Vue
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "https://*.vercel.app",     # Allow any subdomain from Vercel - BE CAREFUL IN PRODUCTION
    # Add your actual Vercel production URL here when you have it
    # e.g., "https://couplescan-frontend-your-hash.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # More permissive for now, tighten this in production to your actual frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
logger.info("CORS middleware added.")

# --- Pydantic Models (Data Validation) ---

class TextInput(BaseModel):
    chat_log: str = Field(..., min_length=50) # Basic validation: require at least 50 chars

class AnalyzeResponse(BaseModel):
    report_id: str
    teaser_text: str

class ReportResponse(BaseModel):
    full_report: str

# --- API Endpoints ---

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_chat(text_input: TextInput):
    """
    Receives a chat log, analyzes it using Gemini, stores the result,
    and returns a teaser and ID.
    """
    logger.info("Received request for /analyze")
    try:
        # Construct the prompt for Gemini
        # This prompt explicitly asks for JSON output.
        prompt = f"""
        You are a relationship counselor AI. Analyze the following chat log thoroughly.
        Identify the primary sentiment, communication styles (e.g., passive-aggressive, open, avoidant),
        potential conflict points, and overall health indicators.

        Chat Log:
        ---
        {text_input.chat_log}
        ---

        Based on your analysis, return ONLY a valid JSON object (no extra text before or after) with exactly two keys:
        1. "teaser": A compelling, slightly dramatic, one-sentence teaser (max 25 words) hinting at the main findings (e.g., "Analysis reveals underlying tensions and moments of strong connection.")
        2. "full_report": A comprehensive 500-word analysis covering sentiment, communication styles, conflict points, and health indicators. Structure it clearly.
        """

        logger.info("Sending request to Gemini API.")
        # Call the Gemini API - specifying JSON output if the model supports it directly
        # Note: Direct JSON mode might require specific model versions or settings.
        # This approach relies on the prompt instructing the model to return JSON.
        response = await gemini_model.generate_content_async(
             prompt,
             generation_config=genai.types.GenerationConfig(
                 # candidate_count=1, # Default is 1
                 # stop_sequences=['...'], # Optional stop words
                 # max_output_tokens=2048, # Adjust if needed, 500 words ~ 700 tokens + teaser
                 temperature=0.7, # Controls randomness, adjust as needed
                 # response_mime_type="application/json" # Uncomment if using a model/version supporting direct JSON output
             )
         )

        logger.info("Received response from Gemini API.")

        # --- Safely Parse Gemini's Response ---
        try:
            # Clean potential markdown/code block formatting around JSON
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip() # Clean again after stripping markers

            # Parse the cleaned text as JSON
            analysis_result = json.loads(raw_text)
            teaser = analysis_result.get("teaser")
            full_report = analysis_result.get("full_report")

            if not teaser or not full_report:
                logger.error("Gemini response missing 'teaser' or 'full_report' key after JSON parsing.")
                raise HTTPException(status_code=500, detail="AI analysis failed to produce expected format.")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse Gemini response as JSON. Raw response: {response.text[:500]}...") # Log beginning of raw response
            raise HTTPException(status_code=500, detail="AI analysis response was not valid JSON.")
        except Exception as e: # Catch other potential errors during parsing
             logger.error(f"Error processing Gemini response: {e}")
             raise HTTPException(status_code=500, detail="Error processing AI analysis.")

        # --- Store in Supabase ---
        logger.info("Inserting report into Supabase.")
        try:
            data, count = supabase.table('reports').insert({
                'teaser_text': teaser,
                'report_text': full_report
            }).execute()

            # Check if insertion was successful and data is returned
            if not data or len(data[1]) == 0:
                 logger.error(f"Supabase insert failed or returned no data. Count: {count}")
                 raise HTTPException(status_code=500, detail="Failed to store analysis report.")

            inserted_report = data[1][0] # Supabase returns a tuple (status, data_list)
            report_id = inserted_report['id']
            logger.info(f"Report inserted successfully with ID: {report_id}")

            return AnalyzeResponse(report_id=report_id, teaser_text=teaser)

        except Exception as e:
            logger.error(f"Error inserting data into Supabase: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred in /analyze: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during analysis.")

@app.get("/get_report", response_model=ReportResponse)
async def get_report(report_id: str = Query(..., description="The UUID of the report to retrieve.")):
    """
    Retrieves the full analysis report from Supabase using its ID.
    This endpoint should ideally be protected in a real application
    (e.g., verifying payment), but is kept simple here.
    """
    logger.info(f"Received request for /get_report with ID: {report_id}")
    try:
        data, count = supabase.table('reports').select('report_text').eq('id', report_id).limit(1).execute()

        # Check if data was found
        if not data or len(data[1]) == 0:
            logger.warning(f"Report not found for ID: {report_id}")
            raise HTTPException(status_code=404, detail="Report not found.")

        report_data = data[1][0]
        full_report = report_data.get('report_text')

        if not full_report:
             logger.error(f"Report found for ID {report_id}, but 'report_text' field is missing or empty.")
             raise HTTPException(status_code=500, detail="Stored report data is corrupted or incomplete.")

        logger.info(f"Successfully retrieved report for ID: {report_id}")
        return ReportResponse(full_report=full_report)

    except HTTPException as e:
         # Re-raise HTTPExceptions to return proper status codes
         raise e
    except Exception as e:
        logger.error(f"Error retrieving report from Supabase for ID {report_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error while retrieving report: {e}")


# --- Health Check Endpoint (Good Practice) ---
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    logger.info("Health check endpoint called.")
    return {"status": "ok"}


# --- Uvicorn Runner (for local development) ---
# This part allows you to run the app directly using 'python main.py'
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # Render will use its own command, specified in its dashboard, for production.