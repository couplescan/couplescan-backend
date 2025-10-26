import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from supabase import create_client, Client
from dotenv import load_dotenv
import json
import logging
import zipfile
import io # Required for reading zip file in memory
from typing import Optional # Required for optional parameters

# --- Configuration & Initialization ---

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not all([GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    logger.error("FATAL: Missing one or more environment variables.")
    # Exit or raise critical error in a real production app

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using Pro for better structured output quality, Flash can also work but might need more prompt tuning
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    logger.info("Google Generative AI client configured successfully.")
except Exception as e:
    logger.error(f"FATAL: Error configuring Google Generative AI client: {e}")
    # Exit or raise critical error

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully.")
except Exception as e:
    logger.error(f"FATAL: Error initializing Supabase client: {e}")
    # Exit or raise critical error

app = FastAPI(
    title="CoupleScan API",
    description="Analyzes relationship chat logs (text paste or WhatsApp zip upload) using AI.",
    version="1.1.0" # Bump version
)

# --- CORS Middleware ---
# Define allowed origins explicitly for better security
origins = [
    "http://localhost",          # Allow local development 
    "http://localhost:3000",     # Common local dev port
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "https://couplescan.vercel.app", # <-- ADD YOUR EXACT VERCEL URL HERE
    # You might keep "*.vercel.app" if you rely heavily on preview URLs, 
    # but being specific is safer for production.
    # "https://*.vercel.app", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the specific list
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Allow only needed methods
    allow_headers=["*"], # Or specify allowed headers if known
)
logger.info(f"CORS middleware updated for specific origins: {origins}")

# --- Pydantic Models (Data Validation) ---

class TextInput(BaseModel):
    # Chat log is now optional, validated within the endpoint
    chat_log: Optional[str] = Field(None, min_length=50)

class AnalyzeResponse(BaseModel):
    report_id: str
    teaser_text: str

class ReportResponse(BaseModel):
    full_report: str

# --- Helper Functions ---

def extract_text_from_zip(file_content: bytes) -> Optional[str]:
    """Extracts text content from the first .txt file found in a zip archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_content)) as z:
            # Find the first .txt file in the zip archive
            txt_files = [f for f in z.namelist() if f.lower().endswith('.txt') and not f.startswith('__MACOSX')]
            if not txt_files:
                logger.warning("No .txt file found in the uploaded zip archive.")
                return None
            
            # Read the content of the first .txt file found
            # Assuming WhatsApp export format is UTF-8
            with z.open(txt_files[0]) as txt_file:
                chat_bytes = txt_file.read()
                try:
                    chat_text = chat_bytes.decode('utf-8')
                    logger.info(f"Successfully extracted text from {txt_files[0]} in zip.")
                    return chat_text
                except UnicodeDecodeError:
                    logger.warning(f"Could not decode {txt_files[0]} as UTF-8. Trying latin-1.")
                    try:
                       chat_text = chat_bytes.decode('latin-1')
                       logger.info(f"Successfully extracted text from {txt_files[0]} using latin-1.")
                       return chat_text
                    except UnicodeDecodeError:
                        logger.error(f"Failed to decode {txt_files[0]} using UTF-8 or latin-1.")
                        return None
    except zipfile.BadZipFile:
        logger.warning("Uploaded file is not a valid zip archive.")
        return None
    except Exception as e:
        logger.error(f"Error processing zip file: {e}", exc_info=True)
        return None

def create_teaser(full_report_text: str, word_count: int = 20) -> str:
    """Creates a teaser snippet from the beginning of the text."""
    words = full_report_text.split()
    if len(words) > word_count:
        return " ".join(words[:word_count]) + "..."
    else:
        return full_report_text # Return full text if it's shorter than word_count


# --- API Endpoints ---

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_chat(
    user_identifier: str = Form(..., description="The name or identifier of the user submitting the chat (e.g., 'Anant', 'Partner A')"), # Now required
    chat_log: Optional[str] = Form(None), # Keep validation here
    file: Optional[UploadFile] = File(None, description="A WhatsApp .zip export file.") # Add description
):
    """
    Receives chat log via text OR WhatsApp zip file upload, analyzes using Gemini,
    stores the full report, and returns a teaser and ID.
    """
    logger.info("Received request for /analyze")
    processed_chat_log = None

    # --- Input Processing ---
    if file:
        logger.info(f"Processing uploaded file: {file.filename}, content type: {file.content_type}")
        if file.content_type == 'application/zip' or file.filename.lower().endswith('.zip'):
            file_content = await file.read()
            processed_chat_log = extract_text_from_zip(file_content)
            if not processed_chat_log:
                raise HTTPException(status_code=400, detail="Could not extract text from the zip file. Ensure it contains a valid .txt chat log.")
        else:
            # Handle other file types later if needed (e.g., direct .txt upload)
             raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WhatsApp export .zip file.")
        await file.close() # Close the file handle
    elif chat_log:
        logger.info("Processing pasted text input.")
        if len(chat_log) < 50:
            logger.warning("Pasted chat log is too short.")
            raise HTTPException(status_code=400, detail="Pasted chat log must be at least 50 characters.")
        processed_chat_log = chat_log
    else:
        logger.warning("No input provided (neither text nor file).")
        raise HTTPException(status_code=400, detail="Please provide either pasted text or upload a WhatsApp .zip file.")

    if not processed_chat_log: # Should be caught above, but as a safeguard
         raise HTTPException(status_code=500, detail="Internal error processing input.")


    # --- AI Analysis ---
    try:
        # **UPDATED PROMPT FOR STRUCTURED OUTPUT**
        # **UPDATED PROMPT V6: STRICT Direct Address + Interaction Analysis**
        prompt = f"""
Act as a warm, empathetic, and insightful relationship advisor. **You are speaking *directly* to the user identified as '{user_identifier}'.** Your task is to analyze the following chat log, focusing on the communication dynamics between '{user_identifier}' (referred to as 'you') and the other person, and provide personalized feedback **only to 'you' ('{user_identifier}')**.

**VERY IMPORTANT INSTRUCTIONS:**
1.  **ALWAYS Use Direct Address:** Address '{user_identifier}' exclusively as "**you**" and "**your**" throughout the entire response. **NEVER** refer to '{user_identifier}' in the third person (e.g., "Anant did...", "Suhani should...").
2.  **Referencing the Other Person:** Refer to the other participant simply as "**the other person**" or by their name if it's clearly identifiable and distinct from '{user_identifier}'.
3.  **Analyze the Interaction, Report to the User:** Your analysis (Vibe, Positives, Improvements, Sentiment) should describe the **dynamics *between* you and the other person**, but phrase these observations *as if you are explaining them directly to '{user_identifier}'*.
4.  **User-Centric Advice:** The 'Actionable Advice' and 'Next Conversation Suggestion' sections **must** be exclusively focused on actionable steps for **you** ('{user_identifier}').
5.  **Tone:** Maintain a supportive, constructive, and insightful tone.
6.  **Formatting:** Start the response *directly* with the summary paragraph (NO first heading). Use Markdown headings (e.g., `## Communication Positives`) for all subsequent sections. Use paragraphs and bullet points appropriately.
7.  **Referencing Examples:** Cite examples using short quotes ("When you said '...'") or brief context ("During the talk about X..."), avoiding dates/times.
8.  **Emojis:** Incorporate relevant emojis naturally and sparingly (1-2 per main section) for warmth. 😊 🤔 🗣️ 🌱 💡 ✨
9.  **Length:** Be comprehensive.

Chat Log:
---
{processed_chat_log}
---

(Do NOT include a heading. Start directly with the paragraph analyzing the overall dynamic **you experienced**.)
(Provide a concise paragraph summarizing the conversation's general tone, mood, and the key **dynamics between you and the other person**.)

## Communication Positives 😊
(Explain observed healthy **patterns in the interaction** using paragraphs. Use bullet points for specific examples from the chat that showcase positive dynamics, highlighting **your role** or how things unfolded **between you both**.)
* Example: [e.g., It was great how **you** built on their idea about X, showing good synergy.]
* Example: [e.g., When the other person shared Y, **your** supportive response ('...') likely strengthened the connection.]

## Areas for Improvement 🤔
(Explain **patterns in the interaction** indicating potential friction or missed opportunities, framing constructively from **your perspective** or suggesting where **your actions** could influence the dynamic positively. Use bullet points for specific examples.)
* Example: [e.g., I noticed an imbalance in topic initiation; perhaps **you** felt you were leading most often?]
* Example: [e.g., The exchange about Z felt a bit abrupt. Exploring feelings more might have been beneficial **for both of you**, and perhaps **you** could gently guide it next time by asking...]

## Key Themes Discussed 🗣️
(Use bullet points to list the main topics relevant to the **interaction between you both**.)
* Theme 1
* Theme 2

## Sentiment Shift 🌱
(In a paragraph, describe shifts in the **overall sentiment of the interaction**, based on messages from **both you and the other person**.)

## Actionable Advice (for You) 💡
(Based on the analysis, provide 1-3 practical, constructive suggestions specifically **for you** ('{user_identifier}'). Use bullet points.)
* Suggestion: [e.g., To encourage more balance, **you** could try asking an open-ended question like 'What's been on your mind lately?' after sharing something of your own.]
* Suggestion: [e.g., When discussing sensitive topics like X, **you** might consider explicitly stating your positive intention first.]

## Next Conversation Suggestion (for You) ✨
(Suggest one specific, positive, or constructive question or topic **you** ('{user_identifier}') could bring up next time. Explain the reasoning briefly.)
"""

        logger.info("Sending structured request to Gemini API.")
        response = await gemini_model.generate_content_async(
             prompt,
             generation_config=genai.types.GenerationConfig(
                 temperature=0.7,
                 # Consider increasing max_output_tokens if reports get cut off
                 # max_output_tokens=4096 
             )
         )

        logger.info("Received structured response from Gemini API.")

        # --- Process Gemini's Response & Create Teaser ---
        try:
            full_report = response.text.strip()
            # Remove potential intro lines often added by models
            common_intros = [
                "Here's a detailed, structured analysis:",
                "Okay, here is the analysis:",
                "Here is the analysis:",
                "Here's the analysis:",
                "Based on the chat log:",
                "Okay, here's a breakdown:",
            ]
            original_report_length = len(full_report) # Store length before stripping
            for intro in common_intros:
                # Case-insensitive check and strip
                if full_report.lower().startswith(intro.lower()):
                    full_report = full_report[len(intro):].lstrip() # Remove intro and leading whitespace
                    logger.info(f"Removed intro phrase: '{intro}'")
                    break # Stop checking once an intro is removed

            # Basic check for report validity AFTER stripping intro
            if not full_report or len(full_report) < 150: # Adjust min length as needed
                logger.error(f"Gemini response too short or empty after stripping intro. Original Length: {original_report_length}, Final Length: {len(full_report)}. Response: {full_report[:150]}...")
                raise HTTPException(status_code=500, detail="AI analysis failed to produce a valid structured report.")

            # Create the teaser snippet from the beginning of the potentially stripped full report
            teaser_snippet = create_teaser(full_report, word_count=25)
            if not full_report or len(full_report) < 150: # Increased minimum length for structured report
                logger.error(f"Gemini structured response seems too short or empty. Length: {len(full_report)}. Response: {full_report[:150]}...")
                raise HTTPException(status_code=500, detail="AI analysis failed to produce a valid structured report.")

            # Create the teaser snippet from the beginning of the full report
            teaser_snippet = create_teaser(full_report, word_count=25) # Slightly longer teaser maybe

        except AttributeError:
            logger.error(f"Error accessing structured response text. Response object: {response}", exc_info=True)
            raise HTTPException(status_code=500, detail="Could not extract text from AI structured response.")
        except Exception as e:
            logger.error(f"Error processing Gemini structured response: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error processing AI structured analysis.")

        # --- Store in Supabase ---
        logger.info("Inserting full structured report into Supabase.")
        try:
            # Only insert the full report text
            data, count = supabase.table('reports').insert({
                'report_text': full_report
            }).execute()

            if not data or len(data[1]) == 0:
                 logger.error(f"Supabase insert failed or returned no data. Count: {count}")
                 raise HTTPException(status_code=500, detail="Failed to store analysis report.")

            inserted_report = data[1][0]
            report_id = inserted_report['id']
            logger.info(f"Structured report inserted successfully with ID: {report_id}")

            # Return the ID and the generated teaser snippet
            return AnalyzeResponse(report_id=report_id, teaser_text=teaser_snippet)

        except Exception as e:
            logger.error(f"Error inserting structured data into Supabase: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Database error: Failed to store structured report.")

    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions (like input validation)
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in /analyze: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during analysis.")


# --- GET Report Endpoint (No changes needed here) ---
@app.get("/get_report", response_model=ReportResponse)
async def get_report(report_id: str = Query(..., description="The UUID of the report to retrieve.")):
    """
    Retrieves the full analysis report from Supabase using its ID.
    """
    # ... (Keep the existing code for this function) ...
    logger.info(f"Received request for /get_report with ID: {report_id}")
    try:
        data, count = supabase.table('reports').select('report_text').eq('id', report_id).limit(1).execute()
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
         raise e
    except Exception as e:
        logger.error(f"Error retrieving report from Supabase for ID {report_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error while retrieving report: {e}")

# --- Health Check Endpoint (No changes needed) ---
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    logger.info("Health check endpoint called.")
    return {"status": "ok"}


# --- Uvicorn Runner (No changes needed) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Added reload=True for easier dev