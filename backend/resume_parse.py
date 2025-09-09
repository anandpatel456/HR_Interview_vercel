import fitz  # PyMuPDF
from io import BytesIO
from langchain_openai import ChatOpenAI
import logging
import json
from fastapi import HTTPException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

def extract_text_from_pdf(file: BytesIO) -> str:
    try:
        file.seek(0)  # Ensure stream is at start
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text("text").strip()
            if page_text:  # Skip empty pages
                text += f"Page {page_num}:\n{page_text}\n\n"
        doc.close()
        logger.info(f"Extracted {len(text)} characters from PDF with {len(doc)} pages")
        if not text:
            raise ValueError("PDF appears to be empty or unreadable")
        return text.strip()
    except fitz.FileDataError:
        logger.error("PDF is encrypted or corrupted")
        raise HTTPException(status_code=400, detail="PDF is encrypted or corrupted")
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

def parse_resume_to_json(text: str) -> dict:
    system_prompt = """
You are an expert resume parser. Parse the provided resume text into a structured JSON format.

Resume Text:
\"""{resume_text}\"""

Instructions:
1. Extract and categorize information into the following JSON schema:
{
  "name": str,  // Full name or "Unknown" if not found
  "contact": {
    "email": str,  // or "" if not found
    "phone": str,  // or "" if not found
    "linkedin": str  // or "" if not found
  },
  "education": [
    {
      "degree": str,  // e.g., "B.S. Computer Science"
      "institution": str,  // e.g., "MIT"
      "years": str  // e.g., "2018-2022" or "" if not found
    }
  ],
  "experience": [
    {
      "role": str,  // e.g., "Software Engineer"
      "company": str,  // e.g., "Google"
      "years": str,  // e.g., "2022-Present" or "" if not found
      "description": str  // Key responsibilities or "" if not found
    }
  ],
  "skills": [str],  // List of skills, e.g., ["Python", "Java"]
  "projects": [
    {
      "name": str,  // Project name or "Unnamed Project" if not specified
      "description": str  // Project details or "" if not found
    }
  ]
}
2. Use context clues (e.g., headings, formatting) to identify sections.
3. If a section is missing, return empty lists or strings as appropriate.
4. Output ONLY valid JSON. If parsing fails, return the schema with empty/default values.
""".strip()
    try:
        response = llm.invoke(system_prompt.format(resume_text=text)).content.strip()
        parsed = json.loads(response)
        logger.info(f"Parsed resume into JSON: {json.dumps(parsed, indent=2)}")
        return parsed
    except Exception as e:
        logger.error(f"Error parsing resume to JSON: {e}")
        # Return default schema on failure
        return {
            "name": "Unknown",
            "contact": {"email": "", "phone": "", "linkedin": ""},
            "education": [],
            "experience": [],
            "skills": [],
            "projects": []
        }