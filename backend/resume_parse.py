import fitz  # PyMuPDF
from io import BytesIO
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(file: BytesIO) -> str:
    try:
        file.seek(0)  # Reset stream position
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text("text").strip()
            if page_text:  # Skip empty pages
                text += page_text + "\n\n"  # Double newline for page separation
        doc.close()
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")