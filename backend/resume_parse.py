import fitz  # PyMuPDF
import logging
from io import BytesIO
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file: BytesIO) -> str:
    try:
        content = file.read()
        if not content:
            raise Exception("Empty file content")
        pdf_file = BytesIO(content)
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        if not text.strip():
            raise Exception("No text extracted from PDF")
        logger.info("Successfully extracted text from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise
