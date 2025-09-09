import fitz  # PyMuPDF
import json
from io import BytesIO
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

def extract_text_from_pdf(file: BytesIO) -> str:
    try:
        # Create a new document from the BytesIO object
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()  # Explicitly close the document
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def parse_resume_to_json(text: str) -> Dict:
    prompt = f"""
    Extract key details from this resume text into JSON format with sections for personal_info, education, experience, skills, and projects.
    Keep the response concise and relevant.
    Resume: {text}
    """
    try:
        response = llm.invoke(prompt).content.strip()
        return json.loads(response)
    except Exception as e:
        raise Exception(f"Error parsing resume to JSON: {str(e)}")