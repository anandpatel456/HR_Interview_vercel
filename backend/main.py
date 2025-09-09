from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import os
import time
from typing import Dict, List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import logging
import asyncio
import fitz  # PyMuPDF

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to ["http://localhost:3000"] or Vercel frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

# Session storage (in-memory)
interview_sessions: Dict[str, Dict] = {}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MIN_INTERVIEW_DURATION = 60  # 1 minute in seconds
SESSION_TTL = 3600  # 1 hour, for auto-cleanup

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

async def process_resume(file: BytesIO, session_id: str):
    text = extract_text_from_pdf(file)
    interview_sessions[session_id]["resume"] = text
    logger.info(f"Processed resume for session {session_id}, text length: {len(text)}")

# ====== LangChain Setup ======

class CustomChatHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def add_messages(self, messages: List):
        self.messages.extend(messages)

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages = []

chat_histories: Dict[str, CustomChatHistory] = {}

def get_session_history(session_id: str) -> CustomChatHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = CustomChatHistory()
    return chat_histories[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ====== MODELS ======

class UserResponse(BaseModel):
    session_id: str
    answer: str
    is_complete: bool = False

class EndInterviewRequest(BaseModel):
    session_id: str

class StartInterviewRequest(BaseModel):
    session_id: str
    difficulty: str = "Medium"

# ====== INTERVIEW LOGIC ======

def run_interview(resume_text: str, chat_history: list, category: str = "general", difficulty: str = "Medium") -> str:
    difficulty_instruction = {
        "Easy": "Ask simple, beginner-friendly questions based on resume details.",
        "Medium": "Ask balanced, intermediate-level questions tied to resume specifics.",
        "Hard": "Ask challenging, in-depth questions directly related to resume content."
    }

    formatted_history = "\n".join([
        f"Interviewer: {turn.get('question', '')}\nCandidate: {turn.get('answer', '')}\nEvaluation: {turn.get('evaluation', '')}"
        for turn in chat_history
    ])
    system_prompt = f"""
You are a professional HR interviewer conducting a job interview. {difficulty_instruction.get(difficulty, '')}

Resume:
\"\"\"{resume_text}\"\"\"

Past Conversation:
{formatted_history}

Instructions:
1. Extract specific details from the resume (e.g., job roles, skills, projects, education).
2. Ask a single, concise question based on the resume for the category: {category}.
3. For 'general', ask about soft skills or experiences (e.g., teamwork, leadership) tied to resume.
4. For 'technical', ask about specific skills or tools listed in the resume.
5. For 'projects', ask about a specific project or achievement mentioned in the resume.
6. Ensure the question is clear, relevant, and no longer than 20 words.
""".strip()
    try:
        response = conversation.invoke(
            {"input": "Ask the next question.", "system_prompt": system_prompt},
            config={"configurable": {"session_id": f"interview_{category}"}},
        ).content.strip()
        words = response.split()
        if len(words) > 20:
            response = " ".join(words[:20]) + "?"
        elif not response.endswith("?"):
            response += "?"
        logger.info(f"Generated question for category {category} [{difficulty}]: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating question: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate question due to API error")

def evaluate_answer(question: str, answer: str, resume_text: str, chat_history: list, category: str, difficulty: str) -> Dict:
    difficulty_threshold = {"Easy": 2, "Medium": 3, "Hard": 4}  # Stricter for harder difficulties
    formatted_history = "\n".join([
        f"Interviewer: {turn.get('question', '')}\nCandidate: {turn.get('answer', '')}"
        for turn in chat_history
    ])
    system_prompt = f"""
You are an HR expert evaluating a candidate's answer in a mock interview.

Question: {question}
Answer: {answer}
Resume: \"{resume_text}\"
Past Conversation: {formatted_history}
Category: {category}

Instructions:
1. Score the answer on a 1-5 scale for:
   - Relevance: Alignment with question, category, and resume details.
   - Quality: Clarity (structure and understandability), Depth (use of examples, details from resume), and Completeness (does it feel finished, not cut off).
   Overall score is the average (integer).
2. If overall score < {difficulty_threshold.get(difficulty, 3)}, provide a polite, concise comment (1-2 sentences) explaining the issue (e.g., off-topic, lacking detail, incomplete) and suggest improvement.
3. If score >= {difficulty_threshold.get(difficulty, 3)}, just say "Good answer."
4. Output ONLY in JSON: {{"score": int, "comment": str}} where comment is empty if score is good.
""".strip()
    try:
        response = conversation.invoke(
            {"input": "Evaluate the answer.", "system_prompt": system_prompt},
            config={"configurable": {"session_id": "evaluation"}},
        ).content.strip()
        import json
        eval_result = json.loads(response)
        logger.info(f"Evaluated answer for question '{question}': score={eval_result['score']}, comment='{eval_result['comment']}'")
        return eval_result
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        return {"score": 3, "comment": ""}  # Fallback to neutral

def generate_feedback(chat_history: list) -> str:
    if not chat_history:
        return "No interview data available for feedback"
    logger.info(f"Generating feedback with chat_history: {chat_history}")
    history = "\n".join([
        f"Interviewer: {turn.get('question', '')}\nCandidate: {turn.get('answer', '')}\nEvaluation: {turn.get('evaluation', '')}"
        for turn in chat_history if isinstance(turn, dict) and 'question' in turn and 'answer' in turn
    ])
    logger.info(f"Feedback transcript: {history}")
    system_prompt = f"""
You are an HR expert providing a single, overall feedback summary after a mock interview.

Transcript:
{history}

Instructions:
1. Evaluate the overall performance based on all answers and evaluations for:
   - Relevance: How well answers addressed questions and aligned with the resume.
   - Quality: Clarity, depth, completeness of responses.
2. Identify overall strengths (e.g., clear communication, relevant examples) as a bullet list starting with "*Overall Strengths:*".
3. Always highlight at least one area for improvement (e.g., lack of detail, off-topic responses) as a bullet list starting with "*Areas for Improvement:*", even if minor.
4. Provide concise, actionable feedback in a friendly tone, starting with "Keep practicing".
""".strip()
    try:
        feedback = conversation.invoke(
            {"input": "Provide feedback based on the transcript.", "system_prompt": system_prompt},
            config={"configurable": {"session_id": "feedback"}},
        ).content.strip()
        logger.info(f"Generated feedback: {feedback}")
        return feedback
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate feedback due to API error")

# ====== ROUTES ======

@app.on_event("startup")
async def startup_event():
    global interview_sessions
    interview_sessions = {}  # Initialize empty sessions on startup
    logger.info("Initialized in-memory session storage")

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    session = interview_sessions.get(session_id)
    if not session:
        logger.error(f"Status check failed: Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "resume_processed": bool(session["resume"])}

@app.post("/upload-resume")
async def upload_resume(resume: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not resume.filename.endswith(".pdf"):
        logger.error("Upload failed: Only PDF files are supported")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    contents = await resume.read()
    if len(contents) > MAX_FILE_SIZE:
        logger.error("Upload failed: File size exceeds 5MB limit")
        raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")
    
    session_id = str(time.time())
    interview_sessions[session_id] = {
        "resume": "",
        "qa_history": [],
        "start_time": None,
        "end_time": None,
        "question_count": {"general": 0, "technical": 0, "projects": 0},
        "phase": "general",
        "ended": False,
        "difficulty": "Medium",
        "created_at": time.time()  # For TTL
    }
    
    background_tasks.add_task(process_resume, BytesIO(contents), session_id)
    
    logger.info(f"Initiated resume upload for session {session_id}")
    return {"success": True, "session_id": session_id}

@app.post("/start-interview")
async def start_interview(request: StartInterviewRequest):
    logger.info(f"Received start-interview request for session_id: {request.session_id}, difficulty: {request.difficulty}")
    session = interview_sessions.get(request.session_id)
    if not session:
        logger.error(f"Session {request.session_id} not found")
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
    if session.get("ended", False):
        logger.error(f"Session {request.session_id} already ended")
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} already ended")
    if not session["resume"]:
        logger.error(f"Resume not processed for session {request.session_id}")
        raise HTTPException(status_code=400, detail="Resume processing is not complete")
    
    session["difficulty"] = request.difficulty
    session["start_time"] = time.time()
    category = "general"

    try:
        question = run_interview(session["resume"], session["qa_history"], category, request.difficulty)
        session["qa_history"].append({"question": question, "type": category})
        session["question_count"][category] += 1
        logger.info(f"Started interview for session {request.session_id} with question: {question}")
        return {"success": True, "question": question, "session_id": request.session_id}
    except Exception as e:
        logger.error(f"Failed to start interview for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")

@app.post("/submit-answer")
async def submit_answer(request: UserResponse):
    logger.info(f"Received submit-answer request for session_id: {request.session_id}")
    session = interview_sessions.get(request.session_id)
    if not session or session.get("ended", False):
        logger.error(f"Session {request.session_id} not found or already ended")
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found or already ended")
    
    # Check timeout
    elapsed_time = time.time() - session.get("start_time", time.time())
    if elapsed_time > 15 * 60:
        logger.info(f"Session {request.session_id} timed out, ending interview")
        return await end_interview_internal(request.session_id)

    if not session["qa_history"]:
        logger.error(f"No previous question for session {request.session_id}")
        raise HTTPException(status_code=400, detail="No previous question")

    last_qa = session["qa_history"][-1]
    if "answer" not in last_qa:
        last_qa["answer"] = ""
    if not request.is_complete:
        # Append to partial answer
        last_qa["answer"] += " " + request.answer.strip()
        session["qa_history"][-1] = last_qa
        logger.info(f"Appended partial answer for session {request.session_id}")
        return {"success": True, "question": None, "end_interview": False, "message": "Partial answer received; continue."}
    
    # Complete answer: append final part and evaluate
    last_qa["answer"] += " " + request.answer.strip()
    full_answer = last_qa["answer"].strip()
    if len(full_answer.split()) < 5:
        follow_up = "Could you provide more details or clarify your response?"
        session["qa_history"].append({"question": follow_up, "type": last_qa["type"]})
        logger.info(f"Requested clarification for short answer in session {request.session_id}")
        return {"success": True, "question": follow_up, "end_interview": False, "message": "Answer too short."}

    # Evaluate answer for relevance and quality
    eval_result = evaluate_answer(
        last_qa["question"], full_answer, session["resume"], session["qa_history"][:-1], 
        last_qa["type"], session["difficulty"]
    )
    last_qa["evaluation"] = f"Score: {eval_result['score']}. {eval_result['comment']}"
    session["qa_history"][-1] = last_qa

    if eval_result["score"] < {"Easy": 2, "Medium": 3, "Hard": 4}[session["difficulty"]]:
        # Poor answer: provide comment and ask to re-answer or clarify
        follow_up = f"{eval_result['comment']} Please try answering again."
        session["qa_history"].append({"question": follow_up, "type": last_qa["type"]})
        logger.info(f"Poor answer evaluation for {request.session_id}; follow-up: {follow_up}")
        return {"success": True, "question": follow_up, "end_interview": False, "message": eval_result["comment"]}

    # Good answer: proceed to next
    current_type = last_qa["type"]
    if session["question_count"]["general"] < 1:
        next_category = "general"
    else:
        if session["phase"] != "technical_projects":
            session["phase"] = "technical_projects"
        next_category = "technical" if current_type == "projects" else "projects"

    next_question = run_interview(
        session["resume"], session["qa_history"], next_category, session["difficulty"]
    )
    session["qa_history"].append({"question": next_question, "type": next_category})
    session["question_count"][next_category] += 1
    logger.info(f"Good answer for {request.session_id}, next question: {next_question}")
    return {"success": True, "question": next_question, "end_interview": False}

@app.post("/end-interview")
async def end_interview(request: EndInterviewRequest):
    return await end_interview_internal(request.session_id)

@app.get("/get-feedback/{session_id}")
async def get_feedback(session_id: str):
    try:
        session = interview_sessions.get(session_id)
        if not session or not session.get("ended", False):
            raise HTTPException(status_code=404, detail="Feedback not found or interview not ended")
        feedback = generate_feedback(session["qa_history"])
        logger.info(f"Retrieved feedback for session {session_id} from memory")
        return {"success": True, "feedback": feedback}
    except Exception as e:
        logger.error(f"Error retrieving feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback")

async def end_interview_internal(session_id: str):
    logger.info(f"Ending interview for session {session_id}")
    session = interview_sessions.get(session_id)
    if not session or session.get("ended", False):
        logger.error(f"Session {session_id} not found or already ended")
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or already ended")
    try:
        session["end_time"] = time.time()
        duration = session["end_time"] - session["start_time"]
        logger.info(f"Interview duration for session {session_id}: {duration} seconds")

        if duration < MIN_INTERVIEW_DURATION:
            feedback = "You have to complete the interview for 1 minute to receive feedback."
        else:
            feedback = generate_feedback(session["qa_history"])

        result = {"success": True, "feedback": feedback, "end_interview": True}
        session["ended"] = True
        logger.info(f"Ended interview for session {session_id}")
        loop = asyncio.get_event_loop()
        loop.create_task(cleanup_session(session_id))
        return result
    except Exception as e:
        logger.error(f"Error ending interview {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def cleanup_session(session_id: str):
    await asyncio.sleep(300.0)  # Delay cleanup for 5 minutes
    if session_id in interview_sessions:
        if interview_sessions[session_id].get("ended", False) or (time.time() - interview_sessions[session_id]["created_at"] > SESSION_TTL):
            del interview_sessions[session_id]
            if session_id in chat_histories:
                del chat_histories[session_id]
            logger.info(f"Cleaned up session {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)