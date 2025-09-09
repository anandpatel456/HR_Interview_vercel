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
import json
import atexit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

# Session storage (persistent via JSON file)
SESSIONS_FILE = "interview_sessions.json"
interview_sessions: Dict[str, Dict] = {}

def load_sessions():
    global interview_sessions
    try:
        with open(SESSIONS_FILE, "r") as f:
            interview_sessions = json.load(f)
        logger.info(f"Loaded {len(interview_sessions)} sessions from {SESSIONS_FILE}")
    except FileNotFoundError:
        interview_sessions = {}
        logger.info("No sessions file found, starting fresh")

def save_sessions():
    with open(SESSIONS_FILE, "w") as f:
        json.dump(interview_sessions, f, default=str)  # Handle non-serializable types
    logger.info(f"Saved {len(interview_sessions)} sessions to {SESSIONS_FILE}")

atexit.register(save_sessions)  # Save on exit

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MIN_INTERVIEW_DURATION = 60  # 1 minute in seconds

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

def parse_resume(resume_text: str) -> Dict:
    system_prompt = """
You are a resume parser. Extract and structure the resume into JSON sections:
- "personal_info": Name, email, phone, location.
- "education": List of dicts with degree, institution, dates, GPA if mentioned.
- "experience": List of dicts with job title, company, dates, responsibilities (as bullet list).
- "skills": List of skills.
- "projects": List of dicts with project name, description, technologies.
- "other": Any other sections like certifications, awards.

Output ONLY valid JSON.
""".strip()
    
    try:
        response = llm.invoke(system_prompt + f"\n\nResume Text:\n{resume_text}").content.strip()
        parsed = json.loads(response)
        logger.info("Parsed resume into structured JSON")
        return parsed
    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
        return {}

async def process_resume(file: BytesIO, session_id: str):
    text = extract_text_from_pdf(file)
    parsed = parse_resume(text)
    interview_sessions[session_id]["resume_text"] = text
    interview_sessions[session_id]["parsed_resume"] = parsed
    save_sessions()  # Persist after processing
    logger.info(f"Processed and parsed resume for session {session_id}")

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

def run_interview(parsed_resume: Dict, chat_history: list, category: str = "general", difficulty: str = "Medium") -> str:
    difficulty_instruction = {
        "Easy": "Ask simple, beginner-friendly questions based on resume details.",
        "Medium": "Ask balanced, intermediate-level questions tied to resume specifics.",
        "Hard": "Ask challenging, in-depth questions directly related to resume content."
    }

    formatted_history = "\n".join([
        f"Interviewer: {turn.get('question', '')}\nCandidate: {turn.get('answer', '')}\nEvaluation: {turn.get('evaluation', {}).get('comment', 'N/A')}"
        for turn in chat_history if "question" in turn
    ])
    resume_json = json.dumps(parsed_resume, indent=2)
    system_prompt = f"""
You are a professional HR interviewer conducting a job interview. {difficulty_instruction.get(difficulty, '')}

Structured Resume (JSON):
{resume_json}

Past Conversation:
{formatted_history}

Instructions:
1. Use specific details from the structured resume sections (e.g., target questions to a particular experience, skill, or project).
2. Ask a single, concise question based on the resume for the category: {category}.
3. For 'general', ask about soft skills or experiences (e.g., teamwork, leadership) tied to resume.
4. For 'technical', ask about specific skills or tools listed in the resume.
5. For 'projects', ask about a specific project or achievement mentioned in the resume.
6. For 'wrap_up', ask standard HR wrap-up questions like "Why are you interested in this role?" or "Do you have any questions for us?".
7. Ensure the question is clear, relevant, and no longer than 20 words.
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

def evaluate_answer(question: str, answer: str, parsed_resume: Dict, difficulty: str) -> Dict:
    resume_json = json.dumps(parsed_resume, indent=2)
    system_prompt = f"""
You are an HR interviewer evaluating a candidate's answer.

Question: {question}
Answer: {answer}
Structured Resume: {resume_json}

Based on the difficulty ({difficulty}), score the answer on three metrics (0-10):
- Relevance: How well it addresses the question and ties to the resume.
- Clarity: How clear, structured, and concise the response is.
- Depth: How in-depth the response is, with use of examples or details.

Provide a short, friendly comment (1-2 sentences).
If any score <6, suggest a follow-up message (e.g., "That doesn't seem related—could you elaborate on [specific resume detail]?") to guide the user without giving away answers.

Output ONLY JSON: {{"relevance_score": int, "clarity_score": int, "depth_score": int, "comment": str, "needs_followup": bool, "followup_message": str or ""}}
""".strip()
    
    try:
        response = llm.invoke(system_prompt).content.strip()
        eval_result = json.loads(response)
        logger.info(f"Evaluated answer for question '{question}': {eval_result}")
        return eval_result
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        return {"relevance_score": 5, "clarity_score": 5, "depth_score": 5, "comment": "Evaluation failed.", "needs_followup": False, "followup_message": ""}

def generate_feedback(chat_history: list) -> str:
    if not chat_history:
        return json.dumps({"error": "No interview data available for feedback"})
    
    logger.info(f"Generating feedback with chat_history: {chat_history}")
    history = "\n".join([
        f"Interviewer: {turn.get('question', '')}\nCandidate: {turn.get('answer', '')}\nEvaluation: {turn.get('evaluation', {}).get('comment', 'N/A')}"
        for turn in chat_history if isinstance(turn, dict) and 'question' in turn
    ])
    
    # Compute averages
    evaluations = [turn["evaluation"] for turn in chat_history if turn.get("evaluation")]
    if evaluations:
        avg_relevance = sum(e["relevance_score"] for e in evaluations) / len(evaluations)
        avg_clarity = sum(e["clarity_score"] for e in evaluations) / len(evaluations)
        avg_depth = sum(e["depth_score"] for e in evaluations) / len(evaluations)
    else:
        avg_relevance = avg_clarity = avg_depth = 0
    scores = {
        "relevance": round(avg_relevance, 1),
        "clarity": round(avg_clarity, 1),
        "depth": round(avg_depth, 1)
    }
    
    system_prompt = f"""
You are an HR expert providing structured feedback after a mock interview.

Transcript:
{history}

Instructions:
1. Provide a concise summary (2-3 sentences) of the candidate's overall performance.
2. Evaluate performance based on all answers for:
   - Relevance: How well answers addressed questions and aligned with the resume.
   - Clarity: Overall clarity and structure of responses.
   - Depth: General depth and use of examples across answers.
3. Identify 1-3 specific strengths (e.g., clear communication, relevant examples) with examples from the transcript.
4. Identify 1-2 areas for improvement (e.g., lack of detail, off-topic responses) with specific examples, even if minor.
5. Provide 1-2 actionable suggestions tailored to the lowest-scoring metric (relevance, clarity, or depth).
6. Use a friendly, encouraging tone.

Output ONLY JSON with the following structure:
{{
  "summary": str,
  "scores": {{"relevance": float, "clarity": float, "depth": float}},
  "strengths": [str, ...],
  "areas_for_improvement": [str, ...],
  "actionable_advice": [str, ...]
}}
""".strip()
    
    try:
        response = conversation.invoke(
            {"input": "Provide structured feedback based on the transcript.", "system_prompt": system_prompt},
            config={"configurable": {"session_id": "feedback"}},
        ).content.strip()
        feedback = json.loads(response)
        feedback["scores"] = scores  # Ensure scores are included
        logger.info(f"Generated structured feedback: {feedback}")
        return json.dumps(feedback, indent=2)
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        return json.dumps({"error": "Failed to generate feedback due to API error"})

# ====== ROUTES ======

@app.on_event("startup")
async def startup_event():
    load_sessions()
    logger.info("Initialized session storage from file")

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    session = interview_sessions.get(session_id)
    if not session:
        logger.error(f"Status check failed: Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "resume_processed": bool(session.get("parsed_resume"))}

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
        "resume_text": "",
        "parsed_resume": {},
        "qa_history": [],
        "start_time": None,
        "end_time": None,
        "question_count": {"general": 0, "technical": 0, "projects": 0, "wrap_up": 0},
        "phase": "general",
        "ended": False,
        "difficulty": "Medium"
    }
    save_sessions()
    
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
    if not session.get("parsed_resume"):
        logger.error(f"Resume not processed for session {request.session_id}")
        raise HTTPException(status_code=400, detail="Resume processing is not complete")
    
    session["difficulty"] = request.difficulty
    session["start_time"] = time.time()
    category = "general"

    try:
        question = run_interview(session["parsed_resume"], session["qa_history"], category, request.difficulty)
        session["qa_history"].append({"question": question, "type": category})
        session["question_count"][category] += 1
        save_sessions()
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
    
    elapsed_time = time.time() - session.get("start_time", time.time())
    if elapsed_time > 15 * 60:
        logger.info(f"Session {request.session_id} timed out, ending interview")
        return await end_interview_internal(request.session_id)

    if not session["qa_history"]:
        logger.error(f"No previous question for session {request.session_id}")
        raise HTTPException(status_code=400, detail="No previous question")

    last_qa = session["qa_history"][-1]
    if "answer" not in last_qa:
        last_qa["answer"] = request.answer
        last_qa["evaluation"] = None  # Will add evaluation below
        session["qa_history"][-1] = last_qa

    if request.is_complete and request.answer:
        answer_words = len(request.answer.strip().split())
        if answer_words < 5:
            follow_up = "Could you provide more details or clarify your response?"
            session["qa_history"].append({"question": follow_up, "type": last_qa["type"], "is_followup": True})
            save_sessions()
            logger.info(f"Requested clarification for session {request.session_id} (short answer)")
            return {"success": True, "question": follow_up, "end_interview": False}

        # Evaluate relevance
        evaluation = evaluate_answer(last_qa["question"], request.answer, session["parsed_resume"], session.get("difficulty", "Medium"))
        last_qa["evaluation"] = evaluation
        session["qa_history"][-1] = last_qa  # Update with eval
        save_sessions()

        if evaluation["needs_followup"]:
            follow_up = evaluation["followup_message"]
            session["qa_history"].append({"question": follow_up, "type": last_qa["type"], "is_followup": True})
            save_sessions()
            logger.info(f"Generated follow-up for irrelevant answer in session {request.session_id}: {follow_up}")
            return {"success": True, "question": follow_up, "end_interview": False, "comment": evaluation["comment"]}

        # If good, proceed to next
        current_type = last_qa["type"]
        if session["phase"] == "general" and session["question_count"]["general"] < 3:
            next_category = "general"
        elif session["phase"] == "technical_projects" and (session["question_count"]["technical"] < 2 or session["question_count"]["projects"] < 2):
            next_category = "technical" if session["question_count"]["technical"] < session["question_count"]["projects"] else "projects"
        elif session["phase"] != "wrap_up":
            session["phase"] = "wrap_up"
            next_category = "wrap_up"
        else:
            # End after 2 wrap-up questions
            if session["question_count"]["wrap_up"] >= 2:
                return await end_interview_internal(request.session_id)
            next_category = "wrap_up"

        if session["phase"] == "general" and next_category != "general":
            session["phase"] = "technical_projects"

        next_question = run_interview(
            session["parsed_resume"], 
            session["qa_history"], 
            next_category, 
            session.get("difficulty", "Medium")
        )
        session["qa_history"].append({"question": next_question, "type": next_category})
        session["question_count"][next_category] += 1
        save_sessions()
        logger.info(f"Submitted answer for {request.session_id}, next question: {next_question}")
        return {"success": True, "question": next_question, "end_interview": False, "comment": evaluation["comment"]}

    return {"success": True, "question": None, "end_interview": False}

@app.post("/end-interview")
async def end_interview(request: EndInterviewRequest):
    return await end_interview_internal(request.session_id)

@app.get("/get-feedback/{session_id}")
async def get_feedback(session_id: str):
    try:
        session = interview_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        feedback = generate_feedback(session["qa_history"])
        logger.info(f"Retrieved feedback for session {session_id} from memory")
        return {"success": True, "feedback": json.loads(feedback)}  # Parse JSON string to dict for API response
    except Exception as e:
        logger.error(f"Error retrieving feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback")

@app.get("/list-sessions")
async def list_sessions():
    sessions = [{"session_id": sid, "start_time": s.get("start_time"), "ended": s.get("ended")} for sid, s in interview_sessions.items()]
    return {"success": True, "sessions": sessions}

@app.get("/get-history/{session_id}")
async def get_history(session_id: str):
    session = interview_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "qa_history": session["qa_history"]}

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
            feedback = json.dumps({"error": "You have to complete the interview for 1 minute to receive feedback."})
        else:
            feedback = generate_feedback(session["qa_history"])

        result = {"success": True, "feedback": json.loads(feedback), "end_interview": True}
        session["ended"] = True
        save_sessions()
        logger.info(f"Ended interview for session {session_id}")
        loop = asyncio.get_event_loop()
        loop.create_task(cleanup_session(session_id))
        return result
    except Exception as e:
        logger.error(f"Error ending interview {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def cleanup_session(session_id: str):
    await asyncio.sleep(3600.0)  # Delay cleanup for 1 hour to allow reviews
    if session_id in interview_sessions and interview_sessions[session_id].get("ended", False):
        # Optional: Keep persisted, no delete
        logger.info(f"Session {session_id} ready for long-term storage (no cleanup)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)