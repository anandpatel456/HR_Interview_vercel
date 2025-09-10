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
MIN_INTERVIEW_DURATION = 60  # 1 minute in seconds for feedback eligibility
MAX_IRRELEVANT_RETRIES = 2  # Maximum retries for irrelevant answers
DEBOUNCE_DELAY = 3.0  # Seconds to wait for additional input before processing partial answer
INTERVIEW_DURATION = 15 * 60  # 15 minutes in seconds

def extract_text_from_pdf(file: BytesIO) -> str:
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text.strip()
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
    difficulty: str = "Medium"

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
        f"Interviewer: {turn.get('question', '')}\nCandidate: {turn.get('answer', '')}"
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

def generate_feedback(chat_history: list) -> str:
    if not chat_history:
        return "No interview data available for feedback"
    logger.info(f"Generating feedback with chat_history: {chat_history}")
    history = "\n".join([
        f"Interviewer: {turn.get('question', '')}\nCandidate: {turn.get('answer', '')}"
        for turn in chat_history if isinstance(turn, dict) and all(key in turn for key in ['question', 'answer'])
    ])
    irrelevant_count = sum(1 for turn in chat_history if turn.get('question', '').startswith("Your answer seems unrelated"))
    system_prompt = f"""
You are an HR expert providing a single, overall feedback summary after a mock interview.

Transcript:
{history}

Instructions:
1. Evaluate the overall performance based on all answers for:
   - Relevance: How well answers addressed questions and aligned with the resume.
   - Clarity: Overall clarity and structure of responses.
   - Depth: General depth and use of examples across answers.
2. Note that {irrelevant_count} answer(s) were flagged as irrelevant.
3. Identify overall strengths (e.g., clear communication, relevant examples) as a bullet list starting with "Overall Strengths:".
4. Always highlight at least one area for improvement (e.g., lack of detail, off-topic responses, or frequent irrelevant answers) as a bullet list starting with "Areas for Improvement:".
5. Provide concise, actionable feedback in a friendly tone, starting with "Keep practicing".
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

def check_answer_relevance(question: str, answer: str, resume_text: str, category: str) -> tuple[int, str]:
    system_prompt = f"""
You are an HR expert evaluating how relevant a candidate's answer is to a mock interview question.

Question: {question}
Answer: {answer}
Resume: "{resume_text}"
Category: {category}

Instructions:
1. Score relevance on a scale of 0–100:
   - 90–100 → Fully relevant (direct, detailed, aligned with resume).
   - 70–89 → Mostly relevant (answers question with basic reason, even if brief or imperfect grammar).
   - 40–69 → Weak relevance (somewhat related but unclear/incomplete).
   - 0–39 → Irrelevant (off-topic, nonsense, or unrelated).
2. Consider directness (does it name a language and give a reason?) and alignment (matches category).
3. Short answers naming a language with a simple reason should score 70–89 unless off-topic.
4. Return ONLY a JSON object with:
   - "score" (integer 0–100)
   - "reason" (short explanation ≤15 words)

Example:
Q: "Which programming language do you feel comfortable using, and why?"
A: "I like Python because it's easy."
→ {{ "score": 85, "reason": "Names Python, gives simple reason." }}
A: "I enjoy cooking."
→ {{ "score": 10, "reason": "Unrelated to programming languages." }}
""".strip()

    try:
        response = conversation.invoke(
            {"input": "Evaluate the relevance of the answer.", "system_prompt": system_prompt},
            config={"configurable": {"session_id": "relevance_check"}},
        ).content.strip()
        result = json.loads(response)
        logger.info(f"Relevance check result for question '{question}': {result}")
        return int(result.get("score", 100)), result.get("reason", "")
    except Exception as e:
        logger.error(f"Error checking answer relevance: {e}")
        return 100, ""  # Default: assume fully relevant on error

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
        "irrelevant_retries": 0,
        "last_partial_timestamp": None
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
    session["irrelevant_retries"] = 0
    session["last_partial_timestamp"] = None
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
async def submit_answer(request: UserResponse, background_tasks: BackgroundTasks):
    logger.info(f"Received submit-answer request for session_id: {request.session_id}, is_complete: {request.is_complete}")
    session = interview_sessions.get(request.session_id)
    if not session or session.get("ended", False):
        logger.error(f"Session {request.session_id} not found or already ended")
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found or already ended")
    
    elapsed_time = time.time() - session.get("start_time", time.time())
    if elapsed_time > INTERVIEW_DURATION:
        logger.info(f"Session {request.session_id} timed out after {elapsed_time} seconds, ending interview")
        return await end_interview_internal(request.session_id)

    if not session["qa_history"]:
        logger.error(f"No previous question for session {request.session_id}")
        raise HTTPException(status_code=400, detail="No previous question")

    last_qa = session["qa_history"][-1]

    # Append/overwrite the answer
    if "answer" not in last_qa:
        last_qa["answer"] = request.answer
    else:
        last_qa["answer"] += " " + request.answer

    session["qa_history"][-1] = last_qa

    # If answer is marked as complete, process immediately
    if request.is_complete:
        session["last_partial_timestamp"] = None
        return await process_complete_answer(
            session_id=request.session_id,
            last_answer=last_qa["answer"],
            last_qa=last_qa,
            current_type=last_qa["type"],
            difficulty=session.get("difficulty", request.difficulty)
        )

    # Handle partial answer
    current_time = time.time()
    session["last_partial_timestamp"] = current_time
    last_answer = last_qa["answer"]

    logger.info(f"Partial answer received for session {request.session_id}: {request.answer}")

    async def check_answer_completion():
        await asyncio.sleep(DEBOUNCE_DELAY)
        if session_id not in interview_sessions or session.get("ended", False):
            logger.info(f"Session {session_id} ended or not found during debounce check")
            return
        if session.get("last_partial_timestamp") == current_time:
            logger.info(f"Debounce period expired for session {session_id}, processing answer: {last_answer}")
            result = await process_complete_answer(
                session_id=session_id,
                last_answer=last_answer,
                last_qa=last_qa,
                current_type=last_qa["type"],
                difficulty=session.get("difficulty", "Medium")
            )
            logger.info(f"Processed answer result for session {session_id}: {result}")
        else:
            logger.info(f"New partial answer received for session {session_id} during debounce period")

    session_id = request.session_id
    background_tasks.add_task(check_answer_completion)
    return {"success": True, "message": f"Partial answer noted, waiting {DEBOUNCE_DELAY} seconds for completion", "end_interview": False}

async def process_complete_answer(session_id: str, last_answer: str, last_qa: dict, current_type: str, difficulty: str):
    session = interview_sessions.get(session_id)
    if not session:
        logger.error(f"Session {session_id} not found during answer processing")
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Check if answer is too short (unclear)
    if len(last_answer.strip().split()) < 5:
        follow_up = "Could you provide more details or clarify your response?"
        session["qa_history"].append({"question": follow_up, "type": current_type})
        session["irrelevant_retries"] = 0  # Reset retries for unclear answer
        logger.info(f"Unclear/short answer detected for session {session_id}: {last_answer}")
        return {"success": True, "question": follow_up, "message": follow_up, "speak_only": True, "end_interview": False}

    # Check answer relevance with score
    relevance_score, reason = check_answer_relevance(
        question=last_qa["question"],
        answer=last_answer,
        resume_text=session["resume"],
        category=current_type
    )

    logger.info(f"Relevance check for session {session_id}: relevance_score={relevance_score}, reason={reason}, retries={session.get('irrelevant_retries', 0)}")

    if relevance_score < 30:
        session["irrelevant_retries"] = session.get("irrelevant_retries", 0) + 1
        logger.info(f"Incremented irrelevant retries for session {session_id} to {session['irrelevant_retries']}")
        if session["irrelevant_retries"] >= MAX_IRRELEVANT_RETRIES:
            follow_up = "Please focus on the question. Let's move to the next one."
            session["irrelevant_retries"] = 0  # Reset retries
            # Determine next category
            current_count = session["question_count"].get("general", 0)
            logger.info(f"Current general question count: {current_count}, phase: {session['phase']}")
            if session["phase"] == "general" and current_count < 3:
                next_category = "general"
            else:
                if session["phase"] != "technical_projects":
                    session["phase"] = "technical_projects"
                    logger.info(f"Transitioned to technical_projects phase for session {session_id}")
                next_category = "technical" if current_type == "projects" else "projects"
            try:
                next_question = run_interview(
                    session["resume"],
                    session["qa_history"],
                    next_category,
                    difficulty
                )
                session["qa_history"].append({"question": follow_up, "type": current_type})
                session["qa_history"].append({"question": next_question, "type": next_category})
                session["question_count"][next_category] = session["question_count"].get(next_category, 0) + 1
                logger.info(f"Max irrelevant retries reached for session {session_id}. New question: {next_question}, category: {next_category}, question_count: {session['question_count']}")
                return {"success": True, "question": next_question, "message": "", "speak_only": False, "end_interview": False}
            except Exception as e:
                logger.error(f"Failed to generate new question for session {session_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to generate new question: {str(e)}")
        else:
            follow_up = "I didn’t quite get how your answer relates to the question. Could you try again?"
            session["qa_history"].append({"question": follow_up, "type": current_type})
            logger.info(f"Irrelevant answer detected for session {session_id}, retries remaining: {MAX_IRRELEVANT_RETRIES - session['irrelevant_retries']}")
            return {"success": True, "question": follow_up, "message": follow_up, "speak_only": True, "end_interview": False}

    # Reset retries if answer is sufficiently relevant
    session["irrelevant_retries"] = 0
    logger.info(f"Reset irrelevant retries for session {session_id} due to relevant answer (score: {relevance_score})")

    # Decide next category
    current_count = session["question_count"].get("general", 0)
    if session["phase"] == "general" and current_count < 3:
        next_category = "general"
    else:
        if session["phase"] != "technical_projects":
            session["phase"] = "technical_projects"
            logger.info(f"Transitioned to technical_projects phase for session {session_id}")
        next_category = "technical" if current_type == "projects" else "projects"

    # Generate next question
    try:
        next_question = run_interview(
            session["resume"],
            session["qa_history"],
            next_category,
            difficulty
        )
        session["qa_history"].append({"question": next_question, "type": next_category})
        session["question_count"][next_category] = session["question_count"].get(next_category, 0) + 1
        logger.info(f"Generated next question for session {session_id}: {next_question}, category: {next_category}, question_count: {session['question_count']}")
        return {"success": True, "question": next_question, "message": "", "speak_only": False, "end_interview": False}
    except Exception as e:
        logger.error(f"Failed to generate next question for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate new question: {str(e)}")

@app.post("/end-interview")
async def end_interview(request: EndInterviewRequest):
    return await end_interview_internal(request.session_id)

@app.get("/get-feedback/{session_id}")
async def get_feedback(session_id: str):
    try:
        session = interview_sessions.get(session_id)
        if not session or not session.get("ended", False):
            logger.error(f"Feedback not found or interview not ended for session {session_id}")
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
    if session_id in interview_sessions and interview_sessions[session_id].get("ended", False):
        del interview_sessions[session_id]
        if session_id in chat_histories:
            del chat_histories[session_id]
        logger.info(f"Cleaned up session {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)