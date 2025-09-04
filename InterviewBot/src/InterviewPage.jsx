import React, { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { PhoneOff, Mic, MessageCircle } from "lucide-react";
import axios from "axios";

const InterviewPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [started, setStarted] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [interviewEnded, setInterviewEnded] = useState(false);
  const [error, setError] = useState("");
  const [timeLeft, setTimeLeft] = useState(15 * 60);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [difficulty, setDifficulty] = useState("Medium");
  const endButtonRef = useRef(null);
  const BACKEND_URL = "https://hr-interview-vercel.vercel.app"; // Correct backend URL

  // Get sessionId, firstQuestion, difficulty
  useEffect(() => {
    const id = location.state?.sessionId;
    const firstQ = location.state?.firstQuestion;
    const diff = location.state?.difficulty || "Medium";
    setDifficulty(diff);

    if (id && typeof id === "string") {
      console.log(`InterviewPage loaded with sessionId: ${id}, firstQuestion: ${firstQ}, difficulty: ${diff}`);
      setSessionId(id);
      if (firstQ) {
        setQuestion(firstQ);
        setStarted(true);
      }
    } else {
      setError("Invalid or missing session ID. Redirecting to resume upload...");
      console.error("No valid sessionId provided");
      setTimeout(() => navigate("/resume-upload"), 3000);
    }
  }, [location, navigate]);

  useEffect(() => {
    if (started && !interviewEnded) {
      const timer = setInterval(() => {
        setTimeLeft((prev) => (prev <= 0 ? (handleEnd(), 0) : prev - 1));
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [started, interviewEnded]);

  useEffect(() => {
    if (question && started && !interviewEnded && !isBotSpeaking) {
      setIsBotSpeaking(true);
      const utterance = new SpeechSynthesisUtterance(question);
      utterance.lang = "en-US";
      utterance.rate = 0.8;
      utterance.onend = () => {
        setIsBotSpeaking(false);
        if (started && !interviewEnded && "webkitSpeechRecognition" in window) {
          setTimeout(startListening, 2000);
        }
      };
      utterance.onerror = () => {
        setIsBotSpeaking(false);
        setError("Error playing question. Please try again.");
      };
      speechSynthesis.speak(utterance);
      setChatHistory((prev) => [...prev, { role: "bot", text: question }]);
    }
  }, [question, started, interviewEnded]);

  const startListening = useCallback(() => {
    if (!("webkitSpeechRecognition" in window)) {
      setError("Speech recognition not supported. Use Chrome.");
      console.error("Speech recognition not supported");
      return;
    }
    setIsListening(true);
    const recognition = new window.webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";
    recognition.interimResults = true;

    recognition.onresult = (event) => {
      let finalTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript;
      }
      if (finalTranscript) {
        setAnswer(finalTranscript);
        submitAnswer(finalTranscript);
        setChatHistory((prev) => [...prev, { role: "user", text: finalTranscript }]);
        setAnswer("");
      }
    };
    recognition.onend = () => setIsListening(false);
    recognition.onerror = (event) => {
      setIsListening(false);
      setError(`Speech recognition failed: ${event.error}`);
      console.error(`Speech recognition error: ${event.error}`);
    };

    recognition.start();
  }, [started, interviewEnded]);

  const submitAnswer = useCallback(async (answerText) => {
    if (!answerText || !sessionId) {
      console.error("No answer or sessionId provided for submitAnswer");
      return;
    }
    try {
      console.log(`Submitting answer for sessionId: ${sessionId}, answer: ${answerText}`);
      const response = await axios.post(
        `${BACKEND_URL}/submit-answer`,
        { session_id: sessionId, answer: answerText, is_complete: true },
        { headers: { "Content-Type": "application/json" }, timeout: 10000 }
      );
      if (response.data.end_interview) {
        setInterviewEnded(true);
        console.log("Interview ended, fetching feedback...");
        const feedbackResponse = await axios.post(
          `${BACKEND_URL}/end-interview`,
          { session_id: sessionId },
          { headers: { "Content-Type": "application/json" }, timeout: 10000 }
        );
        if (feedbackResponse.data.success && feedbackResponse.data.feedback) {
          navigate("/feedback", { state: { feedback: feedbackResponse.data.feedback } });
        } else {
          setError("Failed to retrieve feedback.");
          console.error("No feedback received");
          navigate("/feedback");
        }
      } else if (response.data.question) {
        setQuestion(response.data.question);
        console.log("Received next question:", response.data.question);
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      setError(`Error submitting answer: ${errorMessage}`);
      console.error("Error submitting answer:", errorMessage);
    }
  }, [sessionId, navigate]);

  const handleEnd = async () => {
    if (endButtonRef.current) endButtonRef.current.disabled = true;
    try {
      speechSynthesis.cancel();
      console.log(`Ending interview for sessionId: ${sessionId}`);
      const response = await axios.post(
        `${BACKEND_URL}/end-interview`,
        { session_id: sessionId },
        { headers: { "Content-Type": "application/json" }, timeout: 10000 }
      );
      setInterviewEnded(true);
      navigate("/feedback", { state: { feedback: response.data.feedback || "No feedback available." } });
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      setError(`Error ending interview: ${errorMessage}`);
      console.error("Error ending interview:", errorMessage);
      navigate("/feedback", { state: { feedback: errorMessage, isError: true } });
    } finally {
      if (endButtonRef.current) endButtonRef.current.disabled = false;
    }
  };

  useEffect(() => {
    const styleSheet = document.createElement("style");
    styleSheet.type = "text/css";
    styleSheet.innerText = `
      @keyframes bubble {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
      }
      .bubble-animation { animation: bubble 1.5s ease-in-out infinite; }
    `;
    document.head.appendChild(styleSheet);
    return () => document.head.removeChild(styleSheet);
  }, []);

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className={`w-64 bg-gray-800 p-4 text-white ${chatOpen ? 'block' : 'hidden'}`}>
        <h2 className="text-xl font-bold mb-4">Chat History</h2>
        <div className="space-y-2 max-h-[calc(100vh-120px)] overflow-y-auto">
          {chatHistory.map((msg, idx) => (
            <div key={idx} className={`p-2 rounded ${msg.role === "bot" ? "bg-blue-700" : "bg-green-700"}`}>
              <p className="font-semibold">{msg.role === "bot" ? "Bot" : "You"}:</p>
              <p>{msg.text}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 bg-[#1e1f20] flex flex-col justify-center items-center text-white p-6 relative">
        {error && <p className="text-red-500 mb-4">{error}</p>}
        {interviewEnded ? (
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4">Interview Completed</h2>
            <button onClick={handleEnd} ref={endButtonRef} className="bg-indigo-600 px-6 py-3 rounded-xl">
              View Feedback
            </button>
          </div>
        ) : (
          <div className="space-y-6 w-full max-w-2xl">
            <p className="text-gray-300 text-center">
              Time Remaining: {Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, "0")} | Difficulty: {difficulty}
            </p>
            {isBotSpeaking && <div className="bg-blue-600 p-4 rounded-lg bubble-animation">Bot Speaking...</div>}
            {answer && <div className="bg-green-600 p-4 rounded-lg"><p className="font-semibold">You:</p><p>{answer}</p></div>}
            {isListening && !answer && <div className="mt-4 flex items-center"><Mic size={24} className="text-green-500 mr-2" />Listening... Speak now!</div>}
            <div className="absolute bottom-10 right-6">
              <button onClick={handleEnd} className="bg-red-600 p-4 rounded-full mr-4"><PhoneOff size={24} /></button>
              <button onClick={() => setChatOpen(!chatOpen)} className="bg-gray-600 p-4 rounded-full"><MessageCircle size={24} /></button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InterviewPage;
