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
  const [timeLeft, setTimeLeft] = useState(15 * 60); // 15 minutes
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const endButtonRef = useRef(null);

  useEffect(() => {
    const id = location.state?.sessionId;
    if (id && typeof id === "string") {
      setSessionId(id);
    } else {
      setError("Invalid or missing session ID. Redirecting to resume upload...");
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
      utterance.rate = 0.8; // Slower speech rate
      utterance.onend = () => {
        setIsBotSpeaking(false);
        if (started && !interviewEnded && 'webkitSpeechRecognition' in window) {
          setTimeout(startListening, 2000); // 2-second delay
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
    if (!('webkitSpeechRecognition' in window)) {
      setError("Speech recognition not supported. Use Chrome.");
      return;
    }
    setIsListening(true);
    const recognition = new window.webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";
    recognition.interimResults = true;

    recognition.onresult = (event) => {
      let finalTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript;
      }
      if (finalTranscript) {
        setAnswer(finalTranscript);
        submitAnswer(finalTranscript, true);
        setChatHistory((prev) => [...prev, { role: "user", text: finalTranscript }]);
        setAnswer("");
      }
    };
    recognition.onend = () => {
      setIsListening(false);
      if (recognition) recognition.stop();
    };
    recognition.onerror = (event) => {
      setIsListening(false);
      setError(`Speech recognition failed: ${event.error}`);
      if (recognition) recognition.stop();
    };

    recognition.start();
  }, [started, interviewEnded]);

  const submitAnswer = useCallback(async (answerText) => {
    if (!answerText || !sessionId) return;
    try {
      const response = await axios.post("http://localhost:8000/submit-answer", {
        session_id: sessionId,
        answer: answerText,
        is_complete: true,
      });
      console.log("Submit Answer Response:", response.data);
      if (response.data.end_interview) {
        setInterviewEnded(true);
        const feedbackResponse = await axios.post("http://localhost:8000/end-interview", { session_id: sessionId });
        console.log("Feedback Response:", feedbackResponse.data);
        if (feedbackResponse.data.success && feedbackResponse.data.feedback) {
          navigate("/feedback", { state: { feedback: feedbackResponse.data.feedback } });
        } else {
          setError("Failed to retrieve feedback.");
          navigate("/feedback");
        }
      } else if (response.data.question) {
        setQuestion(response.data.question);
      }
    } catch (error) {
      setError(`Error submitting answer: ${error.response?.data?.detail || error.message}`);
    }
  }, [sessionId, navigate]);

  const handleStart = async () => {
    if (!sessionId) {
      setError("Session ID missing. Please upload resume again.");
      return;
    }
    try {
      const response = await axios.post("http://localhost:8000/start-interview", { session_id: sessionId });
      setQuestion(response.data.question);
      setStarted(true);
      setError("");
    } catch (error) {
      setError(`Error starting interview: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleEnd = async () => {
    if (endButtonRef.current) {
      endButtonRef.current.disabled = true; // Prevent multiple clicks
    }
    try {
      speechSynthesis.cancel();
      const response = await axios.post("http://localhost:8000/end-interview", { session_id: sessionId });
      console.log("End Interview Response:", response.data);
      setInterviewEnded(true);
      if (response.data.success && response.data.feedback) {
        navigate("/feedback", { state: { feedback: response.data.feedback } });
      } else {
        setError("Failed to retrieve feedback.");
        navigate("/feedback");
      }
    } catch (error) {
      setError(`Error ending interview: ${error.response?.data?.detail || error.message}`);
      navigate("/feedback");
    } finally {
      if (endButtonRef.current) {
        endButtonRef.current.disabled = false; // Re-enable after request
      }
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

  if (!window.speechSynthesis || !('webkitSpeechRecognition' in window)) {
    return <p className="text-red-500">Browser does not support speech features. Use Chrome and check permissions.</p>;
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className={`w-64 bg-gray-800 p-4 text-white transition-all duration-300 ${chatOpen ? 'block' : 'hidden'}`}>
        <h2 className="text-xl font-bold mb-4">Chat History</h2>
        <div className="space-y-2 max-h-[calc(100vh-120px)] overflow-y-auto">
          {chatHistory.map((msg, index) => (
            <div key={index} className={`p-2 rounded ${msg.role === "bot" ? "bg-blue-700" : "bg-green-700"}`}>
              <p className="font-semibold">{msg.role === "bot" ? "Bot" : "You"}:</p>
              <p>{msg.text}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Main Area */}
      <div className="flex-1 bg-[#1e1f20] flex flex-col items-center justify-center text-white p-6 relative">
        {error && <p className="text-red-500 mb-4">{error}</p>}
        {!started ? (
          <div className="text-center">
            <h1 className="text-3xl font-bold mb-4">🎤 Interview Session</h1>
            <button
              onClick={handleStart}
              className="bg-indigo-600 px-6 py-3 rounded-xl text-lg font-semibold hover:bg-indigo-700 transition"
              disabled={!sessionId}
            >
              Join Interview
            </button>
          </div>
        ) : (
          <div className="w-full max-w-2xl">
            {interviewEnded ? (
              <div className="text-center">
                <h2 className="text-2xl font-bold mb-4">Interview Completed</h2>
                <button
                  ref={endButtonRef}
                  onClick={handleEnd}
                  className="bg-indigo-600 px-6 py-3 rounded-xl text-lg font-semibold hover:bg-indigo-700 transition"
                >
                  View Feedback
                </button>
              </div>
            ) : (
              <div className="space-y-6">
                <p className="text-gray-300 text-center">
                  Time Remaining: {Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, "0")}
                </p>
                {isBotSpeaking && (
                  <div className="bg-blue-600 text-white p-4 rounded-lg max-w-prose bubble-animation">
                    <p className="font-semibold">Bot Speaking...</p>
                  </div>
                )}
                {answer && (
                  <div className="bg-green-600 text-white p-4 rounded-lg max-w-prose">
                    <p className="font-semibold">You:</p>
                    <p>{answer}</p>
                  </div>
                )}
                {isListening && !answer && (
                  <div className="mt-4 flex items-center">
                    <Mic size={24} className="text-green-500 mr-2" />
                    <p className="text-green-500 font-semibold">Listening... Speak now!</p>
                  </div>
                )}
                <div className="absolute bottom-10 right-6">
                  <button
                    ref={endButtonRef}
                    onClick={handleEnd}
                    className="bg-red-600 p-4 rounded-full hover:bg-red-700 transition mr-4"
                    title="End Interview"
                  >
                    <PhoneOff size={24} />
                  </button>
                  <button
                    onClick={() => setChatOpen(!chatOpen)}
                    className="bg-gray-600 p-4 rounded-full hover:bg-gray-700 transition"
                    title="Toggle Chat"
                  >
                    <MessageCircle size={24} />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default InterviewPage;