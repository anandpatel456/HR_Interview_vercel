import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Mic } from "lucide-react";
import axios from "axios";

const GuidelineInterview = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const sessionId = location.state?.sessionId;
  const difficulty = location.state?.difficulty || "Medium";
  const BACKEND_URL = "https://mock-interview-backend-fmu8.onrender.com"; // Correct backend URL

  const handleStart = async () => {
    if (!sessionId) {
      console.error("No sessionId provided, redirecting to resume-upload");
      navigate("/resume-upload");
      return;
    }

    try {
      console.log(`Starting interview with sessionId: ${sessionId}, difficulty: ${difficulty}`);
      const response = await axios.post(
        `${BACKEND_URL}/start-interview`,
        { session_id: sessionId, difficulty },
        { headers: { "Content-Type": "application/json" }, timeout: 10000 }
      );

      if (response.data.success) {
        console.log("Interview started successfully:", response.data);
        navigate("/interview", {
          state: {
            sessionId,
            difficulty,
            firstQuestion: response.data.question,
          },
        });
      } else {
        console.error("Failed to start interview:", response.data.detail);
        alert(`Failed to start interview: ${response.data.detail}`);
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      console.error("Error starting interview:", errorMessage);
      alert(`Failed to start interview: ${errorMessage}`);
    }
  };

  return (
    <div className="flex min-h-screen bg-gray-100">
      <div className="w-1/4 bg-indigo-800 text-white p-8 space-y-6 rounded-r-3xl shadow-md">
        <h1 className="text-2xl font-bold">Interview Guidelines</h1>
        <ul className="space-y-4 text-sm">
          <li>✔️ Be in a quiet place</li>
          <li>✔️ Ensure stable internet</li>
          <li>✔️ Allow microphone access</li>
          <li>✔️ Speak clearly</li>
          <li>✔️ Be confident and relaxed</li>
        </ul>
      </div>

      <div className="w-3/4 p-12 flex flex-col justify-center items-start space-y-6">
        <h2 className="text-3xl font-bold text-gray-800">Get Ready for Your AI Interview</h2>
        <p className="text-gray-600 text-md">
          The interview will take around 10-15 minutes. Make sure you're prepared with the right setup.
        </p>

        <div className="flex items-center gap-4 bg-white p-6 rounded-xl shadow">
          <div className="bg-red-500 p-4 rounded-full">
            <Mic className="text-white h-6 w-6" />
          </div>
          <div>
            <p className="text-gray-800 font-semibold text-lg">
              Microphone permission required
            </p>
            <p className="text-gray-500 text-sm">
              Please allow access to your microphone to proceed.
            </p>
          </div>
        </div>

        <button
          onClick={handleStart}
          className="mt-4 bg-indigo-600 text-white px-6 py-3 rounded-xl text-lg font-semibold hover:bg-indigo-700 transition"
        >
          Start Interview
        </button>

        <p className="text-xs text-gray-500 mt-4 max-w-lg">
          Your voice responses are used only to assess your application and will never be used for
          training any AI model.
        </p>
      </div>
    </div>
  );
};

export default GuidelineInterview;