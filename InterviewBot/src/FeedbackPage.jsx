import React from "react";
import { useLocation, useNavigate } from "react-router-dom";

const FeedbackPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const feedbackData = location.state?.feedback || {}; // Default to empty object if undefined
  console.log("Feedback Data:", feedbackData); // Debug log

  // Ensure feedbackText is extracted from the feedback object
  const feedbackText = typeof feedbackData === "string" ? feedbackData : (feedbackData.feedback || "");
  if (!feedbackText.trim()) {
    return <div className="text-center p-6 text-red-500">No feedback available.</div>;
  }

  // Split feedback into lines and filter out empty ones
  const feedbackLines = feedbackText.split("\n").filter(line => line.trim());
  const intro = feedbackLines[0] || "";
  const strengths = feedbackLines.find(line => line.startsWith("**Strengths:**")) || "";
  const areasToImprove = feedbackLines.find(line => line.startsWith("**Areas to improve:**")) || "";
  const clarityComments = feedbackLines.find(line => line.startsWith("**Comments on clarity and confidence:**")) || "";

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="w-64 bg-gray-100 p-6">
        <h2 className="text-xl font-bold mb-6">&laquo;</h2>
        <ul className="space-y-4">
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer" onClick={() => navigate("/")}>Home</li>
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer">Resume Upload</li>
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer">Interview</li>
          <li className="text-blue-700 font-semibold">Feedback</li>
        </ul>
      </div>

      {/* Main Area */}
      <div className="flex-1 bg-[#1e1f20] p-6 text-white overflow-auto">
        <h1 className="text-3xl font-bold mb-6">Interview Feedback</h1>
        <div className="max-w-2xl">
          <h2 className="text-xl font-semibold mb-4">Feedback</h2>
          <div className="mb-4 p-4 bg-gray-800 rounded-lg break-words">
            <p className="text-gray-200">{intro}</p>
            {strengths && <p className="text-gray-200 mt-4"><strong>{strengths}</strong></p>}
            {areasToImprove && <p className="text-gray-200 mt-4"><strong>{areasToImprove}</strong></p>}
            {clarityComments && <p className="text-gray-200 mt-4"><strong>{clarityComments}</strong></p>}
            {!strengths && !areasToImprove && !clarityComments && (
              <>
                <p className="text-gray-400 mt-4">Feedback format not recognized. Displaying raw text:</p>
                <pre className="text-gray-300">{feedbackText}</pre>
              </>
            )}
          </div>
          <button
            onClick={() => navigate("/")}
            className="bg-indigo-600 px-6 py-3 rounded-xl text-lg font-semibold hover:bg-indigo-700 transition mt-4"
          >
            Back to Home
          </button>
        </div>
      </div>
    </div>
  );
};

export default FeedbackPage;