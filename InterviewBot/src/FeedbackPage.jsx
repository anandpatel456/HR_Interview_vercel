import React from "react";
import { useLocation, useNavigate } from "react-router-dom";

const FeedbackPage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  // Get feedback data passed from InterviewPage
  const { feedback, isError } = location.state || {
    feedback: "No feedback available.",
    isError: false
  };

  // Normalize to string
  const feedbackText =
    typeof feedback === "string" ? feedback : JSON.stringify(feedback);

  // Split into trimmed lines
  const lines = feedbackText.split("\n").filter(line => line.trim());

  // Helper: extract only the content after a given header
  function extractSectionContent(lines, header) {
    const start = lines.findIndex(
      line => line.trim().toLowerCase() === header.toLowerCase()
    );
    if (start === -1) return "";
    let end = start + 1;
    while (end < lines.length && !lines[end].startsWith("**")) end++;
    return lines.slice(start + 1, end).join("\n").trim();
  }

  // Grab just the content under each section
  const strengths = extractSectionContent(lines, "Strengths:");
  const areasToImprove = extractSectionContent(lines, "Areas to Improve:");
  const clarityComments = extractSectionContent(
    lines,
    "Comments on Clarity and Confidence:"
  );

  const isStructured = strengths || areasToImprove || clarityComments;

  if (!feedbackText.trim()) {
    return (
      <div className="text-center p-6 text-red-500">
        No feedback available.
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-900 text-white p-6">
      <div className="flex-1 flex flex-col items-center justify-center">
        <h1 className="text-3xl font-bold mb-6">Interview Feedback</h1>
        <div className="max-w-2xl w-full">
          {isError ? (
            <div className="mb-4 p-4 bg-red-800 rounded-lg break-words">
              <p className="text-red-200">Error: {feedbackText}</p>
            </div>
          ) : isStructured ? (
            <div className="mb-4 p-4 bg-gray-800 rounded-lg break-words">
              {strengths && (
                <div className="mt-4">
                  <strong className="text-yellow-400">Strengths:</strong>
                  <pre className="text-gray-300 whitespace-pre-wrap">
                    {strengths}
                  </pre>
                </div>
              )}
              {areasToImprove && (
                <div className="mt-4">
                  <strong className="text-yellow-400">Areas to Improve:</strong>
                  <pre className="text-gray-300 whitespace-pre-wrap">
                    {areasToImprove}
                  </pre>
                </div>
              )}
              {clarityComments && (
                <div className="mt-4">
                  <strong className="text-yellow-400">
                    Comments on Clarity and Confidence:
                  </strong>
                  <pre className="text-gray-300 whitespace-pre-wrap">
                    {clarityComments}
                  </pre>
                </div>
              )}
            </div>
          ) : (
            <div className="mb-4 p-4 bg-gray-800 rounded-lg break-words">
              <p className="text-gray-200">Feedback:</p>
              <pre className="text-gray-300 whitespace-pre-wrap">
                {feedbackText}
              </pre>
            </div>
          )}

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
