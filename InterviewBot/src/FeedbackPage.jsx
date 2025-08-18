import React from "react";
import { useLocation, useNavigate } from "react-router-dom";

const Feedback = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { feedback, isError } = location.state || { feedback: "No feedback available." };

  // Parse feedback into sections
  const parseFeedback = (text) => {
    const sections = {
      evaluations: [], // Array of evaluation sections (e.g., Leadership Experience, Coding Languages)
      strengths: [],
      improvements: [],
      actionable: "",
    };
    const lines = text.split("\n").filter(line => line.trim());
    let currentSection = "evaluations";
    let currentEval = {};

    lines.forEach(line => {
      if (line.startsWith("Thank you for participating")) {
        return; // Skip introductory line
      } else if (line.match(/^\d+\.\s+\*\*[A-Za-z\s]+\*\*:/)) {
        if (Object.keys(currentEval).length > 0) {
          sections.evaluations.push(currentEval);
        }
        currentEval = { title: line.replace(/^\d+\.\s+\*\*(.+)\*\*:/, "$1").trim(), details: {} };
        currentSection = "evaluations";
      } else if (line.startsWith("- **Relevance:**")) {
        currentEval.details.relevance = line.replace("- **Relevance:**", "").trim();
      } else if (line.startsWith("- **Clarity:**")) {
        currentEval.details.clarity = line.replace("- **Clarity:**", "").trim();
      } else if (line.startsWith("- **Depth:**")) {
        currentEval.details.depth = line.replace("- **Depth:**", "").trim();
      } else if (line.startsWith("- **Feedback:**")) {
        currentEval.details.feedback = line.replace("- **Feedback:**", "").trim();
      } else if (line.startsWith("**Overall Strengths:**")) {
        currentSection = "strengths";
      } else if (line.startsWith("**Areas for Improvement:**")) {
        currentSection = "improvements";
      } else if (line.startsWith("Keep practicing")) {
        currentSection = "actionable";
      } else if (currentSection === "evaluations" && line.trim() && Object.keys(currentEval).length > 0) {
        // Handle multi-line feedback
        currentEval.details.feedback = (currentEval.details.feedback || "") + " " + line.trim();
      } else if (currentSection === "strengths" && line.trim().startsWith("-")) {
        sections.strengths.push(line.replace("-", "").trim());
      } else if (currentSection === "improvements" && line.trim().startsWith("-")) {
        sections.improvements.push(line.replace("-", "").trim());
      } else if (currentSection === "actionable" && line.trim()) {
        sections.actionable = (sections.actionable || "") + " " + line.trim();
      }
    });

    if (Object.keys(currentEval).length > 0) {
      sections.evaluations.push(currentEval);
    }

    return sections;
  };

  const feedbackSections = parseFeedback(feedback);

  return (
    <div className="flex min-h-screen bg-gray-100">
      <div className="w-1/4 bg-indigo-800 text-white p-8 space-y-6 rounded-r-3xl shadow-md">
        <h1 className="text-2xl font-bold">Menu</h1>
        <ul className="space-y-4 text-sm">
          <li className="text-gray-300 hover:text-white cursor-pointer" onClick={() => navigate("/")}>
            Home
          </li>
          <li className="text-gray-300 hover:text-white cursor-pointer" onClick={() => navigate("/resume-upload")}>
            Resume Upload
          </li>
          <li className="text-gray-300 hover:text-white cursor-pointer" onClick={() => navigate("/interview")}>
            Interview
          </li>
          <li className="text-white font-semibold">Feedback</li>
        </ul>
      </div>

      <div className="w-3/4 p-12 flex flex-col justify-center items-start space-y-8">
        <h2 className="text-3xl font-bold text-gray-800">Interview Feedback</h2>
        {isError ? (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <span className="block sm:inline">{feedback}</span>
          </div>
        ) : (
          <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-2xl space-y-6">
            <div>
              <h3 className="text-xl font-semibold text-gray-700">Evaluations:</h3>
              {feedbackSections.evaluations.map((evalItem, index) => (
                <div key={index} className="mt-4">
                  <h4 className="text-lg font-medium text-gray-600">{evalItem.title}</h4>
                  <ul className="list-disc pl-5 mt-2 space-y-2">
                    <li><strong>Relevance:</strong> {evalItem.details.relevance || "Not provided"}</li>
                    <li><strong>Clarity:</strong> {evalItem.details.clarity || "Not provided"}</li>
                    <li><strong>Depth:</strong> {evalItem.details.depth || "Not provided"}</li>
                    <li><strong>Feedback:</strong> {evalItem.details.feedback || "Not provided"}</li>
                  </ul>
                </div>
              ))}
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-700">Strengths:</h3>
              <ul className="list-disc pl-5 mt-2 space-y-2">
                {feedbackSections.strengths.length > 0 ? (
                  feedbackSections.strengths.map((strength, index) => <li key={index}>{strength}</li>)
                ) : (
                  <li>No strengths highlighted</li>
                )}
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-700">Areas for Improvement:</h3>
              <ul className="list-disc pl-5 mt-2 space-y-2">
                {feedbackSections.improvements.length > 0 ? (
                  feedbackSections.improvements.map((improvement, index) => <li key={index}>{improvement}</li>)
                ) : (
                  <li>No areas for improvement noted</li>
                )}
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-700">Actionable Feedback:</h3>
              <p className="mt-2">{feedbackSections.actionable || "No actionable feedback provided"}</p>
            </div>
          </div>
        )}
        <button
          onClick={() => navigate("/")}
          className="mt-6 bg-indigo-600 text-white px-6 py-3 rounded-xl text-lg font-semibold hover:bg-indigo-700 transition"
        >
          Back to Home
        </button>
      </div>
    </div>
  );
};

export default Feedback;