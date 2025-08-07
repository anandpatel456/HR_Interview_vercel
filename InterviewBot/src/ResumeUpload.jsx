import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const ResumeUpload = () => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const navigate = useNavigate();

  const handleUpload = async () => {
    if (!file) {
      setStatus("Please upload a PDF first.");
      return;
    }

    setStatus("📃 Uploading resume...");

    const formData = new FormData();
    formData.append("resume", file);

    try {
      const response = await fetch("http://localhost:8000/upload-resume", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setStatus("✅ Resume uploaded successfully! Redirecting...");
        setTimeout(() => {
          navigate("/GuidelineInterview", { state: { sessionId: data.session_id } });
        }, 2000);
      } else {
        setStatus("❌ Error: " + data.error);
      }
    } catch (error) {
      setStatus("❌ Server Error: " + error.message);
    }
  };

  return (
    <div className="flex min-h-screen bg-white">
      {/* Sidebar */}
      <div className="w-64 bg-gray-100 p-6">
        <h2 className="text-xl font-bold mb-6">&laquo;</h2>
        <ul className="space-y-4">
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer" onClick={() => navigate("/")}>Home</li>
          <li className="text-blue-700 font-semibold">Resume Upload</li>
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer" onClick={() => navigate("/interview")}>Interview</li>
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer" onClick={() => navigate("/feedback")}>Feedback</li>
        </ul>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center justify-center px-4">
        <h1 className="text-4xl font-bold text-center mb-2">AI Voice Interview</h1>
        <p className="text-gray-600 text-center mb-8">
          “Unlock Your Potential with AI” – Suggests personal growth through the system.
        </p>

        <div className="border-2 border-dashed border-blue-300 p-10 rounded-xl w-full max-w-md flex flex-col items-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-12 w-12 text-blue-500 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M12 12v8m0-8l-4 4m4-4l4 4M12 4v4m0-4l-4 4m4-4l4 4"
            />
          </svg>

          <p className="mb-2 font-medium">Upload your resume (PDF)</p>

          <input
            type="file"
            accept=".pdf"
            onChange={(e) => {
              const selectedFile = e.target.files[0];
              if (selectedFile) setFile(selectedFile);
            }}
            className="mb-4"
          />

          {file && <p className="text-sm text-gray-500 mb-2">Selected: {file.name}</p>}

          <button
            onClick={handleUpload}
            className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 transition"
          >
            Upload and start interview
          </button>

          {status && (
            <p className="mt-4 text-sm text-center text-gray-700">{status}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResumeUpload;