
import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { PDFDocument } from "pdf-lib"; // For client-side PDF compression
import axios from "axios"; // For progress tracking and retries
import axiosRetry from "axios-retry";

axiosRetry(axios, { retries: 3, retryDelay: axiosRetry.exponentialDelay });

const ResumeUpload = () => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();
  const location = useLocation();
  const difficulty = location.state?.difficulty || "Medium";
  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
  const BACKEND_URL = "https://mock-interview-backend-fmu8.onrender.com"; // Backend URL

  // Compress PDF using pdf-lib
  const compressPDF = async (file) => {
    try {
      const pdfBytes = await file.arrayBuffer();
      const pdfDoc = await PDFDocument.load(pdfBytes);
      const compressedPdfBytes = await pdfDoc.save({ useObjectStreams: true });
      return new File([compressedPdfBytes], file.name, { type: "application/pdf" });
    } catch (error) {
      console.error("Error compressing PDF:", error);
      throw new Error("Failed to compress PDF.");
    }
  };

  // Poll backend to check if resume processing is complete
  const pollResumeStatus = async (sessionId) => {
    const maxAttempts = 30; // Poll for up to 30 seconds (2s interval)
    let attempts = 0;

    const checkStatus = async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/status/${sessionId}`);
        if (response.data.success && response.data.resume_processed) {
          setStatus("✅ Resume processed successfully! Redirecting...");
          setTimeout(() => {
            navigate("/GuidelineInterview", { state: { sessionId, difficulty } });
          }, 1000);
        } else {
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(checkStatus, 2000); // Poll every 2 seconds
          } else {
            setStatus("❌ Timeout: Resume processing took too long.");
          }
        }
      } catch (error) {
        setStatus("❌ Error checking resume status: " + error.message);
      }
    };

    checkStatus();
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus("Please upload a PDF first.");
      return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      setStatus("❌ File size exceeds 5MB limit. Please upload a smaller file.");
      return;
    }

    setStatus("📃 Compressing resume...");
    let uploadFile = file;

    // Compress PDF if larger than 1MB
    if (file.size > 1024 * 1024) {
      try {
        uploadFile = await compressPDF(file);
        setStatus("📃 Uploading compressed resume...");
      } catch (error) {
        setStatus("❌ Error compressing PDF: " + error.message);
        return;
      }
    } else {
      setStatus("📃 Uploading resume...");
    }

    const formData = new FormData();
    formData.append("resume", uploadFile);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/upload-resume`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 30000, // 30s timeout
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          },
        }
      );

      const data = response.data;

      if (data.success) {
        setStatus("✅ Resume uploaded! Processing...");
        pollResumeStatus(data.session_id);
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
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer" onClick={() => navigate("/")}>
            Home
          </li>
          <li className="text-blue-700 font-semibold">Resume Upload</li>
          <li
            className="text-gray-700 hover:text-blue-600 cursor-pointer"
            onClick={() => navigate("/interview")}
          >
            Interview
          </li>
          <li
            className="text-gray-700 hover:text-blue-600 cursor-pointer"
            onClick={() => navigate("/feedback")}
          >
            Feedback
          </li>
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
            disabled={status.includes("Uploading") || status.includes("Processing")}
          >
            Upload and start interview
          </button>

          {status && (
            <p className="mt-4 text-sm text-center text-gray-700">{status}</p>
          )}

          {uploadProgress > 0 && uploadProgress < 100 && (
            <div className="w-full max-w-xs mt-4">
              <div className="bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-blue-600 h-2.5 rounded-full"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p className="text-sm text-center text-gray-600 mt-1">{uploadProgress}% Uploaded</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResumeUpload;
