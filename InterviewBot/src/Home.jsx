import React from "react";
import { useNavigate } from "react-router-dom";
import logo from "./assets/Logo.jpg"; // make sure to use your logo path

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen bg-white">
      {/* Sidebar */}
      <div className="w-64 bg-gray-100 p-6">
        <h2 className="text-xl font-bold mb-6">&laquo;</h2>
        <ul className="space-y-4">
          <li className="text-blue-700 font-semibold">Home</li>
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer">Resume Upload</li>
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer">Interview</li>
          <li className="text-gray-700 hover:text-blue-600 cursor-pointer">Feedback</li>
        </ul>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-10">
        <div className="text-4xl font-bold text-gray-900 mb-2">
          🎯 Your perfect interview starts here
        </div>
        <div className="text-lg text-gray-600 mb-8">
          Practice with AI-powered mock interviews, get personalized feedback, and land your dream job.
        </div>

        {/* Card */}
        <div className="flex justify-center">
          <div className="w-80 bg-white rounded-2xl shadow-md p-6 text-center transition transform hover:scale-105 border">
            <div className="bg-gray-200 p-6 rounded-xl mb-4 flex justify-center">
              <img src={logo} alt="logo" className="w-16 h-16 rounded-md" />
            </div>
            <div className="text-md font-semibold mb-1">Pick Your Interview Challenge</div>
            <div className="text-sm text-gray-600 mb-3">
              No pressure, just progress. Pick a topic and begin.
            </div>
            <div className="flex justify-center items-center gap-3 text-sm mb-4">
              <div>⏱️ 15m</div>
              <span className="bg-yellow-200 px-3 py-1 rounded-full">Medium</span>
            </div>
            <button
              onClick={() => navigate("/resume")}
              className="bg-black text-white py-2 px-6 rounded-lg hover:bg-gray-800"
            >
              🚀 Start Interview
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
