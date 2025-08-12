import React from "react";
import { useNavigate } from "react-router-dom";

const Home = () => {
  const navigate = useNavigate();

  const interviewCards = [
    {
      level: "Easy",
      title: "Beginner Challenge",
      description: "Basic introductory questions — perfect for first-time interview practice.",
      duration: "15m",
      color: "bg-green-200",
      textColor: "text-green-800",
      borderColor: "#86efac",
      emoji: "🌱"
    },
    {
      level: "Medium",
      title: "Standard Challenge",
      description: "Balanced difficulty for experienced candidates. Test your skills.",
      duration: "15m",
      color: "bg-yellow-200",
      textColor: "text-yellow-800",
      borderColor: "#fde047",
      emoji: "⚡"
    },
    {
      level: "Hard",
      title: "Expert Challenge",
      description: "Tough and detailed questions for seasoned professionals.",
      duration: "15m",
      color: "bg-red-200",
      textColor: "text-red-800",
      borderColor: "#fca5a5",
      emoji: "🔥"
    }
  ];

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

        {/* Cards Container */}
        <div className="flex justify-center gap-6 flex-wrap">
          {interviewCards.map((card, index) => (
            <div
              key={index}
              className="w-80 bg-white rounded-2xl shadow-md p-6 text-center transition transform hover:scale-105 border-2 hover:shadow-lg"
              style={{ borderColor: card.borderColor }}
            >
              <div className={`${card.color} p-6 rounded-xl mb-4 flex justify-center`}>
                <div className="text-4xl">{card.emoji}</div>
              </div>
              <div className="text-md font-semibold mb-1">{card.title}</div>
              <div className="text-sm text-gray-600 mb-3">
                {card.description}
              </div>
              <div className="flex justify-center items-center gap-3 text-sm mb-4">
                <div>⏱️ {card.duration}</div>
                <span className={`${card.color} ${card.textColor} px-3 py-1 rounded-full font-medium`}>
                  {card.level}
                </span>
              </div>
              <button
                onClick={() => navigate("/resume", { state: { difficulty: card.level } })}
                className="bg-black text-white py-2 px-6 rounded-lg hover:bg-gray-800 transition-colors"
              >
                🚀 Start Interview
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Home;
