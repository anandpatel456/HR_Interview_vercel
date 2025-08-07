import React from "react";
import { Routes, Route } from "react-router-dom";
import Home from "./Home";
import ResumeUpload from "./ResumeUpload";
import GuidelineInterview from "./GuidelineInterview";
import InterviewPage from "./InterviewPage";
import FeedbackPage from "./FeedbackPage";

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/resume" element={<ResumeUpload />} />
      <Route path="/GuidelineInterview" element={<GuidelineInterview />} />
      <Route path="/interview" element={<InterviewPage />} />
      <Route path="/feedback" element={<FeedbackPage />} />
    </Routes>
  );
};

export default App;
