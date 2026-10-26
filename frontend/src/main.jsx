import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import Dashboard from "./pages/Dashboard";
import ResearchDashboard from "./pages/ResearchDashboard";
import AnalysisView from "./pages/AnalysisView";
import ChatAssistantWidget from "./components/chat/ChatAssistantWidget";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/research" element={<ResearchDashboard />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/analysis/:uploadId?" element={<AnalysisView />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ChatAssistantWidget />
    </BrowserRouter>
  </React.StrictMode>
);
