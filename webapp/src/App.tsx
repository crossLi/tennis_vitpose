import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingPage from "@/pages/LandingPage";
import ReportPage from "@/pages/ReportPage";
import TaskPage from "@/pages/TaskPage";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/task/:taskId" element={<TaskPage />} />
        <Route path="/report/:taskId" element={<ReportPage />} />
      </Routes>
    </Router>
  );
}
