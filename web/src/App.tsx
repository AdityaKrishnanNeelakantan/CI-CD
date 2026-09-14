import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { workflows } from "./lib/workflows";
import { HomePage } from "./pages/HomePage";
import { PlaceholderPage } from "./pages/PlaceholderPage";
import { ProgressPage } from "./pages/ProgressPage";
import { ProjectDetailPage } from "./pages/ProjectDetailPage";
import { ProjectsPage } from "./pages/ProjectsPage";
import { ResultsPage } from "./pages/ResultsPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { SchemaTwinPage } from "./pages/SchemaTwinPage";
import { SettingsPage } from "./pages/SettingsPage";
import { TemplateDetailPage } from "./pages/TemplateDetailPage";
import { TemplatesPage } from "./pages/TemplatesPage";
import { WorkflowInputPage } from "./pages/WorkflowInputPage";

const databaseWorkflow = workflows.find((workflow) => workflow.key === "database")!;
const documentWorkflow = workflows.find((workflow) => workflow.key === "document")!;
const interactionWorkflow = workflows.find((workflow) => workflow.key === "interaction")!;

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/schema" element={<SchemaTwinPage />} />
          <Route path="/database" element={<WorkflowInputPage workflow={databaseWorkflow} />} />
          <Route path="/document" element={<WorkflowInputPage workflow={documentWorkflow} />} />
          <Route path="/interaction" element={<WorkflowInputPage workflow={interactionWorkflow} />} />
          <Route path="/progress/:jobId" element={<ProgressPage />} />
          <Route path="/results/:resultId" element={<ResultsPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:projectId" element={<ProjectDetailPage />} />
          <Route path="/projects/:projectId/runs/:runId" element={<RunDetailPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/help" element={<PlaceholderPage title="Help & Guides" />} />
          <Route path="/templates" element={<TemplatesPage />} />
          <Route path="/templates/:templateId" element={<TemplateDetailPage />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;
