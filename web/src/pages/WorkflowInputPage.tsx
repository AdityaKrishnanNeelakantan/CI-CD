import { useMutation } from "@tanstack/react-query";
import { AlertCircle, Loader2, Play } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ApiError,
  WorkflowSession,
  configureDatabaseSession,
  configureDocumentSession,
  configureInteractionSession,
  createDatabaseSession,
  createDocumentSession,
  createInteractionSession,
  startDatabaseGeneration,
  startDocumentGeneration,
  startInteractionGeneration,
  uploadDatabaseSource,
  uploadDocument,
  uploadInteractionTranscript
} from "../api";
import { UploadPanel } from "../components/UploadPanel";
import { WorkflowStepper } from "../components/WorkflowStepper";
import { WorkflowConfig } from "../lib/workflows";

type WorkflowInputPageProps = {
  workflow: WorkflowConfig;
};

export function WorkflowInputPage({ workflow }: WorkflowInputPageProps) {
  const navigate = useNavigate();
  const WorkflowIcon = workflow.Icon;
  const [intent, setIntent] = useState("Development & testing");
  const [session, setSession] = useState<WorkflowSession | null>(null);
  const [connection, setConnection] = useState("");
  const [rowCount, setRowCount] = useState(100);
  const [seed, setSeed] = useState(42);
  const [interactionType, setInteractionType] = useState("support_chat");
  const [outputFormat, setOutputFormat] = useState("structured_json");
  const [removeSensitive, setRemoveSensitive] = useState(true);
  const [uploaded, setUploaded] = useState(false);
  const [error, setError] = useState("");

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      validateUpload(workflow.key, file);
      const currentSession = session ?? (await createSession(workflow.key, intent));
      setSession(currentSession);
      if (workflow.key === "database") return uploadDatabaseSource(currentSession.id, file, sourceTypeForFile(file.name));
      if (workflow.key === "document") return uploadDocument(currentSession.id, file);
      return uploadInteractionTranscript(currentSession.id, file);
    },
    onSuccess: (nextSession) => {
      setSession(nextSession);
      setUploaded(true);
      setError("");
    },
    onError: (err) => setError(messageForError(err))
  });

  const generateMutation = useMutation({
    mutationFn: async () => {
      if (!uploaded) throw new Error(`${workflow.title} requires a supported upload before generation.`);
      if (workflow.key === "interaction" && (!interactionType || !outputFormat)) {
        throw new Error("Choose an interaction type and output format before generation.");
      }
      const currentSession = session ?? (await createSession(workflow.key, intent));
      setSession(currentSession);
      if (workflow.key === "database") {
        await configureDatabaseSession(currentSession.id, { row_count: rowCount, seed });
        return startDatabaseGeneration(currentSession.id);
      }
      if (workflow.key === "document") {
        await configureDocumentSession(currentSession.id, { seed, extraction_method: "auto", llm_text_enabled: false });
        return startDocumentGeneration(currentSession.id);
      }
      await configureInteractionSession(currentSession.id, {
        interaction_type: interactionType,
        output_format: outputFormat,
        remove_sensitive_information: removeSensitive,
        seed
      });
      return startInteractionGeneration(currentSession.id);
    },
    onSuccess: (job) => navigate(`/progress/${job.id}`),
    onError: (err) => setError(messageForError(err))
  });

  const uploadLabel =
    workflow.key === "database" ? "Database or sample file" : workflow.key === "document" ? "PDF document" : "Transcript file";

  return (
    <div className="page-stack">
      <section className="workflow-page-heading">
        <span className={`workflow-icon workflow-${workflow.accent}`}>
          <WorkflowIcon size={24} />
        </span>
        <div>
          <h2>{workflow.title}</h2>
          <p>{workflow.description}</p>
        </div>
      </section>
      <WorkflowStepper activeIndex={session ? 1 : 0} steps={workflow.steps} />
      {error ? (
        <div className="alert error" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      ) : null}
      <section className="two-column">
        <div className="panel">
          <h2>{workflow.steps[0]}</h2>
          <label>
            Intent
            <select value={intent} onChange={(event) => setIntent(event.target.value)}>
              <option>Development & testing</option>
              <option>QA / automated tests</option>
              <option>Demonstrations</option>
              <option>Privacy review</option>
            </select>
          </label>
          {workflow.key === "database" ? (
            <label>
              Connection details
              <textarea disabled placeholder="PostgreSQL connections are coming soon" value={connection} onChange={(event) => setConnection(event.target.value)} />
            </label>
          ) : null}
          <UploadPanel accept={acceptForWorkflow(workflow.key)} busy={uploadMutation.isPending} label={uploadLabel} onFile={(file) => uploadMutation.mutate(file)} />
          {workflow.key === "database" ? <div className="empty">SQLite upload is supported. CSV, Parquet, and PostgreSQL are coming soon.</div> : null}
          {session ? <div className="empty">Session ready: {session.id}</div> : null}
        </div>
        <div className="panel">
          <h2>Configure</h2>
          <div className="config-grid">
            <label>
              Records
              <input min={1} type="number" value={rowCount} onChange={(event) => setRowCount(Number(event.target.value))} />
            </label>
            <label>
              Random seed
              <input min={0} type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </label>
          </div>
          {workflow.key === "interaction" ? (
            <>
              <label>
                Interaction type
                <select value={interactionType} onChange={(event) => setInteractionType(event.target.value)}>
                  <option value="support_chat">Support chat</option>
                  <option value="call_center">Call center</option>
                  <option value="email_thread">Email thread</option>
                </select>
              </label>
              <label>
                Output format
                <select value={outputFormat} onChange={(event) => setOutputFormat(event.target.value)}>
                  <option value="structured_json">Structured JSON</option>
                  <option value="synthetic_logs">Synthetic logs</option>
                </select>
              </label>
              <label className="toggle">
                <input checked={removeSensitive} type="checkbox" onChange={(event) => setRemoveSensitive(event.target.checked)} />
                <span>Remove sensitive information</span>
              </label>
            </>
          ) : null}
          <button className="primary" disabled={!uploaded || generateMutation.isPending} onClick={() => generateMutation.mutate()}>
            {generateMutation.isPending ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Generate
          </button>
        </div>
      </section>
    </div>
  );
}

async function createSession(key: WorkflowConfig["key"], intent: string): Promise<WorkflowSession> {
  if (key === "database") return createDatabaseSession(intent);
  if (key === "document") return createDocumentSession(intent);
  return createInteractionSession(intent);
}

function acceptForWorkflow(key: WorkflowConfig["key"]) {
  if (key === "database") return ".sqlite,.db";
  if (key === "document") return ".pdf";
  return ".txt,.log";
}

function sourceTypeForFile(name: string) {
  return "sqlite";
}

function validateUpload(key: WorkflowConfig["key"], file: File) {
  const name = file.name.toLowerCase();
  if (key === "database" && !name.endsWith(".sqlite") && !name.endsWith(".db")) {
    throw new Error("Database Twin currently supports SQLite uploads only.");
  }
  if (key === "document" && !name.endsWith(".pdf") && file.type !== "application/pdf") {
    throw new Error("Document Twin requires a PDF upload.");
  }
  if (key === "interaction" && !name.endsWith(".txt") && !name.endsWith(".log")) {
    throw new Error("Customer Interaction Twin requires a TXT or LOG transcript.");
  }
}

function messageForError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Request failed";
}
