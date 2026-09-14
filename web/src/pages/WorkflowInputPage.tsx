import { useMutation } from "@tanstack/react-query";
import { AlertCircle, Loader2, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  ApiError,
  WorkflowSession,
  configureDocumentSession,
  configureInteractionSession,
  createDocumentSession,
  createInteractionSession,
  startDocumentGeneration,
  startInteractionGeneration,
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
  const location = useLocation();
  const WorkflowIcon = workflow.Icon;
  const routedContext = useMemo(() => routedContextFromSearch(location.search), [location.search]);
  const [intent, setIntent] = useState("Development & testing");
  const [session, setSession] = useState<WorkflowSession | null>(null);
  const [seed, setSeed] = useState(42);
  const [extractionMethod, setExtractionMethod] = useState("auto");
  const [llmTextEnabled, setLlmTextEnabled] = useState(false);
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

  useEffect(() => {
    if (!routedContext.prompt) return;
    setIntent(routedContext.prompt);
    const prefill = routedContext.intent?.prefill;
    if (typeof prefill?.interaction_type === "string") setInteractionType(prefill.interaction_type);
    if (typeof prefill?.output_format === "string") setOutputFormat(prefill.output_format);
    if (typeof prefill?.other?.seed === "number") setSeed(prefill.other.seed);
    if (typeof prefill?.other?.extraction_method === "string") setExtractionMethod(prefill.other.extraction_method);
    if (typeof prefill?.other?.llm_text_enabled === "boolean") setLlmTextEnabled(prefill.other.llm_text_enabled);
    if (typeof prefill?.other?.remove_sensitive_information === "boolean") setRemoveSensitive(prefill.other.remove_sensitive_information);
  }, [routedContext]);

  const generateMutation = useMutation({
    mutationFn: async () => {
      if (!uploaded) throw new Error(`${workflow.title} requires a supported upload before generation.`);
      if (workflow.key === "interaction" && (!interactionType || !outputFormat)) {
        throw new Error("Choose an interaction type and output format before generation.");
      }
      const currentSession = session ?? (await createSession(workflow.key, intent));
      setSession(currentSession);
      if (workflow.key === "document") {
        await configureDocumentSession(currentSession.id, {
          seed,
          extraction_method: extractionMethod === "auto" ? undefined : extractionMethod,
          llm_text_enabled: llmTextEnabled
        });
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

  const uploadLabel = workflow.key === "document" ? "PDF document" : "Transcript file";
  const document = session?.state.document as Record<string, unknown> | undefined;
  const transcript = session?.state.transcript as Record<string, unknown> | undefined;

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
      {routedContext.prompt || routedContext.filename ? (
        <div className="route-banner">
          <strong>Started from chat request: {routedContext.prompt ? `"${routedContext.prompt}"` : "attached file"}</strong>
          <span>Detected workflow: {workflow.title}</span>
          {routedContext.filename ? <span>Attached file: {routedContext.filename}. Upload it here to continue with generation.</span> : null}
        </div>
      ) : null}
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
          <UploadPanel accept={acceptForWorkflow(workflow.key)} busy={uploadMutation.isPending} label={uploadLabel} onFile={(file) => uploadMutation.mutate(file)} />
          {document ? (
            <div className="summary-grid">
              <Metric label="Document" value={String(document.filename ?? "PDF")} />
              <Metric label="Size" value={formatBytes(Number(document.size ?? 0))} />
            </div>
          ) : null}
          {transcript ? (
            <div className="summary-grid">
              <Metric label="Transcript" value={String(transcript.filename ?? "TXT/LOG")} />
              <Metric label="Turns" value={Number(transcript.turn_count ?? 0)} />
            </div>
          ) : null}
          {session ? <div className="empty">Session ready: {session.id}</div> : null}
        </div>
        <div className="panel">
          <h2>Configure</h2>
          <div className="config-grid">
            <label>
              Random seed
              <input min={0} type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </label>
          </div>
          {workflow.key === "document" ? (
            <>
              <label>
                Extraction method
                <select value={extractionMethod} onChange={(event) => setExtractionMethod(event.target.value)}>
                  <option value="auto">Auto</option>
                  <option value="native">Native PDF</option>
                  <option value="ocr">OCR</option>
                  <option value="docling">Docling</option>
                </select>
              </label>
              <label className="toggle">
                <input checked={llmTextEnabled} type="checkbox" onChange={(event) => setLlmTextEnabled(event.target.checked)} />
                <span>Use LLM for synthetic text fields</span>
              </label>
              <div className="empty">Document type, redaction level, preserve layout, and PDF output controls depend on backend support and remain automatic in this phase.</div>
            </>
          ) : null}
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
              <div className="empty">Conversation and message counts are controlled automatically.</div>
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

function routedContextFromSearch(search: string) {
  const params = new URLSearchParams(search);
  return {
    prompt: params.get("prompt") ?? "",
    filename: params.get("filename") ?? "",
    intent: parseIntent(params.get("intent"))
  };
}

function parseIntent(raw: string | null): {
  prefill?: {
    record_count?: number | null;
    output_format?: string | null;
    interaction_type?: string | null;
    other?: Record<string, unknown>;
  };
} | null {
  if (!raw) return null;
  try {
    return JSON.parse(decodeURIComponent(raw));
  } catch {
    return null;
  }
}

async function createSession(key: WorkflowConfig["key"], intent: string): Promise<WorkflowSession> {
  if (key === "document") return createDocumentSession(intent);
  return createInteractionSession(intent);
}

function acceptForWorkflow(key: WorkflowConfig["key"]) {
  if (key === "document") return ".pdf";
  return ".txt,.log";
}

function formatBytes(size: number) {
  if (!Number.isFinite(size) || size <= 0) return "-";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.round(size / 102.4) / 10} KB`;
  return `${Math.round(size / 1024 / 102.4) / 10} MB`;
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function validateUpload(key: WorkflowConfig["key"], file: File) {
  const name = file.name.toLowerCase();
  if (key === "database" && !name.endsWith(".sqlite") && !name.endsWith(".sqlite3") && !name.endsWith(".db")) {
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
