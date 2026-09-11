import { useMutation } from "@tanstack/react-query";
import { AlertCircle, ArrowRight, Loader2, Paperclip, Send } from "lucide-react";
import { ChangeEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError, WorkflowIntentResponse, classifyWorkflowIntent } from "../api";
import { workflows } from "../lib/workflows";

const acceptedPromptFiles = ".sql,.json,.csv,.xls,.xlsx,.xlsv,.sqlite,.db,.parquet,.pdf,.txt,.log";

export function HomePage() {
  const navigate = useNavigate();
  const [message, setMessage] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");

  const intentMutation = useMutation({
    mutationFn: async () =>
      classifyWorkflowIntent({
        message,
        attachments: file ? [await attachmentMetadata(file)] : [],
        allow_auto_start: false
      }),
    onSuccess: (decision) => {
      setError("");
      navigateToDecision(decision);
    },
    onError: (err) => setError(messageForError(err))
  });

  function submitIntent() {
    if (!message.trim() && !file) {
      setError("Enter a request or attach a supported file.");
      return;
    }
    intentMutation.mutate();
  }

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
  }

  function navigateToDecision(decision: WorkflowIntentResponse) {
    if (decision.workflow_type === "unknown" || !decision.suggested_route || decision.suggested_route === "/") {
      setError("I could not match that request to a supported workflow. Try adding a supported file or a more specific request.");
      return;
    }
    const params = new URLSearchParams({ source: "chat" });
    if (message.trim()) params.set("prompt", message.trim());
    if (file) params.set("filename", file.name);
    params.set("intent", encodeURIComponent(JSON.stringify(decision)));
    navigate(`${decision.suggested_route}?${params.toString()}`);
  }

  return (
    <div className="page-stack home-page">
      <section className="home-hero">
        <div className="home-hero-copy">
          <p className="eyebrow">Home / Chat</p>
          <h2>Choose a workflow or describe the synthetic twin you need.</h2>
        </div>
      </section>
      <section className="workflow-grid" aria-label="Workflow cards">
        {workflows.map(({ Icon, accent, description, formats, route, title }) => (
          <Link className={`workflow-card workflow-${accent}`} key={route} to={route}>
            <span className="workflow-icon">
              <Icon size={24} />
            </span>
            <div>
              <h3>{title}</h3>
              <p>{description}</p>
              <span className="workflow-formats">{formats}</span>
            </div>
            <ArrowRight size={18} />
          </Link>
        ))}
      </section>
      <div className="chat-entry home-chat-entry" aria-label="Chat prompt">
        <div>
          <label className="icon-button attachment-button" title="Attach file">
            <Paperclip size={18} />
            <input
              aria-label="Attach SQL, JSON, CSV, Excel, SQLite, Parquet, PDF, TXT, or LOG files"
              accept={acceptedPromptFiles}
              onChange={handleFile}
              type="file"
            />
          </label>
          <input
            aria-label="Workflow request"
            onChange={(event) => {
              setMessage(event.target.value);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") submitIntent();
            }}
            placeholder="What synthetic twin would you like to create?"
            value={message}
          />
          <button className="icon-button" disabled={intentMutation.isPending} onClick={submitIntent} title="Send" type="button">
            {intentMutation.isPending ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          </button>
        </div>
        {file ? <span className="attached-file">Attached: {file.name}</span> : null}
      </div>
      {error ? (
        <div className="alert error" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      ) : null}
    </div>
  );
}

async function attachmentMetadata(file: File) {
  return {
    filename: file.name,
    content_type: file.type || undefined,
    size_bytes: file.size,
    extension: extensionFor(file.name),
    content_preview: await redactedPreview(file)
  };
}

function extensionFor(filename: string) {
  const index = filename.lastIndexOf(".");
  return index >= 0 ? filename.slice(index).toLowerCase() : "";
}

async function redactedPreview(file: File) {
  if (file.size > 1024 * 1024 || !looksTextLike(file)) return undefined;
  if (typeof file.text !== "function") return undefined;
  const text = (await file.text()).slice(0, 4096);
  return text
    .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, "[email]")
    .replace(/\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b/g, "[id]")
    .replace(/\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b/g, "[phone]")
    .slice(0, 1200);
}

function looksTextLike(file: File) {
  const extension = extensionFor(file.name);
  return [".sql", ".json", ".yaml", ".yml", ".csv", ".txt", ".log"].includes(extension) || file.type.startsWith("text/");
}

function messageForError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Request failed";
}
