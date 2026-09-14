import { useMutation } from "@tanstack/react-query";
import { AlertCircle, ArrowRight, Loader2, Paperclip, Send } from "lucide-react";
import { ChangeEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ApiError,
  GenerateConfig,
  WorkflowIntentResponse,
  classifyWorkflowIntent,
  configureDatabaseSession,
  configureDocumentSession,
  configureInteractionSession,
  createDatabaseSampleSource,
  createDatabaseSession,
  createDocumentSession,
  createInteractionSession,
  createSchemaSession,
  startDatabaseGeneration,
  startDocumentGeneration,
  startInteractionGeneration,
  startSchemaGeneration,
  uploadDatabaseSource,
  uploadDocument,
  uploadInteractionTranscript,
  uploadSchemaFile
} from "../api";
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
      startFromDecision(decision);
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

  async function startFromDecision(decision: WorkflowIntentResponse) {
    if (decision.workflow_type === "unknown" || !decision.suggested_route || decision.suggested_route === "/") {
      setError("I could not match that request to a supported workflow. Try adding a supported file or a more specific request.");
      return;
    }
    try {
      const job = file
        ? await startAttachedFileWorkflow(decision, file, message)
        : await startPromptOnlyWorkflow(decision, message);
      navigate(`/progress/${job.id}`);
    } catch (err) {
      setError(messageForError(err));
    }
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

async function startAttachedFileWorkflow(decision: WorkflowIntentResponse, file: File, message: string) {
  const extension = extensionFor(file.name);
  if (isDatabaseFile(extension) && decision.workflow_type !== "database_twin") {
    throw new Error("SQLite database files must use Database Twin. Open Database Twin and upload the .db/.sqlite file there.");
  }
  if (decision.workflow_type === "schema_twin") {
    const session = await createSchemaSession(message || "Chat request");
    await uploadSchemaFile(session.id, file);
    return startSchemaGeneration(session.id, schemaConfig(decision));
  }
  if (decision.workflow_type === "database_twin") {
    const session = await createDatabaseSession(message || "Chat request");
    await uploadDatabaseSource(session.id, file);
    await configureDatabaseSession(session.id, databaseConfig(decision, { preferSourceCounts: true }));
    return startDatabaseGeneration(session.id);
  }
  if (decision.workflow_type === "document_twin") {
    const session = await createDocumentSession(message || "Chat request");
    await uploadDocument(session.id, file);
    await configureDocumentSession(session.id, {
      seed: numberOption(decision, "seed") ?? 42,
      extraction_method: stringOption(decision, "extraction_method"),
      llm_text_enabled: booleanOption(decision, "llm_text_enabled") ?? false
    });
    return startDocumentGeneration(session.id);
  }
  if (decision.workflow_type === "interaction_twin") {
    const session = await createInteractionSession(message || "Chat request");
    await uploadInteractionTranscript(session.id, file);
    await configureInteractionSession(session.id, {
      interaction_type: decision.prefill.interaction_type ?? "support_chat",
      output_format: interactionOutputFormat(decision.prefill.output_format),
      remove_sensitive_information: booleanOption(decision, "remove_sensitive_information") ?? true,
      seed: numberOption(decision, "seed") ?? 42
    });
    return startInteractionGeneration(session.id);
  }
  throw new Error("No supported workflow matched this request.");
}

async function startPromptOnlyWorkflow(decision: WorkflowIntentResponse, message: string) {
  if (decision.workflow_type === "database_twin") {
    const session = await createDatabaseSession(message || "Database demo");
    const rowCount = decision.prefill.record_count ?? 100;
    await createDatabaseSampleSource(session.id, {
      customer_count: sampleCustomerCountFor(rowCount),
      seed: numberOption(decision, "seed") ?? 42
    });
    await configureDatabaseSession(session.id, databaseConfig(decision, { preferSourceCounts: decision.prefill.record_count == null }));
    return startDatabaseGeneration(session.id);
  }
  throw new Error("Attach a supported source file to start this workflow from chat.");
}

function schemaConfig(decision: WorkflowIntentResponse): GenerateConfig {
  const outputFormat = decision.prefill.output_format === "parquet" ? "parquet" : "csv";
  const rowCount = decision.prefill.record_count ?? 100;
  return {
    row_count: rowCount,
    locale: stringOption(decision, "locale") ?? "en_US",
    seed: numberOption(decision, "seed") ?? 42,
    export_format: outputFormat,
    preview_rows: Math.min(100, rowCount),
    llm_text_enabled: booleanOption(decision, "llm_text_enabled") ?? false
  };
}

function databaseConfig(decision: WorkflowIntentResponse, options: { preferSourceCounts: boolean }) {
  const config: {
    preserve_source_counts?: boolean;
    target_record_count?: number;
    sample_limit: number;
    seed: number;
  } = {
    sample_limit: numberOption(decision, "sample_limit") ?? 5000,
    seed: numberOption(decision, "seed") ?? 42
  };
  if (typeof decision.prefill.record_count === "number") {
    config.target_record_count = decision.prefill.record_count;
  } else if (options.preferSourceCounts) {
    config.preserve_source_counts = true;
  }
  return config;
}

function interactionOutputFormat(format: string | null) {
  if (format === "json") return "structured_json";
  if (format === "log" || format === "logs") return "synthetic_logs";
  return "structured_json";
}

function stringOption(decision: WorkflowIntentResponse, key: string): string | undefined {
  const value = decision.prefill.other?.[key];
  return typeof value === "string" ? value : undefined;
}

function numberOption(decision: WorkflowIntentResponse, key: string): number | undefined {
  const value = decision.prefill.other?.[key];
  return typeof value === "number" ? value : undefined;
}

function booleanOption(decision: WorkflowIntentResponse, key: string): boolean | undefined {
  const value = decision.prefill.other?.[key];
  return typeof value === "boolean" ? value : undefined;
}

function sampleCustomerCountFor(rowCount: number) {
  return Math.max(50, Math.min(2000, Math.ceil(rowCount / 50) * 50));
}

function isDatabaseFile(extension: string) {
  return [".db", ".sqlite", ".sqlite3"].includes(extension);
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
