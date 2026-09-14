import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertCircle, Loader2, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  ApiError,
  GenerateConfig,
  LlmTextColumn,
  SchemaColumn,
  SchemaSession,
  SchemaSummary,
  UploadSchemaResponse,
  createSchemaSession,
  getSettings,
  startSchemaGeneration,
  uploadSchemaFile
} from "../api";
import { DataTable } from "../components/DataTable";
import { UploadPanel } from "../components/UploadPanel";
import { WorkflowStepper } from "../components/WorkflowStepper";
import { workflows } from "../lib/workflows";

const localeOptions = ["en_US", "en_GB", "en_IN", "de_DE", "fr_FR"];
const schemaWorkflow = workflows.find((workflow) => workflow.key === "schema")!;

export function SchemaTwinPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const WorkflowIcon = schemaWorkflow.Icon;
  const [intent, setIntent] = useState("Development & testing");
  const [session, setSession] = useState<SchemaSession | null>(null);
  const [summary, setSummary] = useState<SchemaSummary | null>(null);
  const [columns, setColumns] = useState<SchemaColumn[]>([]);
  const [llmColumns, setLlmColumns] = useState<LlmTextColumn[]>([]);
  const [error, setError] = useState("");
  const [config, setConfig] = useState<GenerateConfig>({
    row_count: 100,
    locale: "en_US",
    seed: 42,
    export_format: "csv",
    preview_rows: 100,
    llm_text_enabled: false
  });

  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: getSettings });

  useEffect(() => {
    const state = location.state as { templateData?: UploadSchemaResponse } | null;
    if (!state?.templateData) return;
    setSession(state.templateData.session);
    setSummary(state.templateData.summary);
    setColumns(state.templateData.columns);
    setLlmColumns(state.templateData.llm_text_columns);
    navigate(".", { replace: true, state: null });
  }, [location.state, navigate]);

  useEffect(() => {
    const settings = settingsQuery.data;
    if (!settings) return;
    setConfig((current) => ({
      ...current,
      row_count: settings.default_record_count,
      export_format: settings.default_output_format === "parquet" ? "parquet" : "csv",
      preview_rows: Math.min(100, settings.default_record_count)
    }));
  }, [settingsQuery.data]);

  const activeStep = useMemo(() => (summary ? 1 : 0), [summary]);

  const createSessionMutation = useMutation({
    mutationFn: createSchemaSession,
    onSuccess: setSession,
    onError: (err) => setError(messageForError(err))
  });

  const uploadMutation = useMutation({
    mutationFn: ({ sessionId, file }: { sessionId: string; file: File }) => uploadSchemaFile(sessionId, file),
    onSuccess: (data: UploadSchemaResponse) => {
      setSummary(data.summary);
      setColumns(data.columns);
      setLlmColumns(data.llm_text_columns);
      setSession(data.session);
    },
    onError: (err) => setError(messageForError(err))
  });

  const generateMutation = useMutation({
    mutationFn: ({ sessionId, nextConfig }: { sessionId: string; nextConfig: GenerateConfig }) =>
      startSchemaGeneration(sessionId, nextConfig),
    onSuccess: (job) => navigate(`/progress/${job.id}`),
    onError: (err) => setError(messageForError(err))
  });

  async function handleFile(file: File) {
    setError("");
    const currentSession = session ?? (await createSessionMutation.mutateAsync(intent));
    uploadMutation.mutate({ sessionId: currentSession.id, file });
  }

  function handleGenerate() {
    if (!session) return;
    setError("");
    generateMutation.mutate({
      sessionId: session.id,
      nextConfig: { ...config, preview_rows: Math.min(100, config.row_count) }
    });
  }

  return (
    <div className="page-stack">
      <section className="workflow-page-heading">
        <span className={`workflow-icon workflow-${schemaWorkflow.accent}`}>
          <WorkflowIcon size={24} />
        </span>
        <div>
          <h2>{schemaWorkflow.title}</h2>
          <p>Create validated synthetic data from a schema or natural language description.</p>
        </div>
      </section>
      <WorkflowStepper activeIndex={activeStep} steps={schemaWorkflow.steps} />
      {error ? <Alert message={error} /> : null}
      <section className="two-column">
        <div className="panel">
          <h2>Input</h2>
          <label>
            Intent
            <select value={intent} onChange={(event) => setIntent(event.target.value)}>
              <option>Development & testing</option>
              <option>QA / automated tests</option>
              <option>Demonstrations</option>
              <option>Data pipeline development</option>
            </select>
          </label>
          <UploadPanel accept=".json,.yaml,.yml,.sql,.csv,.xlsx" busy={uploadMutation.isPending} label="Schema file" onFile={handleFile}>
            <label>
              Natural language schema description
              <textarea disabled placeholder="Coming in Phase 4" />
            </label>
            <button className="secondary" disabled type="button">
              Template generation starts from the Templates page
            </button>
          </UploadPanel>
          {summary ? <Summary summary={summary} /> : null}
        </div>
        <div className="panel">
          <h2>Generate</h2>
          <div className="config-grid">
            <label>
              Rows per table
              <input min={1} type="number" value={config.row_count} onChange={(event) => setConfig({ ...config, row_count: Number(event.target.value) })} />
            </label>
            <label>
              Locale
              <select value={config.locale} onChange={(event) => setConfig({ ...config, locale: event.target.value })}>
                {localeOptions.map((locale) => (
                  <option key={locale}>{locale}</option>
                ))}
              </select>
            </label>
            <label>
              Random seed
              <input min={0} type="number" value={config.seed} onChange={(event) => setConfig({ ...config, seed: Number(event.target.value) })} />
            </label>
            <label>
              Export format
              <select value={config.export_format} onChange={(event) => setConfig({ ...config, export_format: event.target.value as "csv" | "parquet" })}>
                <option value="csv">CSV</option>
                <option value="parquet">Parquet</option>
              </select>
            </label>
          </div>
          {llmColumns.length ? (
            <label className="toggle">
              <input checked={config.llm_text_enabled} type="checkbox" onChange={(event) => setConfig({ ...config, llm_text_enabled: event.target.checked })} />
              <span>Use LLM for text-heavy columns</span>
            </label>
          ) : null}
          <button className="primary" disabled={!summary || generateMutation.isPending} onClick={handleGenerate}>
            {generateMutation.isPending ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Generate
          </button>
        </div>
      </section>
      {columns.length ? (
        <section className="panel wide">
          <h2>Review Columns</h2>
          <DataTable rows={columns} />
        </section>
      ) : null}
    </div>
  );
}

function Summary({ summary }: { summary: SchemaSummary }) {
  return (
    <div className="summary-grid">
      <Metric label="Tables" value={summary.table_count} />
      <Metric label="Columns" value={summary.column_count} />
      <Metric label="Relationships" value={summary.relationship_count} />
      <Metric label="Dataset" value={summary.name} />
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Alert({ message }: { message: string }) {
  return (
    <div className="alert error" role="alert">
      <AlertCircle size={18} />
      <span>{message}</span>
    </div>
  );
}

function messageForError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Request failed";
}
