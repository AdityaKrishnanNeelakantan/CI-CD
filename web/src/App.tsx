import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertCircle, CheckCircle2, Download, FileUp, Loader2, Play } from "lucide-react";
import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  GenerateConfig,
  Job,
  LlmTextColumn,
  PreviewResponse,
  SchemaColumn,
  SchemaSession,
  SchemaSummary,
  UploadSchemaResponse,
  ValidationResponse,
  createDownload,
  createSchemaSession,
  fetchDownload,
  getJob,
  getPreview,
  getSettings,
  getValidation,
  startSchemaGeneration,
  uploadSchemaFile
} from "./api";

type StepKey = "input" | "configure" | "generate" | "results";

const localeOptions = ["en_US", "en_GB", "en_IN", "de_DE", "fr_FR"];

function App() {
  const [intent, setIntent] = useState("Development & testing");
  const [session, setSession] = useState<SchemaSession | null>(null);
  const [summary, setSummary] = useState<SchemaSummary | null>(null);
  const [columns, setColumns] = useState<SchemaColumn[]>([]);
  const [llmColumns, setLlmColumns] = useState<LlmTextColumn[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [selectedTable, setSelectedTable] = useState("");
  const [downloadBlocked, setDownloadBlocked] = useState("");
  const [downloadReady, setDownloadReady] = useState(false);
  const [error, setError] = useState("");
  const [config, setConfig] = useState<GenerateConfig>({
    row_count: 100,
    locale: "en_US",
    seed: 42,
    export_format: "csv",
    preview_rows: 100,
    llm_text_enabled: false
  });

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings
  });

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
      setJob(null);
      setPreview(null);
      setValidation(null);
      setSelectedTable("");
      setDownloadBlocked("");
      setDownloadReady(false);
    },
    onError: (err) => setError(messageForError(err))
  });

  const generateMutation = useMutation({
    mutationFn: ({ sessionId, nextConfig }: { sessionId: string; nextConfig: GenerateConfig }) =>
      startSchemaGeneration(sessionId, nextConfig),
    onSuccess: (newJob) => {
      setJob(newJob);
      setPreview(null);
      setValidation(null);
      setDownloadBlocked("");
      setDownloadReady(false);
    },
    onError: (err) => setError(messageForError(err))
  });

  const downloadMutation = useMutation({
    mutationFn: async (sessionId: string) => {
      const ticket = await createDownload(sessionId);
      return fetchDownload(ticket.url);
    },
    onSuccess: (blob) => {
      setDownloadReady(true);
      setDownloadBlocked("");
      saveBlob(blob, `${summary?.name ?? "schema_twin"}_synthetic.zip`);
    },
    onError: (err) => {
      const message = messageForError(err);
      setDownloadReady(false);
      setDownloadBlocked(message);
    }
  });

  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed") return;
    const timer = window.setInterval(async () => {
      try {
        const latest = await getJob(job.id);
        setJob(latest);
        if (latest.status === "failed") {
          setError(messageForJobFailure(latest));
        }
      } catch (err) {
        setError(messageForError(err));
      }
    }, 800);
    return () => window.clearInterval(timer);
  }, [job]);

  useEffect(() => {
    if (!session || job?.status !== "succeeded" || preview || validation) return;
    let alive = true;
    Promise.all([getPreview(session.id), getValidation(session.id)])
      .then(([nextPreview, nextValidation]) => {
        if (!alive) return;
        setPreview(nextPreview);
        setValidation(nextValidation);
        setSelectedTable(Object.keys(nextPreview.tables)[0] ?? "");
      })
      .catch((err) => setError(messageForError(err)));
    return () => {
      alive = false;
    };
  }, [job, preview, session, validation]);

  const activeStep = useMemo<StepKey>(() => {
    if (preview && validation) return "results";
    if (job) return "generate";
    if (summary) return "configure";
    return "input";
  }, [job, preview, summary, validation]);

  const tableRows = selectedTable ? preview?.tables[selectedTable] ?? [] : [];

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
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
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Synth Platform</p>
          <h1>Schema Twin</h1>
        </div>
        <div className="service-pill">{settingsQuery.isSuccess ? "API connected" : "API pending"}</div>
      </header>

      <nav className="steps" aria-label="Schema Twin progress">
        <Step label="Input / Review" active={activeStep === "input"} done={Boolean(summary)} />
        <Step label="Configure" active={activeStep === "configure"} done={Boolean(job)} />
        <Step label="Generate" active={activeStep === "generate"} done={Boolean(preview)} />
        <Step label="Results" active={activeStep === "results"} done={Boolean(downloadReady)} />
      </nav>

      {error ? (
        <div className="alert error" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      ) : null}

      <section className="band two-column">
        <div className="panel">
          <h2>Input / Review</h2>
          <label>
            Intent
            <select value={intent} onChange={(event) => setIntent(event.target.value)}>
              <option>Development & testing</option>
              <option>QA / automated tests</option>
              <option>Demonstrations</option>
              <option>Data pipeline development</option>
              <option>Early project environments</option>
            </select>
          </label>
          <label className="file-drop">
            <FileUp size={22} />
            <span>{uploadMutation.isPending ? "Uploading..." : "Choose schema file"}</span>
            <input aria-label="Schema file" type="file" accept=".json,.yaml,.yml,.sql" onChange={handleFileChange} />
          </label>
          {summary ? <Summary summary={summary} /> : null}
        </div>

        <div className="panel">
          <h2>Configure</h2>
          <div className="config-grid">
            <label>
              Rows per table
              <input
                min={1}
                type="number"
                value={config.row_count}
                onChange={(event) => setConfig({ ...config, row_count: Number(event.target.value) })}
              />
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
              <input
                min={0}
                type="number"
                value={config.seed}
                onChange={(event) => setConfig({ ...config, seed: Number(event.target.value) })}
              />
            </label>
            <label>
              Export format
              <select
                value={config.export_format}
                onChange={(event) => setConfig({ ...config, export_format: event.target.value as "csv" | "parquet" })}
              >
                <option value="csv">CSV</option>
                <option value="parquet">Parquet</option>
              </select>
            </label>
          </div>
          {llmColumns.length ? (
            <label className="toggle">
              <input
                type="checkbox"
                checked={config.llm_text_enabled}
                onChange={(event) => setConfig({ ...config, llm_text_enabled: event.target.checked })}
              />
              <span>Use LLM for text-heavy columns</span>
            </label>
          ) : null}
          <button className="primary" disabled={!summary || generateMutation.isPending} onClick={handleGenerate}>
            {generateMutation.isPending ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Generate
          </button>
        </div>
      </section>

      <section className="band">
        <div className="panel wide">
          <h2>Generate</h2>
          <JobStatusPanel job={job} />
        </div>
      </section>

      <section className="band two-column results-band">
        <div className="panel">
          <h2>Preview</h2>
          {preview ? (
            <>
              <select
                aria-label="Preview table"
                value={selectedTable}
                onChange={(event) => setSelectedTable(event.target.value)}
              >
                {Object.keys(preview.tables).map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
              <DataTable rows={tableRows} />
            </>
          ) : (
            <EmptyState text="Waiting for generated tables" />
          )}
        </div>

        <div className="panel">
          <h2>Validation / Download</h2>
          {validation ? <ValidationPanel validation={validation} /> : <EmptyState text="Waiting for validation" />}
          <button
            className="primary"
            disabled={!validation || downloadMutation.isPending || Boolean(downloadBlocked)}
            onClick={() => session && downloadMutation.mutate(session.id)}
          >
            {downloadMutation.isPending ? <Loader2 className="spin" size={18} /> : <Download size={18} />}
            Download ZIP
          </button>
          {downloadReady ? (
            <div className="alert success" role="status">
              <CheckCircle2 size={18} />
              <span>Download approved</span>
            </div>
          ) : null}
          {downloadBlocked ? (
            <div className="alert error" role="alert">
              <AlertCircle size={18} />
              <span>{downloadBlocked}</span>
            </div>
          ) : null}
        </div>
      </section>

      {columns.length ? (
        <section className="band">
          <div className="panel wide">
            <h2>Columns</h2>
            <DataTable rows={columns} />
          </div>
        </section>
      ) : null}
    </main>
  );
}

function Step({ label, active, done }: { label: string; active: boolean; done: boolean }) {
  return (
    <div className={`step ${active ? "active" : ""} ${done ? "done" : ""}`}>
      {done ? <CheckCircle2 size={17} /> : <span className="dot" />}
      <span>{label}</span>
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

function JobStatusPanel({ job }: { job: Job | null }) {
  if (!job) return <EmptyState text="No generation job started" />;
  return (
    <div>
      <div className={`job-status ${job.status}`}>
        {job.status === "running" || job.status === "queued" ? <Loader2 className="spin" size={18} /> : null}
        <strong>{job.status}</strong>
      </div>
      <ol className="progress-list">
        {job.progress.map((message, index) => (
          <li key={`${message}-${index}`}>{message}</li>
        ))}
      </ol>
      {job.error ? (
        <div className="alert error" role="alert">
          <AlertCircle size={18} />
          <span>{messageForJobFailure(job)}</span>
        </div>
      ) : null}
    </div>
  );
}

function ValidationPanel({ validation }: { validation: ValidationResponse }) {
  const { highlights } = validation;
  return (
    <div className="validation">
      <div className={`validation-badge ${highlights.hard_checks_passed ? "passed" : "failed"}`}>
        {highlights.hard_checks_passed ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
        <span>{highlights.status}</span>
      </div>
      <div className="summary-grid compact">
        <Metric label="Export files" value={highlights.export_count} />
        <Metric label="Tables" value={highlights.tables.length} />
      </div>
      <pre>{JSON.stringify(validation.report, null, 2)}</pre>
    </div>
  );
}

function DataTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  if (!rows.length) return <EmptyState text="No rows available" />;
  const headers = Object.keys(rows[0]);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 12).map((row, index) => (
            <tr key={index}>
              {headers.map((header) => (
                <td key={header}>{String(row[header] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty">{text}</div>;
}

function messageForError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === "llm_policy_error") return error.message;
    if (error.code === "transfer_blocked") return error.message;
    if (error.code === "schema_parse_error") return `Schema upload failed: ${error.message}`;
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return "Request failed";
}

function messageForJobFailure(job: Job): string {
  if (!job.error) return "Generation failed";
  if (job.error.code === "llm_policy_error") return job.error.message;
  return `Generation failed: ${job.error.message}`;
}

function saveBlob(blob: Blob, filename: string) {
  if (!("createObjectURL" in URL)) return;
  const href = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = href;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(href);
}

export default App;
