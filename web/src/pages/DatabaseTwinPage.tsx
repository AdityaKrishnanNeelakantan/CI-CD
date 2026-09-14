import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertCircle, Database, FileArchive, Loader2, Play, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  ApiError,
  WorkflowSession,
  configureDatabaseSession,
  createDatabaseSampleSource,
  createDatabaseSession,
  getSettings,
  startDatabaseGeneration,
  uploadDatabaseSource
} from "../api";
import { UploadPanel } from "../components/UploadPanel";
import { WorkflowStepper } from "../components/WorkflowStepper";
import { workflows } from "../lib/workflows";

type CountMode = "preserve" | "default" | "target" | "scale";

type SourceTable = {
  name: string;
  rowCount: number;
  columns: string[];
  columnCount: number;
};

type SourceSummary = {
  filename: string;
  sourceType: string;
  tables: SourceTable[];
  totalRows: number;
  warnings: string[];
};

const databaseWorkflow = workflows.find((workflow) => workflow.key === "database")!;

export function DatabaseTwinPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const routedContext = useMemo(() => routedContextFromSearch(location.search), [location.search]);
  const WorkflowIcon = databaseWorkflow.Icon;
  const settingsQuery = useQuery({ queryKey: ["settings", "database-twin"], queryFn: getSettings });

  const [intent, setIntent] = useState("Development & testing");
  const [session, setSession] = useState<WorkflowSession | null>(null);
  const [sampleCustomerCount, setSampleCustomerCount] = useState(400);
  const [sampleLimit, setSampleLimit] = useState(5000);
  const [seed, setSeed] = useState(42);
  const [countMode, setCountMode] = useState<CountMode>("preserve");
  const [targetRecordCount, setTargetRecordCount] = useState(100);
  const [scaleFactor, setScaleFactor] = useState(1);
  const [error, setError] = useState("");

  const source = sourceFromSession(session);
  const hasSource = Boolean(source?.tables.length);
  const defaultRecordCount = settingsQuery.data?.default_record_count ?? 100;
  const privacyLevel = settingsQuery.data?.privacy_level ?? "standard";

  useEffect(() => {
    const prefill = routedContext.intent?.prefill;
    if (routedContext.prompt) setIntent(routedContext.prompt);
    if (typeof prefill?.record_count === "number") {
      setTargetRecordCount(prefill.record_count);
      setCountMode("target");
    } else if (typeof settingsQuery.data?.default_record_count === "number") {
      setTargetRecordCount(settingsQuery.data.default_record_count);
    }
    if (typeof prefill?.other?.sample_limit === "number") setSampleLimit(prefill.other.sample_limit);
    if (typeof prefill?.other?.seed === "number") setSeed(prefill.other.seed);
  }, [routedContext, settingsQuery.data?.default_record_count]);

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      assertSQLite(file);
      const currentSession = session ?? (await createDatabaseSession(intent));
      setSession(currentSession);
      return uploadDatabaseSource(currentSession.id, file, "sqlite");
    },
    onSuccess: (nextSession) => {
      setSession(nextSession);
      setCountMode("preserve");
      setError("");
    },
    onError: (err) => setError(messageForError(err))
  });

  const sampleMutation = useMutation({
    mutationFn: async () => {
      const currentSession = session ?? (await createDatabaseSession(intent));
      setSession(currentSession);
      return createDatabaseSampleSource(currentSession.id, { customer_count: sampleCustomerCount, seed });
    },
    onSuccess: (nextSession) => {
      setSession(nextSession);
      setCountMode("preserve");
      setError("");
    },
    onError: (err) => setError(messageForError(err))
  });

  const generateMutation = useMutation({
    mutationFn: async () => {
      if (!session || !hasSource) throw new Error("Upload and discover a SQLite database before generation.");
      await configureDatabaseSession(session.id, buildConfigurePayload(countMode, targetRecordCount, scaleFactor, sampleLimit, seed));
      return startDatabaseGeneration(session.id);
    },
    onSuccess: (job) => navigate(`/progress/${job.id}`),
    onError: (err) => setError(messageForError(err))
  });

  return (
    <div className="page-stack">
      <section className="workflow-page-heading">
        <span className={`workflow-icon workflow-${databaseWorkflow.accent}`}>
          <WorkflowIcon size={24} />
        </span>
        <div>
          <h2>{databaseWorkflow.title}</h2>
          <p>{databaseWorkflow.description}</p>
        </div>
      </section>
      <WorkflowStepper activeIndex={hasSource ? 1 : 0} steps={databaseWorkflow.steps} />
      {routedContext.prompt || routedContext.filename ? (
        <div className="route-banner">
          <strong>Started from chat request: {routedContext.prompt ? `"${routedContext.prompt}"` : "attached file"}</strong>
          <span>Detected workflow: Database Twin</span>
          {routedContext.filename ? <span>Attached file: {routedContext.filename}. Upload it here to continue.</span> : null}
        </div>
      ) : null}
      {error ? <Alert message={error} /> : null}

      <section className="two-column">
        <div className="panel">
          <h2>Source</h2>
          <label>
            Intent
            <select value={intent} onChange={(event) => setIntent(event.target.value)}>
              <option>Development & testing</option>
              <option>QA / automated tests</option>
              <option>Demonstrations</option>
              <option>Analytics prototyping</option>
              <option>Safe sharing with partners</option>
            </select>
          </label>
          <div className="sample-source">
            <label>
              Sample customers
              <input
                max={2000}
                min={50}
                step={50}
                type="number"
                value={sampleCustomerCount}
                onChange={(event) => setSampleCustomerCount(Number(event.target.value))}
              />
            </label>
            <button disabled={sampleMutation.isPending || uploadMutation.isPending} onClick={() => sampleMutation.mutate()} type="button">
              {sampleMutation.isPending ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              Generate sample database
            </button>
          </div>
          <UploadPanel accept=".db,.sqlite,.sqlite3" busy={uploadMutation.isPending} label="SQLite database" onFile={(file) => uploadMutation.mutate(file)} />
          {!source ? <div className="empty">Upload a SQLite database to discover real tables, columns, and source row counts.</div> : null}
          {source ? <SourceSummaryPanel source={source} /> : null}
          {session ? <div className="empty">Session ready: {session.id}</div> : null}
        </div>

        <div className="panel">
          <h2>Generate</h2>
          <div className="config-grid">
            <label>
              Sample limit
              <input min={1} type="number" value={sampleLimit} onChange={(event) => setSampleLimit(Number(event.target.value))} />
            </label>
            <label>
              Random seed
              <input min={0} type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </label>
          </div>
          <div className="segmented-control" role="group" aria-label="Database row count mode">
            <button className={countMode === "preserve" ? "active" : ""} disabled={!hasSource} onClick={() => setCountMode("preserve")} type="button">
              Preserve
            </button>
            <button className={countMode === "default" ? "active" : ""} onClick={() => setCountMode("default")} type="button">
              Default
            </button>
            <button className={countMode === "target" ? "active" : ""} onClick={() => setCountMode("target")} type="button">
              Target
            </button>
            <button className={countMode === "scale" ? "active" : ""} disabled={!hasSource} onClick={() => setCountMode("scale")} type="button">
              Scale
            </button>
          </div>
          {countMode === "target" ? (
            <label>
              Target records per table
              <input min={1} type="number" value={targetRecordCount} onChange={(event) => setTargetRecordCount(Number(event.target.value))} />
            </label>
          ) : null}
          {countMode === "scale" ? (
            <label>
              Scale factor
              <input min={0.01} step={0.1} type="number" value={scaleFactor} onChange={(event) => setScaleFactor(Number(event.target.value))} />
            </label>
          ) : null}
          <div className="capability-list">
            <Capability Icon={Database} label="Tables" value={source ? source.tables.map((table) => table.name).join(", ") : "discovered after upload"} />
            <Capability Icon={Database} label="Counts" value={countDescription(countMode, defaultRecordCount, scaleFactor)} />
            <Capability Icon={FileArchive} label="Output" value="SQLite database plus CSV table exports" />
            <Capability Icon={ShieldCheck} label="Privacy" value={`${privacyLevel} product setting with Database Twin safeguards`} />
          </div>
          {source && countMode === "preserve" ? (
            <div className="table-count-grid">
              {source.tables.map((table) => (
                <Metric key={table.name} label={table.name} value={formatNumber(table.rowCount)} />
              ))}
            </div>
          ) : null}
          <button className="primary" disabled={!hasSource || generateMutation.isPending} onClick={() => generateMutation.mutate()}>
            {generateMutation.isPending ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Generate Database Twin
          </button>
        </div>
      </section>
    </div>
  );
}

function buildConfigurePayload(mode: CountMode, targetRecordCount: number, scaleFactor: number, sampleLimit: number, seed: number) {
  return {
    preserve_source_counts: mode === "preserve" ? true : undefined,
    target_record_count: mode === "target" ? targetRecordCount : undefined,
    scale_factor: mode === "scale" ? scaleFactor : undefined,
    sample_limit: sampleLimit,
    seed
  };
}

function SourceSummaryPanel({ source }: { source: SourceSummary }) {
  return (
    <div className="source-summary">
      <div className="summary-grid">
        <Metric label="File" value={source.filename} />
        <Metric label="Source" value={source.sourceType} />
        <Metric label="Tables" value={source.tables.length} />
        <Metric label="Source rows" value={formatNumber(source.totalRows)} />
      </div>
      <div className="table-summary" aria-label="Discovered source tables">
        {source.tables.map((table) => (
          <div className="table-summary-row" key={table.name}>
            <strong>{table.name}</strong>
            <span>{formatNumber(table.rowCount)} rows</span>
            <span>{table.columnCount} columns</span>
          </div>
        ))}
      </div>
      {source.warnings.length ? <Alert message={source.warnings.join(" ")} warning /> : null}
    </div>
  );
}

function sourceFromSession(session: WorkflowSession | null): SourceSummary | null {
  const source = session?.state.source as Record<string, unknown> | undefined;
  if (!source) return null;
  const tables = parseTables(source.tables);
  return {
    filename: String(source.filename ?? "SQLite database"),
    sourceType: String(source.source_type ?? "sqlite"),
    tables,
    totalRows: Number(source.total_rows ?? tables.reduce((sum, table) => sum + table.rowCount, 0)),
    warnings: Array.isArray(source.warnings) ? source.warnings.map(String) : []
  };
}

function parseTables(value: unknown): SourceTable[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((table) => {
      if (typeof table === "string") return { name: table, rowCount: 0, columns: [], columnCount: 0 };
      const record = table as Record<string, unknown>;
      const columns = Array.isArray(record.columns) ? record.columns.map(String) : [];
      return {
        name: String(record.name ?? ""),
        rowCount: Number(record.row_count ?? 0),
        columns,
        columnCount: Number(record.column_count ?? columns.length)
      };
    })
    .filter((table) => table.name);
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

function assertSQLite(file: File) {
  const name = file.name.toLowerCase();
  if (!name.endsWith(".db") && !name.endsWith(".sqlite") && !name.endsWith(".sqlite3")) {
    throw new Error("Database Twin currently supports SQLite uploads only.");
  }
}

function countDescription(mode: CountMode, defaultRecordCount: number, scaleFactor: number) {
  if (mode === "preserve") return "source row counts per table";
  if (mode === "target") return "target records per generated table";
  if (mode === "scale") return `${scaleFactor}x source row counts`;
  return `${defaultRecordCount} rows per table from product settings`;
}

function formatNumber(value: number) {
  if (!Number.isFinite(value)) return "0";
  return new Intl.NumberFormat().format(value);
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Capability({ Icon, label, value }: { Icon: typeof Database; label: string; value: string }) {
  return (
    <div className="capability-item">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Alert({ message, warning = false }: { message: string; warning?: boolean }) {
  return (
    <div className={`alert ${warning ? "warning" : "error"}`} role="alert">
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
