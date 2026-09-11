import { useMutation } from "@tanstack/react-query";
import { AlertCircle, CheckCircle2, RotateCcw, Save } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArtifactFileMetadata, ResultBundle, fetchDownload, saveResult } from "../api";
import { workflowForType } from "../lib/workflows";
import { DataTable } from "./DataTable";
import { DownloadList } from "./DownloadList";
import { QualitySummary } from "./QualitySummary";

type TabKey = "preview" | "files" | "quality" | "summary";

export function ResultsTabs({ result }: { result: ResultBundle }) {
  const [tab, setTab] = useState<TabKey>("preview");
  const [previewTable, setPreviewTable] = useState("");
  const [downloadError, setDownloadError] = useState("");
  const workflow = workflowForType(result.workflow_type);
  const WorkflowIcon = workflow.Icon;
  const preview = useMemo(() => previewToTables(result.preview), [result.preview]);
  const selectedPreview = preview.tables.find((table) => table.name === (previewTable || preview.tables[0]?.name)) ?? preview.tables[0];
  const saveMutation = useMutation({ mutationFn: () => saveResult(result.result_id) });

  async function handleDownload(artifact: ArtifactFileMetadata) {
    if (!artifact.download_url) return;
    setDownloadError("");
    try {
      const blob = await fetchDownload(artifact.download_url);
      saveBlob(blob, artifact.filename ?? artifact.name);
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : "Download failed");
    }
  }

  return (
    <section className="panel wide">
      <div className="results-header">
        <div className="workflow-page-heading inline-heading">
          <span className={`workflow-icon workflow-${workflow.accent}`}>
            <WorkflowIcon size={24} />
          </span>
          <div>
            <p className="eyebrow">{workflow.title}</p>
            <h2>Generated Results</h2>
          </div>
        </div>
        <div className="button-row">
          <Link className="secondary" to={workflow.route}>
            <RotateCcw size={17} />
            Generate Again
          </Link>
          <button className="secondary" disabled={saveMutation.isPending || saveMutation.isSuccess} onClick={() => saveMutation.mutate()}>
            <Save size={17} />
            Save to My Twins
          </button>
        </div>
      </div>
      <div className="tabs" role="tablist">
        {(["preview", "files", "quality", "summary"] as TabKey[]).map((key) => (
          <button className={tab === key ? "active" : ""} key={key} onClick={() => setTab(key)} role="tab">
            {key}
          </button>
        ))}
      </div>
      {tab === "preview" ? (
        <div className="preview-panel">
          {preview.tables.length > 1 ? (
            <div className="table-switcher" aria-label="Preview tables">
              {preview.tables.map((table) => (
                <button
                  aria-label={`${table.name} ${formatNumber(table.rowCount)}`}
                  className={table.name === selectedPreview?.name ? "active" : ""}
                  key={table.name}
                  onClick={() => setPreviewTable(table.name)}
                  type="button"
                >
                  {table.name}
                  <span>{formatNumber(table.rowCount)}</span>
                </button>
              ))}
            </div>
          ) : null}
          <DataTable
            key={selectedPreview?.name ?? "empty-preview"}
            pagination={selectedPreview ? { pageSize: 20, totalRows: selectedPreview.rowCount } : undefined}
            rows={selectedPreview?.rows ?? []}
          />
        </div>
      ) : null}
      {tab === "files" ? <DownloadList artifacts={result.artifacts} onDownload={handleDownload} /> : null}
      {tab === "quality" ? <QualitySummary report={result.quality_report} /> : null}
      {tab === "summary" ? <WorkflowSummary result={result} /> : null}
      {downloadError ? (
        <div className="alert error" role="alert">
          <AlertCircle size={18} />
          <span>{downloadError}</span>
        </div>
      ) : null}
      {saveMutation.isSuccess ? (
        <div className="alert success" role="status">
          <CheckCircle2 size={18} />
          <span>
            Saved to My Twins. <Link to="/projects">View twins</Link>
          </span>
        </div>
      ) : null}
      {saveMutation.isError ? (
        <div className="alert error" role="alert">
          <AlertCircle size={18} />
          <span>{saveMutation.error instanceof Error ? saveMutation.error.message : "Save failed"}</span>
        </div>
      ) : null}
    </section>
  );
}

function previewToTables(preview: Record<string, unknown> | null): { tables: Array<{ name: string; rowCount: number; rows: Array<Record<string, unknown>> }> } {
  if (!preview) return { tables: [] };
  const tables = preview.tables;
  const rowCounts = preview.row_counts && typeof preview.row_counts === "object" && !Array.isArray(preview.row_counts)
    ? (preview.row_counts as Record<string, unknown>)
    : {};
  if (tables && typeof tables === "object" && !Array.isArray(tables)) {
    return {
      tables: Object.entries(tables as Record<string, unknown>)
        .filter(([, rows]) => Array.isArray(rows))
        .map(([name, rows]) => {
          const previewRows = rows as Array<Record<string, unknown>>;
          const rowCount = typeof rowCounts[name] === "number" ? Number(rowCounts[name]) : previewRows.length;
          return { name, rowCount, rows: previewRows };
        })
    };
  }
  if (Array.isArray(preview.turns)) return { tables: [{ name: "turns", rowCount: preview.turns.length, rows: preview.turns as Array<Record<string, unknown>> }] };
  if (Array.isArray(preview.rows)) return { tables: [{ name: "rows", rowCount: preview.rows.length, rows: preview.rows as Array<Record<string, unknown>> }] };
  return { tables: [{ name: "preview", rowCount: 1, rows: [preview] }] };
}

function WorkflowSummary({ result }: { result: ResultBundle }) {
  const summary = result.summary ?? {};
  const workflow = workflowForType(result.workflow_type);
  const rows = summaryRows(summary);
  return (
    <div className="page-stack">
      <div className="summary-grid">
        <Metric label="Workflow" value={workflow.title} />
        <Metric label="Artifacts" value={result.artifacts.length} />
        <Metric label="Status" value={result.status} />
        <Metric label="Result" value={result.result_id} />
      </div>
      {rows.length ? <DataTable rows={rows} /> : <pre>{JSON.stringify(result.metadata, null, 2)}</pre>}
    </div>
  );
}

function summaryRows(summary: Record<string, unknown>) {
  const rowCounts = summary.row_counts;
  if (rowCounts && typeof rowCounts === "object" && !Array.isArray(rowCounts)) {
    return Object.entries(rowCounts as Record<string, unknown>).map(([table, rows]) => ({ table, rows }));
  }
  const outputFormats = summary.output_formats;
  if (Array.isArray(outputFormats)) {
    return outputFormats.map((format) => ({ output: String(format) }));
  }
  return Object.entries(summary).map(([key, value]) => ({ key, value: formatValue(value) }));
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatValue(value: unknown) {
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value ?? "-");
}

function formatNumber(value: number) {
  return Number.isFinite(value) ? value.toLocaleString() : "-";
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
