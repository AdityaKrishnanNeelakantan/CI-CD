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
  const [downloadError, setDownloadError] = useState("");
  const workflow = workflowForType(result.workflow_type);
  const WorkflowIcon = workflow.Icon;
  const previewRows = useMemo(() => previewToRows(result.preview), [result.preview]);
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
            Save to My Projects
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
      {tab === "preview" ? <DataTable rows={previewRows} /> : null}
      {tab === "files" ? <DownloadList artifacts={result.artifacts} onDownload={handleDownload} /> : null}
      {tab === "quality" ? <QualitySummary report={result.quality_report} /> : null}
      {tab === "summary" ? <pre>{JSON.stringify(result.summary ?? result.metadata, null, 2)}</pre> : null}
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
            Saved to My Projects. <Link to="/projects">View projects</Link>
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

function previewToRows(preview: Record<string, unknown> | null): Array<Record<string, unknown>> {
  if (!preview) return [];
  const tables = preview.tables;
  if (tables && typeof tables === "object" && !Array.isArray(tables)) {
    const first = Object.values(tables as Record<string, unknown>)[0];
    if (Array.isArray(first)) return first as Array<Record<string, unknown>>;
  }
  if (Array.isArray(preview.rows)) return preview.rows as Array<Record<string, unknown>>;
  return [preview];
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
