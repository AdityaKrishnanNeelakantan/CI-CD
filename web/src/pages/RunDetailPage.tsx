import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ExternalLink } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { ArtifactFileMetadata, fetchDownload, getRun } from "../api";
import { DownloadList } from "../components/DownloadList";
import { QualitySummary } from "../components/QualitySummary";
import { workflowForType } from "../lib/workflows";

export function RunDetailPage() {
  const { projectId, runId } = useParams();
  const runQuery = useQuery({
    enabled: Boolean(projectId && runId),
    queryKey: ["project-run", projectId, runId],
    queryFn: () => getRun(projectId!, runId!),
    retry: false
  });

  if (runQuery.isLoading) return <div className="empty">Loading run</div>;
  if (runQuery.isError || !runQuery.data) return <div className="alert error">Run could not be loaded.</div>;

  const run = runQuery.data.run;
  const workflow = workflowForType(run.workflow_type);
  const WorkflowIcon = workflow.Icon;

  return (
    <div className="page-stack">
      <section className="panel wide">
        <div className="results-header">
          <div className="workflow-page-heading inline-heading">
            <span className={`workflow-icon workflow-${workflow.accent}`}>
              <WorkflowIcon size={24} />
            </span>
            <div>
              <p className="eyebrow">Run</p>
              <h2>{run.run_id}</h2>
            </div>
          </div>
          <div className="button-row">
            <Link className="secondary" to={`/projects/${run.project_id}`}>
              <ArrowLeft size={17} />
              Project
            </Link>
            {run.result_id ? (
              <Link className="primary" to={`/results/${run.result_id}`}>
                <ExternalLink size={17} />
                Result
              </Link>
            ) : null}
          </div>
        </div>
        <div className="summary-grid">
          <Metric label="Workflow" value={workflow.title} />
          <Metric label="Status" value={run.status} />
          <Metric label="Created" value={formatDate(run.created_at)} />
          <Metric label="Completed" value={formatDate(run.completed_at)} />
        </div>
      </section>

      {run.artifacts.length ? (
        <section className="panel wide">
          <h2>Artifacts</h2>
          <DownloadList artifacts={run.artifacts} onDownload={downloadArtifact} />
        </section>
      ) : null}

      {run.quality_report ? (
        <section className="panel wide">
          <h2>Quality</h2>
          <QualitySummary report={run.quality_report} />
        </section>
      ) : null}

      <section className="two-column">
        <div className="panel">
          <h2>Input</h2>
          <pre>{JSON.stringify(run.input, null, 2)}</pre>
        </div>
        <div className="panel">
          <h2>Config</h2>
          <pre>{JSON.stringify(run.config, null, 2)}</pre>
        </div>
      </section>
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

function formatDate(value?: string | null) {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

async function downloadArtifact(artifact: ArtifactFileMetadata) {
  if (!artifact.download_url) return;
  const blob = await fetchDownload(artifact.download_url);
  if (!("createObjectURL" in URL)) return;
  const href = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = href;
  link.download = artifact.filename ?? artifact.name;
  link.click();
  URL.revokeObjectURL(href);
}
