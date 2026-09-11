import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ExternalLink, RotateCcw } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { fetchDownload, getProject } from "../api";
import type { ArtifactFileMetadata, ProjectDetail } from "../api";
import { DownloadList } from "../components/DownloadList";
import { QualitySummary } from "../components/QualitySummary";
import { workflowForType } from "../lib/workflows";

export function ProjectDetailPage() {
  const { projectId } = useParams();
  const projectQuery = useQuery({
    enabled: Boolean(projectId),
    queryKey: ["project", projectId],
    queryFn: () => getProject(projectId!),
    retry: false
  });

  if (projectQuery.isLoading) return <div className="empty">Loading project</div>;
  if (projectQuery.isError || !projectQuery.data) return <div className="alert error">Project could not be loaded.</div>;

  const project = projectQuery.data.project;
  const workflow = workflowForType(project.workflow_type);
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
              <p className="eyebrow">My Twin</p>
              <h2>{project.name}</h2>
            </div>
          </div>
          <div className="button-row">
            <Link className="secondary" to="/projects">
              <ArrowLeft size={17} />
              My Twins
            </Link>
            <Link className="primary" to={workflow.route}>
              <RotateCcw size={17} />
              New Run
            </Link>
          </div>
        </div>
        <div className="summary-grid">
          <Metric label="Workflow" value={workflow.title} />
          <Metric label="Status" value={label(project.status)} />
          <Metric label="Generated Rows" value={recordsLabel(project)} />
          <Metric label="Validation" value={project.validation ?? validationFromQuality(project.quality_report)} />
          <Metric label="Runs" value={project.runs.length} />
          <Metric label="Artifacts" value={project.artifacts?.length ?? 0} />
          <Metric label="Created" value={formatDate(project.created_at)} />
          <Metric label="Updated" value={formatDate(project.updated_at)} />
        </div>
      </section>

      <section className="panel wide">
        <h2>Runs</h2>
        {project.runs.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Validation</th>
                  <th>Generated Rows</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {project.runs.map((run) => (
                  <tr key={run.run_id}>
                    <td>
                      <Link className="project-name-link" to={`/projects/${project.id}/runs/${run.run_id}`}>
                        <span className="mono-token">{shortId(run.run_id)}</span>
                      </Link>
                    </td>
                    <td><span className={`status-chip ${statusTone(run.status)}`}>{label(run.status)}</span></td>
                    <td>{formatDate(run.created_at)}</td>
                    <td>{run.validation ?? validationLabel(run.validation_passed, run.validation_status)}</td>
                    <td>{run.records ?? "-"}</td>
                    <td>
                      {run.result_id ? (
                        <Link className="text-link inline-link" to={`/results/${run.result_id}`}>
                          <ExternalLink size={15} />
                          Open result
                        </Link>
                      ) : (
                        <Link className="text-link inline-link" to={`/projects/${project.id}/runs/${run.run_id}`}>
                          View run
                        </Link>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">No runs recorded for this project</div>
        )}
      </section>

      {project.artifacts?.length ? (
        <section className="panel wide">
          <h2>Latest Artifacts</h2>
          <DownloadList artifacts={project.artifacts} onDownload={downloadArtifact} />
        </section>
      ) : null}

      {project.quality_report ? (
        <section className="panel wide">
          <h2>Latest Quality</h2>
          <QualitySummary report={project.quality_report} />
        </section>
      ) : null}
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

function recordsLabel(project: ProjectDetail) {
  if (project.records) return project.records;
  const rowCounts = project.summary?.row_counts;
  if (!rowCounts || typeof rowCounts !== "object") return "-";
  const total = Object.values(rowCounts).reduce((sum, value) => sum + (typeof value === "number" ? value : 0), 0);
  return total ? `${total.toLocaleString()} total` : "-";
}

function validationFromQuality(report?: Record<string, unknown> | null) {
  if (!report) return "-";
  const passed = report.passed ?? report.hard_checks_passed;
  if (passed === true) return "Passed";
  if (passed === false) return "Failed";
  const status = report.status;
  return typeof status === "string" ? label(status) : "-";
}

function validationLabel(passed: boolean | null, status: string | null) {
  if (passed === true) return "Passed";
  if (passed === false) return "Failed";
  return label(status);
}

function statusTone(status?: string | null) {
  if (status === "completed" || status === "available") return "passed";
  if (status === "failed") return "failed";
  return "pending";
}

function label(value?: string | null) {
  if (!value) return "-";
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function shortId(value: string) {
  return value.length > 14 ? `${value.slice(0, 12)}...` : value;
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
