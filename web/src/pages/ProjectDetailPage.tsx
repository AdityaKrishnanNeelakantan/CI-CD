import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Eye, Play, RotateCcw } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { ArtifactFileMetadata, fetchDownload, getProject } from "../api";
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
              <p className="eyebrow">Project</p>
              <h2>{project.name}</h2>
            </div>
          </div>
          <div className="button-row">
            <Link className="secondary" to="/projects">
              <ArrowLeft size={17} />
              Projects
            </Link>
            <Link className="primary" to={workflow.route}>
              <RotateCcw size={17} />
              New Run
            </Link>
          </div>
        </div>
        <div className="summary-grid">
          <Metric label="Workflow" value={workflow.title} />
          <Metric label="Status" value={project.status} />
          <Metric label="Created" value={formatDate(project.created_at)} />
          <Metric label="Updated" value={formatDate(project.updated_at)} />
        </div>
        {project.latest_result_id ? (
          <Link className="secondary" to={`/results/${project.latest_result_id}`}>
            <Eye size={17} />
            Latest Result
          </Link>
        ) : null}
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
                  <th>Transfer</th>
                  <th>Result</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {project.runs.map((run) => (
                  <tr key={run.run_id}>
                    <td>{run.run_id}</td>
                    <td>{run.status}</td>
                    <td>{formatDate(run.created_at)}</td>
                    <td>{validationLabel(run.validation_passed, run.validation_status)}</td>
                    <td>{run.transfer_status}</td>
                    <td>{run.result_id ?? "-"}</td>
                    <td>
                      <Link className="icon-button" title="View run" to={`/projects/${project.project_id}/runs/${run.run_id}`}>
                        <Play size={17} />
                      </Link>
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

function validationLabel(passed: boolean | null, status: string | null) {
  if (passed === true) return "Passed";
  if (passed === false) return "Failed";
  return status ?? "-";
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
