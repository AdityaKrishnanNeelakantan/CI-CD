import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { Job } from "../api";
import { generationStages } from "../lib/workflows";

function normalizeStage(stage: string) {
  return stage.replaceAll("_", " ").toLowerCase();
}

export function ProgressTimeline({ job }: { job: Job | null | undefined }) {
  if (!job) return <div className="empty">Waiting for job status</div>;
  const active = normalizeStage(job.message || job.stage);
  return (
    <div className="progress-panel">
      <div className={`job-status ${job.status}`}>
        {job.status === "running" || job.status === "queued" ? <Loader2 className="spin" size={18} /> : null}
        {job.status === "succeeded" ? <CheckCircle2 size={18} /> : null}
        {job.status === "failed" ? <AlertCircle size={18} /> : null}
        <strong>{job.status}</strong>
        <span>{job.percent}%</span>
      </div>
      <div className="progress-bar" aria-label="Generation progress">
        <span style={{ width: `${Math.max(0, Math.min(100, job.percent))}%` }} />
      </div>
      <ol className="timeline">
        {generationStages.map((stage) => (
          <li className={active.includes(stage.toLowerCase()) ? "active" : ""} key={stage}>
            <span />
            {stage}
          </li>
        ))}
      </ol>
      {job.message ? <p className="status-message">{job.message}</p> : null}
      {job.error ? (
        <div className="alert error" role="alert">
          <AlertCircle size={18} />
          <span>{job.error.message}</span>
        </div>
      ) : null}
    </div>
  );
}
