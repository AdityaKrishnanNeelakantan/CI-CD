import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { Job } from "../api";
import { generationStages } from "../lib/workflows";

function normalizeStage(stage: string) {
  return stage.replaceAll("_", " ").toLowerCase();
}

export function ProgressTimeline({ job }: { job: Job | null | undefined }) {
  if (!job) return <div className="empty">Waiting for job status</div>;
  const active = normalizeStage(job.message || job.stage);
  const activeIndex = Math.max(
    0,
    generationStages.findIndex((stage) => active.includes(stage.toLowerCase()))
  );
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
        {generationStages.map((stage, index) => (
          <li className={timelineClass(job.status, index, activeIndex)} key={stage}>
            <span>{index < activeIndex || job.status === "succeeded" ? <CheckCircle2 size={13} /> : null}</span>
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

function timelineClass(status: Job["status"], index: number, activeIndex: number) {
  if (status === "succeeded") return "done";
  if (status === "failed" && index === activeIndex) return "failed";
  if (index < activeIndex) return "done";
  if (index === activeIndex && (status === "running" || status === "queued")) return "active";
  return "";
}
