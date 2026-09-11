import { useQuery } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { getJob } from "../api";
import { ProgressTimeline } from "../components/ProgressTimeline";
import { workflowForType } from "../lib/workflows";

export function ProgressPage() {
  const { jobId } = useParams();
  const jobQuery = useQuery({
    enabled: Boolean(jobId),
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId!),
    refetchInterval: 1000,
    retry: false
  });
  const job = jobQuery.data;
  const workflow = workflowForType(job?.workflow_type);
  const WorkflowIcon = workflow.Icon;

  return (
    <div className="page-stack">
      <section className="panel wide">
        <div className="workflow-page-heading inline-heading">
          <span className={`workflow-icon workflow-${workflow.accent}`}>
            <WorkflowIcon size={24} />
          </span>
          <div>
            <p className="eyebrow">{workflow.title}</p>
            <h2>Generation Progress</h2>
          </div>
        </div>
        <ProgressTimeline job={job} />
        {job?.status === "succeeded" && job.result_id ? (
          <Link className="primary" to={`/results/${job.result_id}`}>
            View Results
            <ArrowRight size={17} />
          </Link>
        ) : null}
        {job?.status === "failed" ? (
          <Link className="secondary" to={workflow.route}>
            Retry {workflow.title}
          </Link>
        ) : null}
        {job?.status === "running" || job?.status === "queued" ? (
          <p className="muted">You can leave this page and return from the job link while polling continues on the backend.</p>
        ) : null}
        {jobQuery.isError ? <div className="alert error">Job status could not be loaded.</div> : null}
      </section>
    </div>
  );
}
