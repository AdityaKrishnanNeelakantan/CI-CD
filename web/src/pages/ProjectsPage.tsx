import { useQuery } from "@tanstack/react-query";
import { Plus } from "lucide-react";
import { Link } from "react-router-dom";
import { getProjects } from "../api";
import type { ProjectSummary } from "../api";

export function ProjectsPage() {
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: () => getProjects(), retry: false });
  const projects = projectsQuery.data?.projects ?? [];
  return (
    <section className="panel wide">
      <div className="results-header">
        <div>
          <h2>My Twins</h2>
        </div>
        <Link className="primary" to="/">
          <Plus size={17} />
          New Project
        </Link>
      </div>
      {projects.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Workflow</th>
                <th>Generated Rows</th>
                <th>Runs</th>
                <th>Created</th>
                <th>Updated</th>
                <th>Status</th>
                <th>Validation</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((project) => (
                <tr key={project.id}>
                  <td>
                    <Link className="project-name-link" to={projectTarget(project)}>
                      <span>{project.name}</span>
                    </Link>
                  </td>
                  <td>{project.workflow_label ?? project.type ?? project.workflow_type ?? "synthetic_twin"}</td>
                  <td>{project.records ?? project.record_count ?? "-"}</td>
                  <td>{project.run_count ?? (project.latest_run_id ? 1 : 0)}</td>
                  <td>{project.created_on ?? project.created_at ?? project.updated_at ?? "-"}</td>
                  <td>{project.updated_on ?? project.updated_at ?? "-"}</td>
                  <td><span className={`status-chip ${statusTone(project.status)}`}>{label(project.status ?? "available")}</span></td>
                  <td>{project.validation ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty">{projectsQuery.isError ? "My Twins API unavailable" : "No generated twins yet"}</div>
      )}
    </section>
  );
}

function projectTarget(project: ProjectSummary) {
  return `/projects/${project.id}`;
}

function statusTone(status?: string | null) {
  if (status === "completed" || status === "available") return "passed";
  if (status === "failed") return "failed";
  return "pending";
}

function label(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
