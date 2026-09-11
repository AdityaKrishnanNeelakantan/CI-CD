import { useQuery } from "@tanstack/react-query";
import { Eye, MoreHorizontal, Plus } from "lucide-react";
import { Link } from "react-router-dom";
import { getProjects } from "../api";

export function ProjectsPage() {
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: () => getProjects(), retry: false });
  const projects = projectsQuery.data?.projects ?? [];
  return (
    <section className="panel wide">
      <div className="results-header">
        <div>
          <p className="eyebrow">My Twin</p>
          <h2>Generated Twins</h2>
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
                <th>Type</th>
                <th>Records</th>
                <th>Created</th>
                <th>Status</th>
                <th>Validation</th>
                <th>Transfer</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((project) => (
                <tr key={project.id}>
                  <td>
                    <Link className="text-link" to={`/projects/${project.id}`}>
                      {project.name}
                    </Link>
                  </td>
                  <td>{project.workflow_label ?? project.type ?? project.workflow_type ?? "synthetic_twin"}</td>
                  <td>{project.records ?? project.record_count ?? "-"}</td>
                  <td>{project.created_on ?? project.created_at ?? project.updated_at ?? "-"}</td>
                  <td>{project.status ?? "available"}</td>
                  <td>{project.validation ?? "-"}</td>
                  <td>{project.transfer ?? "-"}</td>
                  <td>
                    <div className="button-row">
                      <Link
                        className="icon-button"
                        title="View"
                        to={`/projects/${project.id}`}
                      >
                        <Eye size={17} />
                      </Link>
                      <button className="icon-button" disabled title="More options">
                        <MoreHorizontal size={17} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty">{projectsQuery.isError ? "Projects API unavailable" : "No generated projects yet"}</div>
      )}
    </section>
  );
}
