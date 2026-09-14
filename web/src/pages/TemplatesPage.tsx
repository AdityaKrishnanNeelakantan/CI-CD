import { useQuery } from "@tanstack/react-query";
import { Layers, Lock } from "lucide-react";
import { Link } from "react-router-dom";
import { getTemplates } from "../api";
import { workflowForType } from "../lib/workflows";

export function TemplatesPage() {
  const templatesQuery = useQuery({ queryKey: ["templates"], queryFn: getTemplates, retry: false });
  const templates = templatesQuery.data?.templates ?? [];
  const groups = groupByCategory(templates);

  return (
    <div className="page-stack">
      <section className="workflow-page-heading">
        <span className="workflow-icon">
          <Layers size={24} />
        </span>
        <div>
          <p className="eyebrow">Templates</p>
          <h2>Template Library</h2>
          <p>Schema templates can start generation today. Database, document, and interaction templates are metadata only.</p>
        </div>
      </section>
      {templatesQuery.isLoading ? <div className="empty">Loading templates</div> : null}
      {templatesQuery.isError ? <div className="alert error">Templates API unavailable.</div> : null}
      {Object.entries(groups).map(([category, rows]) => (
        <section className="panel wide" key={category}>
          <h2>{category}</h2>
          <div className="resource-grid">
            {rows.map((template) => {
              const workflow = workflowForType(template.workflow_type);
              const WorkflowIcon = workflow.Icon;
              return (
                <Link
                  className={`resource-card ${template.can_generate ? "accent" : ""}`}
                  key={template.template_id}
                  to={`/templates/${template.template_id}`}
                >
                  <span className={`workflow-icon workflow-${workflow.accent}`}>
                    <WorkflowIcon size={22} />
                  </span>
                  <div>
                    <h3>{template.name}</h3>
                    <p>{template.description}</p>
                    <span>{template.status === "available" ? "Available" : "Coming soon"}</span>
                  </div>
                  {template.can_generate ? <Layers className="resource-check" size={20} /> : <Lock className="resource-check" size={20} />}
                </Link>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}

function groupByCategory<T extends { category: string }>(templates: T[]) {
  return templates.reduce<Record<string, T[]>>((groups, template) => {
    const key = template.category.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
    groups[key] = [...(groups[key] ?? []), template];
    return groups;
  }, {});
}
