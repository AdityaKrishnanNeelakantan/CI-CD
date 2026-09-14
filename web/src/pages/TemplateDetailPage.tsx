import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowLeft, Loader2, Play } from "lucide-react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { attachSchemaTemplate, createSchemaSession, getTemplate } from "../api";
import { DataTable } from "../components/DataTable";
import { workflowForType } from "../lib/workflows";

export function TemplateDetailPage() {
  const navigate = useNavigate();
  const { templateId } = useParams();
  const templateQuery = useQuery({
    enabled: Boolean(templateId),
    queryKey: ["template", templateId],
    queryFn: () => getTemplate(templateId!),
    retry: false
  });
  const useTemplate = useMutation({
    mutationFn: async () => {
      const session = await createSchemaSession("Template-backed generation");
      const templateData = await attachSchemaTemplate(session.id, templateId!);
      return templateData;
    },
    onSuccess: (templateData) => navigate("/schema", { state: { templateData } })
  });

  if (templateQuery.isLoading) return <div className="empty">Loading template</div>;
  if (templateQuery.isError || !templateQuery.data) return <div className="alert error">Template could not be loaded.</div>;

  const template = templateQuery.data.template;
  const workflow = workflowForType(template.workflow_type);
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
              <p className="eyebrow">{workflow.title}</p>
              <h2>{template.name}</h2>
            </div>
          </div>
          <div className="button-row">
            <Link className="secondary" to="/templates">
              <ArrowLeft size={17} />
              Templates
            </Link>
            <button className="primary" disabled={!template.can_generate || useTemplate.isPending} onClick={() => useTemplate.mutate()}>
              {useTemplate.isPending ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
              Use Template
            </button>
          </div>
        </div>
        <p className="muted">{template.description}</p>
        <div className="summary-grid">
          <Metric label="Status" value={template.status === "available" ? "Available" : "Unavailable"} />
          <Metric label="Category" value={template.category} />
          <Metric label="Formats" value={template.supported_formats.join(", ")} />
          <Metric label="Fields" value={template.fields.length} />
        </div>
        {!template.can_generate ? <div className="empty">Template generation is unavailable for this item.</div> : null}
        {useTemplate.isError ? <div className="alert error">Template flow could not be started.</div> : null}
      </section>

      {template.schema_preview ? (
        <section className="panel wide">
          <h2>Schema Preview</h2>
          <pre>{JSON.stringify(template.schema_preview, null, 2)}</pre>
        </section>
      ) : null}

      {template.fields.length ? (
        <section className="panel wide">
          <h2>Fields</h2>
          <DataTable rows={template.fields} />
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
