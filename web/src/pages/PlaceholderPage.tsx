import { ArrowRight, BookOpen, CheckCircle2, Database, FileText, TableProperties, Wrench } from "lucide-react";
import { Link } from "react-router-dom";

const guideCards = [
  {
    title: "Schema Twin",
    body: "Upload SQL, JSON, CSV, YAML, or XLSX-style schema files, review inferred columns, then generate validated tabular data.",
    meta: "Schema upload, progress, results, downloads",
    Icon: TableProperties
  },
  {
    title: "Database Twin",
    body: "Use SQLite uploads for the routed workflow. CSV, Parquet, and PostgreSQL are staged for a later connection flow.",
    meta: "SQLite active, database connectors queued",
    Icon: Database
  },
  {
    title: "Document Twin",
    body: "Upload PDFs to extract structure and prepare synthetic document outputs with validation reports.",
    meta: "PDF intake and synthetic document generation",
    Icon: FileText
  }
];

export function PlaceholderPage({ title }: { title: string }) {
  const isTemplates = title === "Templates";
  return (
    <section className="panel wide resource-page">
      <div className="section-heading">
        <div>
          <p className="eyebrow">{title}</p>
          <h2>{title}</h2>
        </div>
        <div className="status-chip">
          {isTemplates ? <Wrench size={15} /> : <BookOpen size={15} />}
          <span>{isTemplates ? "Planned" : "Ready"}</span>
        </div>
      </div>
      {isTemplates ? (
        <div className="resource-grid single">
          <article className="resource-card">
            <div className="resource-icon">
              <LayoutIcon />
            </div>
            <div>
              <h3>Template-backed generation</h3>
              <p>Reusable templates are coming soon. Today, Schema Twin supports schema file upload as the production path.</p>
            </div>
            <Link className="secondary" to="/schema">
              Open Schema Twin
              <ArrowRight size={16} />
            </Link>
          </article>
        </div>
      ) : (
        <div className="resource-grid">
          {guideCards.map(({ Icon, body, meta, title: cardTitle }) => (
            <article className="resource-card" key={cardTitle}>
              <div className="resource-icon">
                <Icon size={20} />
              </div>
              <div>
                <h3>{cardTitle}</h3>
                <p>{body}</p>
                <span>{meta}</span>
              </div>
              <CheckCircle2 className="resource-check" size={18} />
            </article>
          ))}
          <article className="resource-card accent">
            <div className="resource-icon">
              <BookOpen size={20} />
            </div>
            <div>
              <h3>Customer Interaction Twin</h3>
              <p>Use TXT and LOG transcripts to produce structured synthetic conversations with optional sensitive information removal.</p>
              <span>Transcript parsing and privacy controls</span>
            </div>
          </article>
        </div>
      )}
    </section>
  );
}

function LayoutIcon() {
  return <Wrench size={20} />;
}
