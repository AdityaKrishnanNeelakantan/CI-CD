import { ArrowRight, Send } from "lucide-react";
import { Link } from "react-router-dom";
import { workflows } from "../lib/workflows";

export function HomePage() {
  return (
    <div className="page-stack">
      <section className="home-hero">
        <div>
          <p className="eyebrow">Home / Chat</p>
          <h2>Choose a workflow or describe the synthetic twin you need.</h2>
        </div>
        <label className="chat-entry">
          <span>Prompt</span>
          <div>
            <input placeholder="Generate privacy-preserving customer order data for QA" />
            <button className="icon-button" title="Send" type="button">
              <Send size={18} />
            </button>
          </div>
        </label>
      </section>
      <section className="workflow-grid" aria-label="Workflow cards">
        {workflows.map(({ Icon, accent, description, formats, route, title }) => (
          <Link className={`workflow-card workflow-${accent}`} key={route} to={route}>
            <span className="workflow-icon">
              <Icon size={24} />
            </span>
            <div>
              <h3>{title}</h3>
              <p>{description}</p>
              <span className="workflow-formats">{formats}</span>
            </div>
            <ArrowRight size={18} />
          </Link>
        ))}
      </section>
    </div>
  );
}
