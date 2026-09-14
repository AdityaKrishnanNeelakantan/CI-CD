import { ArrowRight, Paperclip, Send } from "lucide-react";
import { Link } from "react-router-dom";
import { workflows } from "../lib/workflows";

const acceptedPromptFiles = ".sql,.json,.csv,.xls,.xlsx,.xlsv,.sqlite,.db,.parquet,.pdf,.txt,.log";

export function HomePage() {
  return (
    <div className="page-stack home-page">
      <section className="home-hero">
        <div className="home-hero-copy">
          <p className="eyebrow">Home / Chat</p>
          <h2>Choose a workflow or describe the synthetic twin you need.</h2>
        </div>
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
      <div className="chat-entry home-chat-entry" aria-label="Chat prompt">
        <div>
          <label className="icon-button attachment-button" title="Attach file">
            <Paperclip size={18} />
            <input
              aria-label="Attach SQL, JSON, CSV, Excel, SQLite, Parquet, PDF, TXT, or LOG files"
              accept={acceptedPromptFiles}
              type="file"
            />
          </label>
          <input placeholder="What synthetic twin would you like to create?" />
          <button className="icon-button" title="Send" type="button">
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
