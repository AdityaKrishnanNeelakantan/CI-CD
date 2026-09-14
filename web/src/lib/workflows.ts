import { Database, FileText, LucideIcon, MessageSquareText, TableProperties } from "lucide-react";

export type WorkflowKey = "schema" | "database" | "document" | "interaction";

export type WorkflowConfig = {
  key: WorkflowKey;
  workflowType: string;
  title: string;
  description: string;
  route: string;
  formats: string;
  steps: string[];
  Icon: LucideIcon;
  accent: string;
};

export const generationStages = [
  "Analyzing input",
  "Learning data patterns",
  "Generating synthetic data",
  "Validating data quality",
  "Applying privacy checks",
  "Preparing files"
];

export const workflows: WorkflowConfig[] = [
  {
    key: "schema",
    workflowType: "schema_twin",
    title: "Schema Twin",
    description: "Generate synthetic tables from SQL, JSON, CSV, or XLSX-style schema inputs.",
    route: "/schema",
    formats: "SQL, JSON, CSV, XLSX",
    steps: ["Input", "Review", "Generate", "Results"],
    Icon: TableProperties,
    accent: "schema"
  },
  {
    key: "database",
    workflowType: "database_twin",
    title: "Database Twin",
    description: "Inspect a database or tabular sample, configure privacy controls, then generate a twin.",
    route: "/database",
    formats: "SQLite, CSV, Parquet",
    steps: ["Connect", "Configure", "Generate", "Results"],
    Icon: Database,
    accent: "database"
  },
  {
    key: "document",
    workflowType: "pdf_twin",
    title: "Document Twin",
    description: "Extract document structure and prepare synthetic PDFs with reports.",
    route: "/document",
    formats: "PDF",
    steps: ["Upload", "Configure", "Generate", "Results"],
    Icon: FileText,
    accent: "document"
  },
  {
    key: "interaction",
    workflowType: "interaction_twin",
    title: "Customer Interaction Twin",
    description: "Parse transcripts and produce structured synthetic conversations or logs.",
    route: "/interaction",
    formats: "TXT, LOG",
    steps: ["Upload", "Configure", "Generate", "Results"],
    Icon: MessageSquareText,
    accent: "interaction"
  }
];

export function workflowForType(workflowType?: string | null): WorkflowConfig {
  return workflows.find((workflow) => workflow.workflowType === workflowType) ?? workflows[0];
}
