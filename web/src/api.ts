export type ApiEnvelope<T> = { data: T };
export type ApiErrorEnvelope = { error: { code: string; message: string; details?: unknown } };

export type ApiErrorCode =
  | "api_error"
  | "generation_failed"
  | "llm_policy_error"
  | "schema_parse_error"
  | "transfer_blocked"
  | "validation_error";

export class ApiError extends Error {
  code: ApiErrorCode | string;
  status: number;

  constructor(message: string, code: ApiErrorCode | string = "api_error", status = 0) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

export type SchemaSession = {
  id: string;
  workflow: string;
  workflow_type: string;
  state: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type WorkflowSession = SchemaSession;

export type SchemaSummary = {
  name: string;
  table_count: number;
  column_count: number;
  relationship_count: number;
  tables: Array<Record<string, string | number>>;
};

export type SchemaColumn = {
  table: string;
  column: string;
  type: string;
  role: string;
  nullable: string;
  references: string;
  description: string;
};

export type LlmTextColumn = {
  table: string;
  column: string;
  role: string;
};

export type UploadSchemaResponse = {
  session: SchemaSession;
  summary: SchemaSummary;
  columns: SchemaColumn[];
  llm_text_columns: LlmTextColumn[];
};

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export type Job = {
  id: string;
  job_id: string;
  session_id: string | null;
  workflow_type: string | null;
  kind: string;
  status: JobStatus;
  stage: string;
  percent: number;
  message: string;
  progress: string[];
  error: null | { code: string; message: string };
  result_id: string | null;
  result: null | Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type ArtifactFileMetadata = {
  id: string;
  artifact_id: string | null;
  name: string;
  filename: string | null;
  path: string;
  media_type: string;
  content_type: string | null;
  size: number | null;
  size_bytes: number | null;
  role: string | null;
  kind: string | null;
  downloadable: boolean;
  download_url: string | null;
  metadata: Record<string, unknown>;
};

export type ResultBundle = {
  id: string;
  result_id: string;
  session_id: string;
  workflow_type: string;
  status: string;
  preview: Record<string, unknown> | null;
  quality_report: Record<string, unknown> | null;
  summary: Record<string, unknown> | null;
  artifacts: ArtifactFileMetadata[];
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type GenerateConfig = {
  row_count: number;
  locale: string;
  seed: number;
  export_format: "csv" | "parquet";
  preview_rows: number;
  llm_text_enabled: boolean;
};

export type PreviewResponse = {
  tables: Record<string, Array<Record<string, unknown>>>;
  row_counts: Record<string, number>;
};

export type ValidationResponse = {
  report: Record<string, unknown>;
  highlights: {
    hard_checks_passed: boolean;
    status: string;
    issues: unknown[];
    row_counts: Record<string, number>;
    tables: string[];
    export_count: number;
  };
};

export type SettingsResponse = {
  default_record_count: number;
  default_output_format: string;
  generation_mode: string;
  privacy_level: string;
};

export type ProjectSummary = {
  id: string;
  project_id?: string;
  name: string;
  type?: string;
  workflow_type?: string;
  workflow_label?: string;
  records?: string | number;
  record_count?: number;
  latest_run_id?: string | null;
  latest_result_id?: string | null;
  created_at?: string;
  created_on?: string;
  updated_at?: string;
  status?: string;
  validation?: string;
  transfer?: string;
};

export type RunSummary = {
  id: string;
  run_id: string;
  project_id: string;
  workflow_type: string;
  status: string;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  result_id: string | null;
  job_id: string | null;
  validation_status: string | null;
  validation_passed: boolean | null;
  transfer_status: string;
  transfer_allowed: boolean | null;
  transfer_attempted_at: string | null;
  metadata: Record<string, unknown>;
};

export type ProjectDetail = ProjectSummary & {
  project_id: string;
  workflow_type: string;
  created_at: string;
  updated_at: string;
  status: string;
  latest_run_id: string | null;
  latest_result_id: string | null;
  metadata: Record<string, unknown>;
  summary: Record<string, unknown> | null;
  quality_report: Record<string, unknown> | null;
  artifacts: ArtifactFileMetadata[] | null;
  runs: RunSummary[];
};

export type RunDetail = RunSummary & {
  summary: Record<string, unknown> | null;
  quality_report: Record<string, unknown> | null;
  artifacts: ArtifactFileMetadata[];
  input: Record<string, unknown>;
  config: Record<string, unknown>;
};

export type ProjectsResponse = {
  projects: ProjectSummary[];
};

export type ProjectDetailResponse = {
  project: ProjectDetail;
};

export type ProjectRunsResponse = {
  runs: RunSummary[];
};

export type RunDetailResponse = {
  run: RunDetail;
};

export type SaveResultResponse = {
  project_id: string;
  run_id: string;
  saved: boolean;
  project: ProjectSummary;
};

export type DatabaseConfigureRequest = {
  row_count?: number;
  row_counts_by_table?: Record<string, number>;
  sample_limit?: number;
  seed?: number;
  model_type?: string;
};

export type DocumentConfigureRequest = {
  seed?: number;
  extraction_method?: string;
  llm_text_enabled?: boolean;
};

export type InteractionConfigureRequest = {
  interaction_type?: string;
  output_format?: string;
  remove_sensitive_information?: boolean;
  seed?: number;
};

export type TemplateMetadata = {
  template_id: string;
  name: string;
  description: string;
  workflow_type: string;
  category: string;
  supported_formats: string[];
  status: "available" | "coming_soon";
  can_generate: boolean;
  fields: Array<Record<string, unknown>>;
  schema_preview?: Record<string, unknown>;
  schema?: Record<string, unknown>;
};

export type TemplatesResponse = {
  templates: TemplateMetadata[];
};

export type TemplateDetailResponse = {
  template: TemplateMetadata;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: init?.body instanceof FormData ? init.headers : { "Content-Type": "application/json", ...init?.headers }
  });
  if (!response.ok) {
    let parsed: ApiErrorEnvelope | undefined;
    try {
      parsed = (await response.json()) as ApiErrorEnvelope;
    } catch {
      parsed = undefined;
    }
    throw new ApiError(parsed?.error?.message ?? response.statusText, parsed?.error?.code, response.status);
  }
  return ((await response.json()) as ApiEnvelope<T>).data;
}

export function getSettings(): Promise<SettingsResponse> {
  return requestJson<SettingsResponse>("/api/settings");
}

export function updateSettings(settings: Partial<SettingsResponse>): Promise<SettingsResponse> {
  return requestJson<SettingsResponse>("/api/settings", {
    method: "PUT",
    body: JSON.stringify(settings)
  });
}

export function getProjects(workflowType?: string): Promise<ProjectsResponse> {
  const query = workflowType ? `?workflow_type=${encodeURIComponent(workflowType)}` : "";
  return requestJson<ProjectsResponse>(`/api/projects${query}`);
}

export function getProject(projectId: string): Promise<ProjectDetailResponse> {
  return requestJson<ProjectDetailResponse>(`/api/projects/${projectId}`);
}

export function getProjectRuns(projectId: string): Promise<ProjectRunsResponse> {
  return requestJson<ProjectRunsResponse>(`/api/projects/${projectId}/runs`);
}

export function getRun(projectId: string, runId: string): Promise<RunDetailResponse> {
  return requestJson<RunDetailResponse>(`/api/projects/${projectId}/runs/${runId}`);
}

export function getTemplates(): Promise<TemplatesResponse> {
  return requestJson<TemplatesResponse>("/api/templates");
}

export function getTemplate(templateId: string): Promise<TemplateDetailResponse> {
  return requestJson<TemplateDetailResponse>(`/api/templates/${templateId}`);
}

export function createSchemaSession(intent: string): Promise<SchemaSession> {
  return requestJson<SchemaSession>("/api/schema/sessions", {
    method: "POST",
    body: JSON.stringify({ intent })
  });
}

export function createDatabaseSession(intent: string): Promise<WorkflowSession> {
  return requestJson<WorkflowSession>("/api/database/sessions", {
    method: "POST",
    body: JSON.stringify({ intent })
  });
}

export function getDatabaseSession(sessionId: string): Promise<WorkflowSession> {
  return requestJson<WorkflowSession>(`/api/database/sessions/${sessionId}`);
}

export function uploadDatabaseSource(sessionId: string, file: File, sourceType = "sqlite"): Promise<WorkflowSession> {
  const body = new FormData();
  body.append("source_type", sourceType);
  body.append("file", file);
  return requestJson<WorkflowSession>(`/api/database/sessions/${sessionId}/source`, {
    method: "POST",
    body
  });
}

export function configureDatabaseSession(
  sessionId: string,
  config: DatabaseConfigureRequest
): Promise<WorkflowSession> {
  return requestJson<WorkflowSession>(`/api/database/sessions/${sessionId}/configure`, {
    method: "POST",
    body: JSON.stringify(config)
  });
}

export async function startDatabaseGeneration(sessionId: string): Promise<Job> {
  const data = await requestJson<{ job: Job }>(`/api/database/sessions/${sessionId}/generate`, {
    method: "POST"
  });
  return data.job;
}

export function createDocumentSession(intent: string): Promise<WorkflowSession> {
  return requestJson<WorkflowSession>("/api/document/sessions", {
    method: "POST",
    body: JSON.stringify({ intent })
  });
}

export function getDocumentSession(sessionId: string): Promise<WorkflowSession> {
  return requestJson<WorkflowSession>(`/api/document/sessions/${sessionId}`);
}

export function uploadDocument(sessionId: string, file: File): Promise<WorkflowSession> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<WorkflowSession>(`/api/document/sessions/${sessionId}/upload`, {
    method: "POST",
    body
  });
}

export function configureDocumentSession(
  sessionId: string,
  config: DocumentConfigureRequest
): Promise<WorkflowSession> {
  return requestJson<WorkflowSession>(`/api/document/sessions/${sessionId}/configure`, {
    method: "POST",
    body: JSON.stringify(config)
  });
}

export async function startDocumentGeneration(sessionId: string): Promise<Job> {
  const data = await requestJson<{ job: Job }>(`/api/document/sessions/${sessionId}/generate`, {
    method: "POST"
  });
  return data.job;
}

export function createInteractionSession(intent: string): Promise<WorkflowSession> {
  return requestJson<WorkflowSession>("/api/interaction/sessions", {
    method: "POST",
    body: JSON.stringify({ intent })
  });
}

export function getInteractionSession(sessionId: string): Promise<WorkflowSession> {
  return requestJson<WorkflowSession>(`/api/interaction/sessions/${sessionId}`);
}

export function uploadInteractionTranscript(sessionId: string, file: File): Promise<WorkflowSession> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<WorkflowSession>(`/api/interaction/sessions/${sessionId}/upload`, {
    method: "POST",
    body
  });
}

export function configureInteractionSession(
  sessionId: string,
  config: InteractionConfigureRequest
): Promise<WorkflowSession> {
  return requestJson<WorkflowSession>(`/api/interaction/sessions/${sessionId}/configure`, {
    method: "POST",
    body: JSON.stringify(config)
  });
}

export async function startInteractionGeneration(sessionId: string): Promise<Job> {
  const data = await requestJson<{ job: Job }>(`/api/interaction/sessions/${sessionId}/generate`, {
    method: "POST"
  });
  return data.job;
}

export function uploadSchemaFile(sessionId: string, file: File): Promise<UploadSchemaResponse> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<UploadSchemaResponse>(`/api/schema/sessions/${sessionId}/schema-file`, {
    method: "POST",
    body
  });
}

export function attachSchemaTemplate(sessionId: string, templateId: string): Promise<UploadSchemaResponse> {
  return requestJson<UploadSchemaResponse>(`/api/schema/sessions/${sessionId}/template`, {
    method: "POST",
    body: JSON.stringify({ template_id: templateId })
  });
}

export async function startSchemaGeneration(sessionId: string, config: GenerateConfig): Promise<Job> {
  const data = await requestJson<{ job: Job }>(`/api/schema/sessions/${sessionId}/generate`, {
    method: "POST",
    body: JSON.stringify(config)
  });
  return data.job;
}

export function getJob(jobId: string): Promise<Job> {
  return requestJson<Job>(`/api/jobs/${jobId}`);
}

export function getResult(resultId: string): Promise<ResultBundle> {
  return requestJson<ResultBundle>(`/api/results/${resultId}`);
}

export function saveResult(resultId: string): Promise<SaveResultResponse> {
  return requestJson<SaveResultResponse>(`/api/results/${resultId}/save`, {
    method: "POST"
  });
}

export function getPreview(sessionId: string): Promise<PreviewResponse> {
  return requestJson<PreviewResponse>(`/api/schema/sessions/${sessionId}/preview`);
}

export function getValidation(sessionId: string): Promise<ValidationResponse> {
  return requestJson<ValidationResponse>(`/api/schema/sessions/${sessionId}/validation`);
}

export async function createDownload(sessionId: string): Promise<{ download_id: string; url: string }> {
  return requestJson<{ download_id: string; url: string }>(`/api/schema/sessions/${sessionId}/download`, {
    method: "POST"
  });
}

export async function fetchDownload(downloadUrl: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}${downloadUrl}`);
  if (!response.ok) {
    let parsed: ApiErrorEnvelope | undefined;
    try {
      parsed = (await response.json()) as ApiErrorEnvelope;
    } catch {
      parsed = undefined;
    }
    throw new ApiError(parsed?.error?.message ?? response.statusText, parsed?.error?.code, response.status);
  }
  return response.blob();
}
