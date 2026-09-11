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
  state: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

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
  kind: string;
  status: JobStatus;
  progress: string[];
  error: null | { code: string; message: string };
  result: null | Record<string, unknown>;
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

export function createSchemaSession(intent: string): Promise<SchemaSession> {
  return requestJson<SchemaSession>("/api/schema/sessions", {
    method: "POST",
    body: JSON.stringify({ intent })
  });
}

export function uploadSchemaFile(sessionId: string, file: File): Promise<UploadSchemaResponse> {
  const body = new FormData();
  body.append("file", file);
  return requestJson<UploadSchemaResponse>(`/api/schema/sessions/${sessionId}/schema-file`, {
    method: "POST",
    body
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
