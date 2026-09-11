import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import App from "./App";

const session = {
  id: "session-1",
  workflow: "schema_twin",
  workflow_type: "schema_twin",
  state: {},
  created_at: "2026-09-11T00:00:00Z",
  updated_at: "2026-09-11T00:00:00Z"
};

const queuedJob = {
  id: "job-1",
  job_id: "job-1",
  session_id: "session-1",
  workflow_type: "schema_twin",
  kind: "schema.generate",
  status: "queued",
  stage: "queued",
  percent: 0,
  message: "queued",
  progress: ["queued"],
  error: null,
  result_id: null,
  result: null,
  created_at: "2026-09-11T00:00:00Z",
  updated_at: "2026-09-11T00:00:00Z"
};

const succeededJob = {
  ...queuedJob,
  status: "succeeded",
  stage: "preparing_files",
  percent: 100,
  message: "Preparing files",
  result_id: "result-1",
  result: { result_id: "result-1" }
};

const resultBundle = {
  id: "result-1",
  result_id: "result-1",
  session_id: "session-1",
  workflow_type: "schema_twin",
  status: "available",
  preview: {
    tables: {
      users: [{ user_id: 1, status: "active" }]
    }
  },
  quality_report: { status: "passed", passed: true },
  summary: { table_count: 1, row_count: 5 },
  artifacts: [
    {
      id: "artifact-1",
      artifact_id: "artifact-1",
      name: "users.csv",
      filename: "users.csv",
      path: "/tmp/users.csv",
      media_type: "text/csv",
      content_type: "text/csv",
      size: 128,
      size_bytes: 128,
      role: "data",
      kind: "data",
      downloadable: true,
      download_url: "/api/results/result-1/artifacts/artifact-1/download",
      metadata: {}
    },
    {
      id: "artifact-2",
      artifact_id: "artifact-2",
      name: "lineage.json",
      filename: "lineage.json",
      path: "/tmp/missing.json",
      media_type: "application/json",
      content_type: "application/json",
      size: null,
      size_bytes: null,
      role: "report",
      kind: "report",
      downloadable: false,
      download_url: null,
      metadata: {}
    }
  ],
  metadata: {},
  created_at: "2026-09-11T00:00:00Z",
  updated_at: "2026-09-11T00:00:00Z"
};

const projectDetail = {
  project_id: "project-1",
  id: "project-1",
  name: "Schema Twin Result",
  workflow_type: "schema",
  created_at: "2026-09-11T00:00:00Z",
  updated_at: "2026-09-11T00:00:00Z",
  status: "completed",
  latest_run_id: "run-1",
  latest_result_id: "result-1",
  metadata: {},
  summary: { row_counts: { users: 5 } },
  quality_report: { status: "passed", passed: true },
  artifacts: resultBundle.artifacts,
  runs: [
    {
      id: "run-1",
      run_id: "run-1",
      project_id: "project-1",
      workflow_type: "schema",
      status: "completed",
      created_at: "2026-09-11T00:00:00Z",
      updated_at: "2026-09-11T00:00:00Z",
      completed_at: "2026-09-11T00:00:00Z",
      result_id: "result-1",
      job_id: "job-1",
      validation_status: "passed",
      validation_passed: true,
      transfer_status: "not_attempted",
      transfer_allowed: null,
      transfer_attempted_at: null,
      metadata: {}
    }
  ]
};

const runDetail = {
  ...projectDetail.runs[0],
  summary: { row_counts: { users: 5 } },
  quality_report: { status: "passed", passed: true },
  artifacts: resultBundle.artifacts,
  input: { schema_file: { filename: "schema.json", size: 100 } },
  config: { row_count: 5, seed: 7 }
};

const templates = [
  {
    template_id: "ecommerce",
    name: "E-commerce Platform",
    description: "Complete e-commerce dataset with products, orders, and reviews",
    workflow_type: "schema_twin",
    category: "ecommerce",
    supported_formats: ["json", "csv", "parquet"],
    status: "available",
    can_generate: true,
    fields: [{ table: "customers", column: "id", type: "int" }],
    schema_preview: { table_count: 1 }
  },
  {
    template_id: "support-transcript",
    name: "Support Transcript",
    description: "Interaction Twin transcript shape for support chats and logs.",
    workflow_type: "interaction_twin",
    category: "interaction",
    supported_formats: ["txt", "log"],
    status: "coming_soon",
    can_generate: false,
    fields: []
  }
];

function jsonResponse(data: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(data), {
      status,
      headers: { "Content-Type": "application/json" }
    })
  );
}

function renderApp(route = "/") {
  window.history.pushState({}, "", route);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  );
}

function installFetchMock() {
  const calls: string[] = [];
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    calls.push(`${method} ${url}`);

    if (url === "/api/settings") {
      return jsonResponse({
        data: {
          generation_mode: "schema_driven",
          default_record_count: 5,
          privacy_level: "standard",
          default_output_format: "csv"
        }
      });
    }
    if (url === "/api/schema/sessions" && method === "POST") {
      return jsonResponse({ data: session }, 201);
    }
    if (url === "/api/schema/sessions/session-1/schema-file" && method === "POST") {
      return jsonResponse({
        data: {
          session,
          summary: {
            name: "schema_twin_minimal",
            table_count: 1,
            column_count: 2,
            relationship_count: 0,
            tables: []
          },
          columns: [{ table: "users", column: "user_id", type: "int", role: "PK", nullable: "no", references: "-", description: "-" }],
          llm_text_columns: []
        }
      });
    }
    if (url === "/api/schema/sessions/session-1/generate" && method === "POST") {
      return jsonResponse({ data: { job: queuedJob } }, 202);
    }
    if (url === "/api/jobs/job-1") {
      return jsonResponse({ data: succeededJob });
    }
    if (url === "/api/results/result-1") {
      return jsonResponse({ data: resultBundle });
    }
    if (url === "/api/results/result-1/artifacts/artifact-1/download") {
      return Promise.resolve(new Response(new Blob(["user_id,status\n1,active"], { type: "text/csv" }), { status: 200 }));
    }
    if (url === "/api/results/result-1/save" && method === "POST") {
      return jsonResponse({
        data: {
          project_id: "project-1",
          run_id: "run-1",
          saved: true,
          project: {
            id: "project-1",
            name: "Schema Twin Result",
            workflow_type: "schema",
            created_at: "2026-09-11T00:00:00Z",
            status: "completed"
          }
        }
      }, 201);
    }
    if (url === "/api/projects") {
      return jsonResponse({
        data: {
          projects: [
            {
              id: "project-1",
              name: "Schema Twin Result",
              workflow_type: "schema",
              workflow_label: "Schema",
              records: "5",
              created_on: "Sep 11, 2026 12:00 AM",
              status: "completed",
              validation: "Passed",
              transfer: "Not attempted",
              latest_run_id: "run-1",
              latest_result_id: "result-1"
            }
          ]
        }
      });
    }
    if (url === "/api/projects/project-1") {
      return jsonResponse({ data: { project: projectDetail } });
    }
    if (url === "/api/projects/project-1/runs") {
      return jsonResponse({ data: { runs: projectDetail.runs } });
    }
    if (url === "/api/projects/project-1/runs/run-1") {
      return jsonResponse({ data: { run: runDetail } });
    }
    if (url === "/api/templates") {
      return jsonResponse({ data: { templates } });
    }
    if (url === "/api/templates/ecommerce") {
      return jsonResponse({ data: { template: templates[0] } });
    }
    if (url === "/api/schema/sessions/session-1/template" && method === "POST") {
      return jsonResponse({
        data: {
          session,
          summary: {
            name: "E-commerce Platform",
            table_count: 1,
            column_count: 1,
            relationship_count: 0,
            tables: [{ table: "customers", rows: 100 }]
          },
          columns: [{ table: "customers", column: "id", type: "int", role: "PK", nullable: "no", references: "-", description: "-" }],
          llm_text_columns: []
        }
      });
    }
    if (url === "/api/settings" && method === "PUT") {
      return jsonResponse({
        data: {
          generation_mode: "hybrid",
          default_record_count: 25,
          privacy_level: "strict",
          default_output_format: "parquet"
        }
      });
    }
    return jsonResponse({ error: { code: "not_found", message: url } }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls };
}

describe("Synthetic Data Twin routed app", () => {
  beforeEach(() => {
    installFetchMock();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:download") });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.history.pushState({}, "", "/");
  });

  test("renders the shell navigation and home workflow cards", () => {
    renderApp();

    expect(screen.getByRole("heading", { name: "Synthetic Data Twin" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Home/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /My Projects/i })).toBeInTheDocument();
    expect(screen.getByText("Air-Gapped Mode")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Schema Twin/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Database Twin/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Document Twin/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Customer Interaction Twin/i })).toBeInTheDocument();
  });

  test("polls and renders a shared progress job", async () => {
    renderApp("/progress/job-1");

    expect(await screen.findByText("succeeded")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /View Results/i })).toHaveAttribute("href", "/results/result-1");
  });

  test("renders downloadable and metadata-only result artifacts", async () => {
    const user = userEvent.setup();
    renderApp("/results/result-1");

    expect(await screen.findByText("Generated Results")).toBeInTheDocument();
    expect(screen.getByText("user_id")).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "files" }));
    expect(screen.getByText("users.csv")).toBeInTheDocument();
    expect(screen.getByText("lineage.json")).toBeInTheDocument();
    expect(screen.getByText("Metadata only")).toBeInTheDocument();
    await user.click(screen.getByTitle("Download"));
    expect(URL.createObjectURL).toHaveBeenCalled();
  });

  test("saves a result bundle to My Projects", async () => {
    const user = userEvent.setup();
    renderApp("/results/result-1");

    await screen.findByText("Generated Results");
    await user.click(screen.getByRole("button", { name: /Save to My Projects/i }));

    expect(await screen.findByText(/Saved to My Projects/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /View projects/i })).toHaveAttribute("href", "/projects");
  });

  test("renders real project data with a view action", async () => {
    renderApp("/projects");

    expect(await screen.findByText("Schema Twin Result")).toBeInTheDocument();
    expect(screen.getByText("Schema")).toBeInTheDocument();
    expect(screen.getByText("Passed")).toBeInTheDocument();
    expect(screen.getByTitle("View")).toHaveAttribute("href", "/projects/project-1");
  });

  test("renders project detail with runs and artifacts", async () => {
    renderApp("/projects/project-1");

    expect(await screen.findByRole("heading", { name: "Schema Twin Result" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Latest Result/i })).toHaveAttribute("href", "/results/result-1");
    expect(screen.getByText("run-1")).toBeInTheDocument();
    expect(screen.getByTitle("View run")).toHaveAttribute("href", "/projects/project-1/runs/run-1");
    expect(screen.getByText("users.csv")).toBeInTheDocument();
  });

  test("renders run detail with result and config links", async () => {
    renderApp("/projects/project-1/runs/run-1");

    expect(await screen.findByRole("heading", { name: "run-1" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Result/i })).toHaveAttribute("href", "/results/result-1");
    expect(screen.getByText(/schema.json/i)).toBeInTheDocument();
    expect(screen.getByText(/row_count/i)).toBeInTheDocument();
  });

  test("renders templates from the API", async () => {
    renderApp("/templates");

    expect(await screen.findByText("E-commerce Platform")).toBeInTheDocument();
    expect(screen.getByText("Support Transcript")).toBeInTheDocument();
    expect(screen.getByText("Coming soon")).toBeInTheDocument();
  });

  test("renders template detail and starts template-backed schema flow", async () => {
    const user = userEvent.setup();
    renderApp("/templates/ecommerce");

    expect(await screen.findByRole("heading", { name: "E-commerce Platform" })).toBeInTheDocument();
    expect(screen.getByText("customers")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Use Template/i }));

    await waitFor(() => expect(window.location.pathname).toBe("/schema"));
    expect(await screen.findByText("E-commerce Platform")).toBeInTheDocument();
  });

  test("fetches and saves settings", async () => {
    const user = userEvent.setup();
    renderApp("/settings");

    const records = await screen.findByLabelText("Default records");
    await user.clear(records);
    await user.type(records, "25");
    await user.selectOptions(screen.getByLabelText("Generation mode"), "hybrid");
    await user.selectOptions(screen.getByLabelText("Privacy level"), "strict");
    await user.selectOptions(screen.getByLabelText("Default output format"), "parquet");
    await user.click(screen.getByRole("button", { name: /Save Settings/i }));

    expect(await screen.findByText("Settings saved")).toBeInTheDocument();
  });

  test("blocks workflow generation until a required upload exists", () => {
    renderApp("/database");

    expect(screen.getByRole("button", { name: /generate/i })).toBeDisabled();
    expect(screen.getByText(/SQLite upload is supported/i)).toBeInTheDocument();
  });

  test("starts schema generation and navigates to the progress route", async () => {
    const user = userEvent.setup();
    renderApp("/schema");

    const file = new File(['{"tables":{}}'], "schema.json", { type: "application/json" });
    await user.upload(await screen.findByLabelText("Schema file"), file);
    expect(await screen.findByText("schema_twin_minimal")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /generate/i }));

    await waitFor(() => expect(window.location.pathname).toBe("/progress/job-1"));
    expect(await screen.findByText("View Results")).toBeInTheDocument();
  });
});
