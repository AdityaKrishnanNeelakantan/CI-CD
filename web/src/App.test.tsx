import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import App from "./App";

const session = {
  id: "session-1",
  workflow: "schema_twin",
  state: {},
  created_at: "2026-09-11T00:00:00Z",
  updated_at: "2026-09-11T00:00:00Z"
};

const uploadData = {
  session,
  summary: {
    name: "schema_twin_minimal",
    table_count: 2,
    column_count: 6,
    relationship_count: 1,
    tables: []
  },
  columns: [
    {
      table: "users",
      column: "user_id",
      type: "int",
      role: "PK",
      nullable: "no",
      references: "-",
      description: "-"
    }
  ],
  llm_text_columns: []
};

const queuedJob = {
  id: "job-1",
  kind: "schema.generate",
  status: "queued",
  progress: ["queued"],
  error: null,
  result: null,
  created_at: "2026-09-11T00:00:00Z",
  updated_at: "2026-09-11T00:00:00Z"
};

const succeededJob = {
  ...queuedJob,
  status: "succeeded",
  progress: ["queued", "running", "packaging download"],
  result: { session_id: "session-1" }
};

function jsonResponse(data: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(data), {
      status,
      headers: { "Content-Type": "application/json" }
    })
  );
}

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  );
}

function installFetchMock(options: { blockDownload?: boolean } = {}) {
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
      return jsonResponse({ data: uploadData });
    }
    if (url === "/api/schema/sessions/session-1/generate" && method === "POST") {
      return jsonResponse({ data: { job: queuedJob } }, 202);
    }
    if (url === "/api/jobs/job-1") {
      return jsonResponse({ data: succeededJob });
    }
    if (url === "/api/schema/sessions/session-1/preview") {
      return jsonResponse({
        data: {
          tables: { users: [{ user_id: 1, status: "active" }] },
          row_counts: { users: 5 }
        }
      });
    }
    if (url === "/api/schema/sessions/session-1/validation") {
      return jsonResponse({
        data: {
          report: { passed: true, export_ready: true },
          highlights: {
            hard_checks_passed: true,
            status: "passed",
            issues: [],
            row_counts: { users: 5 },
            tables: ["users"],
            export_count: 1
          }
        }
      });
    }
    if (url === "/api/schema/sessions/session-1/download" && method === "POST") {
      return jsonResponse({ data: { download_id: "download-1", url: "/api/downloads/download-1" } }, 201);
    }
    if (url === "/api/downloads/download-1") {
      if (options.blockDownload) {
        return jsonResponse(
          { error: { code: "transfer_blocked", message: "transfer blocked: schema hard validation did not pass" } },
          403
        );
      }
      return Promise.resolve(new Response(new Blob(["zip-bytes"], { type: "application/zip" }), { status: 200 }));
    }
    return jsonResponse({ error: { code: "not_found", message: url } }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, calls };
}

async function uploadSchema(user: ReturnType<typeof userEvent.setup>) {
  const file = new File(['{"tables":{}}'], "schema.json", { type: "application/json" });
  await user.upload(screen.getByLabelText("Schema file"), file);
}

async function uploadAndGenerate(user: ReturnType<typeof userEvent.setup>) {
  await uploadSchema(user);
  await screen.findByText("schema_twin_minimal");
  await user.click(screen.getByRole("button", { name: /generate/i }));
  await screen.findByText("succeeded", {}, { timeout: 2000 });
  await screen.findByText("active");
}

describe("Schema Twin React flow", () => {
  beforeEach(() => {
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn(() => "blob:download"),
      revokeObjectURL: vi.fn()
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  test("creates a session and uploads a schema file", async () => {
    const { calls } = installFetchMock();
    const user = userEvent.setup();
    renderApp();

    await uploadSchema(user);

    expect(await screen.findByText("schema_twin_minimal")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Columns" })).toBeInTheDocument();
    expect(calls).toContain("POST /api/schema/sessions");
    expect(calls).toContain("POST /api/schema/sessions/session-1/schema-file");
  });

  test("polls the generation job and renders results", async () => {
    installFetchMock();
    const user = userEvent.setup();
    renderApp();

    await uploadAndGenerate(user);

    expect(screen.getByText("packaging download")).toBeInTheDocument();
    expect(screen.getByText("passed")).toBeInTheDocument();
    expect(screen.getAllByText("user_id").length).toBeGreaterThan(0);
  });

  test("shows the blocked download state from the transfer gate", async () => {
    installFetchMock({ blockDownload: true });
    const user = userEvent.setup();
    renderApp();

    await uploadAndGenerate(user);
    const validationPanel = screen.getByRole("button", { name: /download zip/i }).closest(".panel") as HTMLElement;
    await user.click(within(validationPanel).getByRole("button", { name: /download zip/i }));

    expect(await screen.findByText("transfer blocked: schema hard validation did not pass")).toBeInTheDocument();
    expect(within(validationPanel).getByRole("button", { name: /download zip/i })).toBeDisabled();
  });
});
