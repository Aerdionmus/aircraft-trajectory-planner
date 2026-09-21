import type {
  ApiError,
  HealthResponse,
  PlanRequest,
  PlanResponse,
  ScenarioDetail,
  ScenarioSummary
} from "../types/api";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init
    });
  } catch {
    throw new Error(`Backend unavailable at ${API_BASE_URL}`);
  }

  const body = (await response.json().catch(() => null)) as ApiError | T | null;
  if (!response.ok) {
    const message =
      (body as ApiError | null)?.error?.message ??
      `Request failed with HTTP ${response.status}`;
    throw new Error(message);
  }
  if (body === null) throw new Error("Backend returned an empty response");
  return body as T;
}

export const api = {
  getHealth: () => request<HealthResponse>("/api/health"),
  getScenarios: () => request<ScenarioSummary[]>("/api/scenarios"),
  getScenario: (name: string) => request<ScenarioDetail>(`/api/scenarios/${encodeURIComponent(name)}`),
  plan: (planRequest: PlanRequest) =>
    request<PlanResponse>("/api/plan", {
      method: "POST",
      body: JSON.stringify(planRequest)
    })
};

export { API_BASE_URL };
