function inferApiBase(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }

  if (typeof window !== "undefined") {
    const { protocol, hostname, port } = window.location;
    // In production (behind nginx), API is on the same origin at /api
    // In development, API is on port 8000
    const isDev = hostname === "localhost" && port === "3000";
    if (isDev) {
      return `${protocol}//${hostname}:8000/api`;
    }
    return `${protocol}//${hostname}${port ? `:${port}` : ""}/api`;
  }

  return "http://localhost:8000/api";
}

export const API_BASE = inferApiBase();
export const WS_BASE = API_BASE.replace(/^http/, "ws");
