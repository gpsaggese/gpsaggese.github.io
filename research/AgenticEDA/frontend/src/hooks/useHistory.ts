"use client";
import { useState, useCallback } from "react";
import { API_BASE } from "@/lib/backend";

interface Version {
  version_id: number;
  trigger: string;
  timestamp: string;
  has_notebook: boolean;
  has_story: boolean;
}

export function useHistory(sessionId: string) {
  const [versions, setVersions] = useState<Version[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchVersions = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/history/${sessionId}`);
      if (res.ok) {
        const data = await res.json();
        setVersions(data.versions || []);
      }
    } catch {
      // History endpoint may not be available — silently ignore
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  const restoreVersion = useCallback(
    async (versionId: number) => {
      const res = await fetch(
        `${API_BASE}/history/${sessionId}/restore/${versionId}`,
        { method: "POST" }
      );
      if (res.ok) {
        await fetchVersions();
        return res.json();
      }
      throw new Error("Restore failed");
    },
    [sessionId, fetchVersions]
  );

  return { versions, loading, fetchVersions, restoreVersion };
}
