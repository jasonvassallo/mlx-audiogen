import { useEffect, useRef, useState } from "react";

const HEARTBEAT_INTERVAL = 3000; // 3 seconds
const MAX_FAILURES = 3; // Close after 3 consecutive failures (9 seconds)

/**
 * Polls /api/health and shows a disconnection banner or closes the tab
 * when the server shuts down (e.g., Ctrl+C).
 */
export function useServerHeartbeat() {
  const [connected, setConnected] = useState(true);
  const failCount = useRef(0);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("/api/health", { signal: AbortSignal.timeout(2000) });
        if (res.ok) {
          failCount.current = 0;
          setConnected(true);
        } else {
          failCount.current++;
        }
      } catch {
        failCount.current++;
      }

      if (failCount.current >= MAX_FAILURES) {
        setConnected(false);
      }
    }, HEARTBEAT_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  return connected;
}
