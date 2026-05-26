import { useEffect, useRef, useState } from "react";
import { getWsUrl } from "../services/api";
import type { RealtimeSocketResponse } from "../types/api";

interface RealtimeState {
  connected: boolean;
  lastResponse: RealtimeSocketResponse | null;
  error: string;
}

export function useRealtimeInference() {
  const socketRef = useRef<WebSocket | null>(null);
  const [state, setState] = useState<RealtimeState>({
    connected: false,
    lastResponse: null,
    error: "",
  });

  useEffect(() => {
    return () => {
      socketRef.current?.close();
    };
  }, []);

  function connect() {
    socketRef.current?.close();
    const socket = new WebSocket(getWsUrl());
    socketRef.current = socket;

    socket.onopen = () => {
      setState((prev) => ({ ...prev, connected: true, error: "" }));
    };

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as RealtimeSocketResponse;
      setState((prev) => ({ ...prev, lastResponse: payload }));
    };

    socket.onerror = () => {
      setState((prev) => ({
        ...prev,
        error: "WebSocket 连接失败",
      }));
    };

    socket.onclose = () => {
      setState((prev) => ({ ...prev, connected: false }));
    };
  }

  function disconnect() {
    socketRef.current?.close();
  }

  function sendFrame(frameId: number, imageBase64: string) {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    socketRef.current.send(
      JSON.stringify({
        frame_id: frameId,
        timestamp: new Date().toISOString(),
        image: imageBase64,
      })
    );
  }

  return {
    ...state,
    connect,
    disconnect,
    sendFrame,
  };
}
