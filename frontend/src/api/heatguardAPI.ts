import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 10000,
  headers: { "Content-Type": "application/json" },
});

/** Matches backend's SensorInput pydantic model */
export interface SensorData {
  body_temp: number;
  ambient_temp: number;
  humidity: number;
  heart_rate: number;
  skin_resistance: number;
  resp_rate: number;
  movement: number;
  model: string;
}

/** Backend response for a single target (heat_stress_label / dehydration_label) */
export interface TargetResult {
  class: number;
  label: "Normal" | "Moderate" | "High";
  probabilities: number[];
}

/** Backend /predict response */
export interface PredictionResponse {
  overall_risk: "Normal" | "Moderate" | "High";
  results: {
    heat_stress_label: TargetResult;
    dehydration_label: TargetResult;
  };
  message?: string;
}

export async function predict(data: SensorData): Promise<PredictionResponse> {
  const res = await api.post<PredictionResponse>("/predict", data);
  return res.data;
}

export async function checkHealth(): Promise<boolean> {
  try {
    await api.get("/health");
    return true;
  } catch {
    return false;
  }
}

export default api;
