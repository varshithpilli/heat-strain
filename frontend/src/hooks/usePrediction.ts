import { useState, useCallback } from "react";
import { predict, type SensorData, type PredictionResponse } from "@/api/heatguardAPI";

export interface SensorState {
  body_temp: number;
  ambient_temp: number;
  humidity: number;
  heart_rate: number;
  skin_resistance: number;
  resp_rate: number;
  movement: number;
}

const DEFAULT_SENSORS: SensorState = {
  body_temp: 37.0,
  ambient_temp: 30.0,
  humidity: 50,
  heart_rate: 75,
  skin_resistance: 50,
  resp_rate: 18,
  movement: 3,
};

export function usePrediction(selectedModel: string) {
  const [sensors, setSensors] = useState<SensorState>(DEFAULT_SENSORS);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateSensor = useCallback(
    <K extends keyof SensorState>(key: K, value: SensorState[K]) => {
      setSensors((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const runPrediction = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data: SensorData = {
        ...sensors,
        model: selectedModel,
      };
      const res = await predict(data);
      setResult(res);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to connect to API";
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [sensors, selectedModel]);

  const resetSensors = useCallback(() => {
    setSensors(DEFAULT_SENSORS);
    setResult(null);
    setError(null);
  }, []);

  return {
    sensors,
    result,
    loading,
    error,
    updateSensor,
    runPrediction,
    resetSensors,
  };
}
