import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Thermometer, Wind, Heart, Droplets, Zap, Activity, Move } from "lucide-react";
import type { SensorState } from "@/hooks/usePrediction";
import type { LucideIcon } from "lucide-react";

interface SensorConfig {
  key: keyof SensorState;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  icon: LucideIcon;
}

const SENSORS: SensorConfig[] = [
  { key: "body_temp", label: "Body Temperature", min: 35, max: 42, step: 0.1, unit: "°C", icon: Thermometer },
  { key: "ambient_temp", label: "Ambient Temperature", min: 20, max: 50, step: 0.5, unit: "°C", icon: Wind },
  { key: "heart_rate", label: "Heart Rate", min: 50, max: 180, step: 1, unit: "bpm", icon: Heart },
  { key: "humidity", label: "Humidity", min: 0, max: 100, step: 1, unit: "%", icon: Droplets },
  { key: "skin_resistance", label: "Skin Resistance", min: 0, max: 500, step: 1, unit: "Ω", icon: Zap },
  { key: "resp_rate", label: "Respiration Rate", min: 10, max: 40, step: 1, unit: "br/min", icon: Activity },
  { key: "movement", label: "Movement Level", min: 0, max: 10, step: 1, unit: "", icon: Move },
];

interface SensorInputsProps {
  sensors: SensorState;
  onUpdate: <K extends keyof SensorState>(key: K, value: SensorState[K]) => void;
  onPredict: () => void;
  onReset: () => void;
  loading: boolean;
}

export default function SensorInputs({
  sensors,
  onUpdate,
  onPredict,
  onReset,
  loading,
}: SensorInputsProps) {
  return (
    <Card className="flex h-full flex-col border-border bg-card">
      <CardHeader className="pb-4">
        <CardTitle className="text-sm font-medium">
          Sensor Inputs
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col justify-between space-y-5">
        {SENSORS.map((sensor) => (
          <div key={sensor.key} className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2 text-xs text-muted-foreground">
                <sensor.icon className="h-3.5 w-3.5" />
                {sensor.label}
              </Label>
              <span className="text-xs font-medium tabular-nums">
                {sensors[sensor.key]}
                {sensor.unit && (
                  <span className="ml-0.5 text-muted-foreground">{sensor.unit}</span>
                )}
              </span>
            </div>
            <Slider
              id={`sensor-${sensor.key}`}
              min={sensor.min}
              max={sensor.max}
              step={sensor.step}
              value={[sensors[sensor.key]]}
              onValueChange={([val]) => onUpdate(sensor.key, val)}
              className="cursor-pointer"
            />
          </div>
        ))}

        <div className="flex gap-3 pt-2">
          <Button
            id="predict-btn"
            onClick={onPredict}
            disabled={loading}
            size="sm"
            className="flex-1"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                Predicting…
              </span>
            ) : (
              "Run Prediction"
            )}
          </Button>
          <Button
            id="reset-btn"
            variant="outline"
            size="sm"
            onClick={onReset}
          >
            Reset
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
