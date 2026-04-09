import { usePrediction } from "@/hooks/usePrediction";
import SensorInputs from "@/components/prediction/SensorInputs";
import ProbabilityChart from "@/components/prediction/ProbabilityChart";
import { Card, CardContent } from "@/components/ui/card";
import { Sparkles, AlertCircle, Flame, Droplets} from "lucide-react";

interface DashboardProps {
  selectedModel: string;
}

// const MODEL_DISPLAY: Record<string, string> = {
//   custom_nn: "Custom Neural Network",
//   random_forest: "Random Forest",
//   gradient_boost: "Gradient Boosting",
//   svm: "SVM",
//   logistic_reg: "Logistic Regression",
// };

type RiskLevel = "Normal" | "Moderate" | "High";

function RiskCard({ title, risk, icon: Icon }: { title: string; risk: RiskLevel; icon: typeof Flame }) {
  return (
    <Card className="border-border bg-card">
      <CardContent className="flex items-center gap-4 p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-accent">
          <Icon className="h-5 w-5 text-muted-foreground" />
        </div>
        <div className="flex-1">
          <p className="text-xs text-muted-foreground">{title}</p>
          <span className="text-base font-semibold">{risk}</span>
        </div>
      </CardContent>
    </Card>
  );
}

export default function Dashboard({ selectedModel }: DashboardProps) {
  const {
    sensors,
    result,
    loading,
    error,
    updateSensor,
    runPrediction,
    resetSensors,
  } = usePrediction(selectedModel);

  const heat = result?.results?.heat_stress_label;
  const dehydration = result?.results?.dehydration_label;
  const hasResult = heat && dehydration;

  // Compute confidence from predicted class probabilities
  // const confidence = hasResult
  //   ? (heat.probabilities[heat.class] + dehydration.probabilities[dehydration.class]) / 2
  //   : 0;

  return (
    <div className="grid min-h-[calc(100vh-7rem)] grid-cols-1 gap-4 lg:grid-cols-[340px_1fr] lg:items-stretch">
      {/* Left — Sensor control panel (narrow) */}
      <SensorInputs
        sensors={sensors}
        onUpdate={updateSensor}
        onPredict={runPrediction}
        onReset={resetSensors}
        loading={loading}
      />

      {/* Right — Results area */}
      <div className="flex flex-col gap-4">
        {error && (
          <Card className="border-border bg-card">
            <CardContent className="flex items-center gap-3 p-4">
              <AlertCircle className="h-4 w-4 shrink-0 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">{error}</p>
            </CardContent>
          </Card>
        )}

        {hasResult ? (
          <>
            {/* Risk cards */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <RiskCard title="Heat Stress Risk" risk={heat.label} icon={Flame} />
              <RiskCard title="Dehydration Risk" risk={dehydration.label} icon={Droplets} />
            </div>

            {/* Probability chart — gets full width for readability */}
            <ProbabilityChart
              heatProbabilities={heat.probabilities}
              dehydrationProbabilities={dehydration.probabilities}
            />

            {/* Confidence summary bar */}
            {/* <Card className="border-border bg-card">
              <CardContent className="flex flex-wrap items-center gap-x-6 gap-y-3 p-4">
                <div className="flex items-center gap-2.5">
                  <div className="flex h-8 w-8 items-center justify-center rounded-md bg-accent">
                    <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Confidence</p>
                    <p className="text-sm font-semibold tabular-nums">{(confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>
                <div className="h-8 w-px bg-border" />
                <div className="flex items-center gap-2.5">
                  <div className="flex h-8 w-8 items-center justify-center rounded-md bg-accent">
                    <Brain className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Active Model</p>
                    <p className="text-sm font-semibold">{MODEL_DISPLAY[selectedModel] ?? selectedModel}</p>
                  </div>
                </div>
                <div className="h-8 w-px bg-border hidden sm:block" />
                <div className="flex items-center gap-2.5">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Quality</p>
                  <span className="rounded-md border border-border px-2 py-0.5 text-xs font-medium text-muted-foreground">
                    {confidence >= 0.85 ? "High" : confidence >= 0.65 ? "Medium" : "Low"}
                  </span>
                </div>
              </CardContent>
            </Card> */}
          </>
        ) : !error ? (
          <div className="flex items-center justify-center py-20">
            <Card className="max-w-sm border-border bg-card">
              <CardContent className="flex flex-col items-center gap-3 p-8 text-center">
                <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-accent">
                  <Sparkles className="h-5 w-5 text-muted-foreground" />
                </div>
                <h3 className="text-sm font-semibold">Ready to Predict</h3>
                <p className="text-xs leading-relaxed text-muted-foreground">
                  Adjust the sensor readings and click{" "}
                  <span className="font-medium text-foreground">Run Prediction</span> to
                  analyze risk levels.
                </p>
              </CardContent>
            </Card>
          </div>
        ) : null}
      </div>
    </div>
  );
}
