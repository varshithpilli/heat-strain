import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Gauge from "@/components/ui/Gauge";
import MetricBox from "@/components/ui/MetricBox";
import { Brain, TrendingUp, Shield } from "lucide-react";

interface ConfidenceMetricsProps {
  heatProbabilities: number[];
  heatClass: number;
  dehydrationProbabilities: number[];
  dehydrationClass: number;
  modelUsed: string;
}

export default function ConfidenceMetrics({
  heatProbabilities,
  heatClass,
  dehydrationProbabilities,
  dehydrationClass,
  modelUsed,
}: ConfidenceMetricsProps) {
  // Confidence = average of the predicted class probabilities
  const heatConf = heatProbabilities[heatClass] ?? 0;
  const dehydConf = dehydrationProbabilities[dehydrationClass] ?? 0;
  const confidence = (heatConf + dehydConf) / 2;

  return (
    <Card className="border-border bg-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">
          Model Confidence
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-start">
          <Gauge value={confidence} label="Confidence" />
          <div className="flex flex-1 flex-col gap-3">
            {/* <MetricBox
              label="Active Model"
              value={modelUsed}
              icon={Brain}
            /> */}
            <MetricBox
              label="Prediction Quality"
              value={confidence >= 0.85 ? "High" : confidence >= 0.65 ? "Medium" : "Low"}
              icon={TrendingUp}
              // subtitle={`${(confidence * 100).toFixed(1)}% confidence score`}
            />
            <MetricBox
              label="Safety Margin"
              value={confidence >= 0.8 ? "Reliable" : "Review Needed"}
              icon={Shield}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
