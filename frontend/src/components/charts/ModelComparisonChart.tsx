import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

/* ─── Real performance data from training results ─── */

export interface ModelPerf {
  model: string;
  accuracy: number;
  f1: number;
  roc_auc: number;
  cv_mean: number;
  cv_std: number;
}

export const HEAT_STRESS_DATA: ModelPerf[] = [
  { model: "Custom Neural Net", accuracy: 95.76, f1: 95.73, roc_auc: 99.50, cv_mean: 95.86, cv_std: 0.33 },
  { model: "GAT Neural Network", accuracy: 93.37, f1: 93.36, roc_auc: 98.96, cv_mean: 93.07, cv_std: 0.97 },
  { model: "Random Forest",     accuracy: 99.46, f1: 99.46, roc_auc: 99.99, cv_mean: 99.67, cv_std: 0.06 },
  { model: "Gradient Boosting", accuracy: 99.69, f1: 99.69, roc_auc: 99.99, cv_mean: 99.74, cv_std: 0.05 },
  { model: "SVM (RBF)",         accuracy: 97.21, f1: 97.20, roc_auc: 99.85, cv_mean: 97.09, cv_std: 0.21 },
  { model: "Logistic Regression", accuracy: 94.28, f1: 94.24, roc_auc: 99.29, cv_mean: 93.90, cv_std: 0.38 },
];

export const DEHYDRATION_DATA: ModelPerf[] = [
  { model: "Custom Neural Net", accuracy: 96.33, f1: 96.34, roc_auc: 99.73, cv_mean: 96.54, cv_std: 0.22 },
  { model: "GAT Neural Network", accuracy: 97.05, f1: 97.08, roc_auc: 99.29, cv_mean: 96.53, cv_std: 2.05 },
  { model: "Random Forest",     accuracy: 99.31, f1: 99.31, roc_auc: 99.99, cv_mean: 99.38, cv_std: 0.09 },
  { model: "Gradient Boosting", accuracy: 99.38, f1: 99.38, roc_auc: 99.99, cv_mean: 99.63, cv_std: 0.04 },
  { model: "SVM (RBF)",         accuracy: 97.48, f1: 97.48, roc_auc: 99.89, cv_mean: 97.78, cv_std: 0.20 },
  { model: "Logistic Regression", accuracy: 90.51, f1: 90.53, roc_auc: 98.39, cv_mean: 90.10, cv_std: 0.15 },
];

const tooltipStyle = {
  backgroundColor: "hsl(0 0% 14%)",
  border: "1px solid hsl(0 0% 20%)",
  borderRadius: "6px",
  color: "hsl(0 0% 80%)",
  fontSize: 12,
};

interface ChartProps {
  target: "heat" | "dehydration";
}

export default function ModelComparisonChart({ target }: ChartProps) {
  const data = target === "heat" ? HEAT_STRESS_DATA : DEHYDRATION_DATA;

  const radarData = [
    { metric: "Accuracy", ...Object.fromEntries(data.slice(0, 3).map((d) => [d.model, d.accuracy])) },
    { metric: "F1 Score", ...Object.fromEntries(data.slice(0, 3).map((d) => [d.model, d.f1])) },
    { metric: "ROC AUC", ...Object.fromEntries(data.slice(0, 3).map((d) => [d.model, d.roc_auc])) },
    { metric: "CV Mean", ...Object.fromEntries(data.slice(0, 3).map((d) => [d.model, d.cv_mean])) },
  ];

  const top3 = data.slice(0, 3).map((d) => d.model);

  return (
    <Card className="border-border bg-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">Performance Visualization</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="bar" className="w-full">
          <TabsList className="mb-4">
            <TabsTrigger value="bar">Bar Chart</TabsTrigger>
            <TabsTrigger value="radar">Radar Chart</TabsTrigger>
          </TabsList>

          <TabsContent value="bar">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={data} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 18%)" vertical={false} />
                <XAxis dataKey="model" tick={{ fill: "hsl(0 0% 50%)", fontSize: 10 }} axisLine={{ stroke: "hsl(0 0% 18%)" }} tickLine={false} />
                <YAxis domain={[85, 100]} tick={{ fill: "hsl(0 0% 50%)", fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v}%`} />
                <Tooltip contentStyle={tooltipStyle} formatter={(value) => [`${value}%`]} />
                <Legend wrapperStyle={{ fontSize: 11, color: "hsl(0 0% 50%)" }} />
                <Bar dataKey="accuracy" name="Accuracy" fill="hsl(0 0% 60%)" radius={[3, 3, 0, 0]} maxBarSize={28} />
                <Bar dataKey="f1" name="F1 Score" fill="hsl(0 0% 48%)" radius={[3, 3, 0, 0]} maxBarSize={28} />
                <Bar dataKey="roc_auc" name="ROC AUC" fill="hsl(0 0% 38%)" radius={[3, 3, 0, 0]} maxBarSize={28} />
              </BarChart>
            </ResponsiveContainer>
          </TabsContent>

          <TabsContent value="radar">
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                <PolarGrid stroke="hsl(0 0% 20%)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "hsl(0 0% 50%)", fontSize: 11 }} />
                <PolarRadiusAxis domain={[85, 100]} tick={{ fill: "hsl(0 0% 40%)", fontSize: 10 }} axisLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Radar name={top3[0]} dataKey={top3[0]} stroke="hsl(0 0% 70%)" fill="hsl(0 0% 70%)" fillOpacity={0.1} />
                <Radar name={top3[1]} dataKey={top3[1]} stroke="hsl(0 0% 50%)" fill="hsl(0 0% 50%)" fillOpacity={0.08} />
                <Radar name={top3[2]} dataKey={top3[2]} stroke="hsl(0 0% 35%)" fill="hsl(0 0% 35%)" fillOpacity={0.08} />
                <Legend wrapperStyle={{ fontSize: 11, color: "hsl(0 0% 50%)" }} />
              </RadarChart>
            </ResponsiveContainer>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
