import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ErrorBar,
  Line,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface CVEntry {
  model: string;
  mean: number;
  std: number;
  status: string;
}

export const HEAT_CV: CVEntry[] = [
  { model: "Custom Neural Net", mean: 95.86, std: 0.33, status: "Excellent" },
  { model: "Random Forest", mean: 99.67, std: 0.06, status: "Excellent" },
  { model: "Gradient Boosting", mean: 99.74, std: 0.05, status: "Excellent" },
  { model: "SVM (RBF)", mean: 97.09, std: 0.21, status: "Excellent" },
  { model: "Logistic Regression", mean: 93.90, std: 0.38, status: "Good" },
];

export const DEHY_CV: CVEntry[] = [
  { model: "Custom Neural Net", mean: 96.54, std: 0.22, status: "Excellent" },
  { model: "Random Forest", mean: 99.38, std: 0.09, status: "Excellent" },
  { model: "Gradient Boosting", mean: 99.63, std: 0.04, status: "Excellent" },
  { model: "SVM (RBF)", mean: 97.78, std: 0.20, status: "Excellent" },
  { model: "Logistic Regression", mean: 90.10, std: 0.15, status: "Good" },
];

const tooltipStyle = {
  backgroundColor: "hsl(0 0% 14%)",
  border: "1px solid hsl(0 0% 20%)",
  borderRadius: "6px",
  color: "hsl(0 0% 80%)",
  fontSize: 12,
};

interface CVChartProps {
  data: CVEntry[];
  title: string;
}

export default function CVChart({ data, title }: CVChartProps) {
  const dataWithError = data.map((d) => ({
    ...d,
    errorY: [d.std, d.std],
  }));

  return (
    <Card className="border-border bg-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">
          {title} — 5-Fold Cross-Validation
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={dataWithError} barCategoryGap="30%">
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(0 0% 18%)"
              vertical={false}
            />
            <XAxis
              dataKey="model"
              tick={{ fill: "hsl(0 0% 50%)", fontSize: 10 }}
              axisLine={{ stroke: "hsl(0 0% 18%)" }}
              tickLine={false}
            />
            <YAxis
              domain={[88, 100]}
              tick={{ fill: "hsl(0 0% 50%)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value: number, name: string) => [
                `${value.toFixed(2)}%`,
                name === "mean" ? "CV Mean" : name,
              ]}
            />
            <Bar
              dataKey="mean"
              fill="hsl(0 0% 45%)"
              radius={[3, 3, 0, 0]}
              maxBarSize={44}
              fillOpacity={0.8}
            >
              <ErrorBar
                dataKey="errorY"
                width={6}
                stroke="hsl(0 0% 60%)"
                strokeWidth={1.5}
              />
            </Bar>
            <Line
              type="monotone"
              dataKey="mean"
              stroke="hsl(0 0% 65%)"
              strokeWidth={1.5}
              dot={{ fill: "hsl(0 0% 65%)", r: 3, strokeWidth: 0 }}
              activeDot={{ r: 4, fill: "hsl(0 0% 70%)" }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
