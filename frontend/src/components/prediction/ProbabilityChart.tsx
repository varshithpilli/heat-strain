import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ProbabilityChartProps {
  heatProbabilities: number[];
  dehydrationProbabilities: number[];
}

const LABELS = ["Normal", "Moderate", "High"];

const tooltipStyle = {
  backgroundColor: "hsl(0 0% 14%)",
  border: "1px solid hsl(0 0% 20%)",
  borderRadius: "6px",
  color: "hsl(0 0% 80%)",
  fontSize: 12,
};

export default function ProbabilityChart({
  heatProbabilities,
  dehydrationProbabilities,
}: ProbabilityChartProps) {
  const data = LABELS.map((label, i) => ({
    label,
    "Heat Stress": +(heatProbabilities[i] * 100).toFixed(1),
    Dehydration: +(dehydrationProbabilities[i] * 100).toFixed(1),
  }));

  return (
    <Card className="flex flex-1 flex-col border-border bg-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">
          Probability Breakdown
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col">
        <ResponsiveContainer width="100%" className="min-h-[240px] flex-1">
          <BarChart data={data} barGap={4} barCategoryGap="25%">
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(0 0% 18%)"
              vertical={false}
            />
            <XAxis
              dataKey="label"
              tick={{ fill: "hsl(0 0% 50%)", fontSize: 11 }}
              axisLine={{ stroke: "hsl(0 0% 18%)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "hsl(0 0% 50%)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              domain={[0, 100]}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value: number) => [`${value}%`]}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, color: "hsl(0 0% 50%)" }}
            />
            <Bar
              dataKey="Heat Stress"
              fill="hsl(0 0% 55%)"
              radius={[3, 3, 0, 0]}
              maxBarSize={36}
            />
            <Bar
              dataKey="Dehydration"
              fill="hsl(0 0% 38%)"
              radius={[3, 3, 0, 0]}
              maxBarSize={36}
            />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
