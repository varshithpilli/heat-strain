import ModelComparisonChart, { HEAT_STRESS_DATA, DEHYDRATION_DATA } from "@/components/charts/ModelComparisonChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import MetricBox from "@/components/ui/MetricBox";
import { Trophy, Target, TrendingUp, Award } from "lucide-react";
import type { ModelPerf } from "@/components/charts/ModelComparisonChart";

function TargetSection({ label, data }: { label: string; data: ModelPerf[] }) {
  const best = [...data].sort((a, b) => b.accuracy - a.accuracy)[0];

  return (
    <div className="space-y-4">
      {/* Summary metrics */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricBox
          label="Best Model"
          value={best.model}
          icon={Trophy}
          subtitle={`${best.accuracy}% accuracy`}
        />
        <MetricBox
          label="Highest F1 Score"
          value={`${best.f1}%`}
          icon={Target}
        />
        <MetricBox
          label="Best ROC AUC"
          value={`${best.roc_auc}%`}
          icon={TrendingUp}
        />
        <MetricBox
          label="Models Evaluated"
          value={data.length}
          icon={Award}
        />
      </div>

      {/* Performance table */}
      <Card className="border-border bg-card">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">
            {label} — Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-border hover:bg-transparent">
                  <TableHead className="text-xs text-muted-foreground">Model</TableHead>
                  <TableHead className="text-right text-xs text-muted-foreground">Accuracy</TableHead>
                  <TableHead className="text-right text-xs text-muted-foreground">F1</TableHead>
                  <TableHead className="text-right text-xs text-muted-foreground">ROC AUC</TableHead>
                  <TableHead className="text-right text-xs text-muted-foreground">CV Mean</TableHead>
                  <TableHead className="text-right text-xs text-muted-foreground">CV Std</TableHead>
                  <TableHead className="text-right text-xs text-muted-foreground">Rank</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[...data]
                  .sort((a, b) => b.accuracy - a.accuracy)
                  .map((row, i) => (
                    <TableRow
                      key={row.model}
                      className="border-border transition-colors hover:bg-accent/50"
                    >
                      <TableCell className="text-sm font-medium">{row.model}</TableCell>
                      <TableCell className="text-right text-sm tabular-nums">{row.accuracy.toFixed(2)}%</TableCell>
                      <TableCell className="text-right text-sm tabular-nums">{row.f1.toFixed(2)}%</TableCell>
                      <TableCell className="text-right text-sm tabular-nums">{row.roc_auc.toFixed(2)}%</TableCell>
                      <TableCell className="text-right text-sm tabular-nums">{row.cv_mean.toFixed(2)}%</TableCell>
                      <TableCell className="text-right text-sm tabular-nums">±{row.cv_std.toFixed(2)}%</TableCell>
                      <TableCell className="text-right">
                        <Badge variant="outline" className="border-border text-xs text-muted-foreground">
                          #{i + 1}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Charts */}
      <ModelComparisonChart target={label === "Heat Stress" ? "heat" : "dehydration"} />
    </div>
  );
}

export default function ModelComparison() {
  return (
    <Tabs defaultValue="heat" className="space-y-4">
      <TabsList>
        <TabsTrigger value="heat">Heat Stress</TabsTrigger>
        <TabsTrigger value="dehydration">Dehydration</TabsTrigger>
      </TabsList>

      <TabsContent value="heat">
        <TargetSection label="Heat Stress" data={HEAT_STRESS_DATA} />
      </TabsContent>

      <TabsContent value="dehydration">
        <TargetSection label="Dehydration" data={DEHYDRATION_DATA} />
      </TabsContent>
    </Tabs>
  );
}
