import CVChart, { HEAT_CV, DEHY_CV, type CVEntry } from "@/components/charts/CVChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import MetricBox from "@/components/ui/MetricBox";
import { BarChart3, Sigma, Award, Layers } from "lucide-react";

function TargetCV({ label, data }: { label: string; data: CVEntry[] }) {
  const best = [...data].sort((a, b) => b.mean - a.mean)[0];
  const avgMean = data.reduce((sum, d) => sum + d.mean, 0) / data.length;
  const avgStd = data.reduce((sum, d) => sum + d.std, 0) / data.length;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricBox
          label="Best CV Score"
          value={`${best.mean.toFixed(2)}%`}
          icon={Award}
          subtitle={best.model}
        />
        <MetricBox
          label="Average CV Mean"
          value={`${avgMean.toFixed(2)}%`}
          icon={BarChart3}
        />
        <MetricBox
          label="Average Std Dev"
          value={`±${avgStd.toFixed(2)}%`}
          icon={Sigma}
        />
        <MetricBox
          label="Folds Used"
          value="5"
          icon={Layers}
        />
      </div>

      <CVChart data={data} title={label} />

      {/* Model ranking + status */}
      <Card className="border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">{label} — CV Rankings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {[...data]
            .sort((a, b) => b.mean - a.mean)
            .map((model, i) => {
              const barWidth = ((model.mean - 88) / 12) * 100;

              return (
                <div key={model.model} className="flex items-center gap-3">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-accent text-xs font-medium text-muted-foreground">
                    {i + 1}
                  </span>
                  <div className="flex-1">
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-xs font-medium">{model.model}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs tabular-nums text-muted-foreground">
                          {model.mean.toFixed(2)}% ±{model.std.toFixed(2)}
                        </span>
                        <Badge
                          variant="outline"
                          className="border-border text-[10px] text-muted-foreground"
                        >
                          {model.status}
                        </Badge>
                      </div>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-accent">
                      <div
                        className="h-full rounded-full bg-muted-foreground/60 transition-all duration-700"
                        style={{ width: `${Math.max(barWidth, 5)}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
        </CardContent>
      </Card>
    </div>
  );
}

export default function CrossValidation() {
  return (
    <Tabs defaultValue="heat" className="space-y-4">
      <TabsList>
        <TabsTrigger value="heat">Heat Stress</TabsTrigger>
        <TabsTrigger value="dehydration">Dehydration</TabsTrigger>
      </TabsList>

      <TabsContent value="heat">
        <TargetCV label="Heat Stress" data={HEAT_CV} />
      </TabsContent>

      <TabsContent value="dehydration">
        <TargetCV label="Dehydration" data={DEHY_CV} />
      </TabsContent>
    </Tabs>
  );
}
