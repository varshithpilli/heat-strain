import { Card, CardContent } from "@/components/ui/card";
import { Flame, Droplets } from "lucide-react";

type RiskLevel = "Normal" | "Moderate" | "High";

interface RiskCardProps {
  title: string;
  risk: RiskLevel;
  icon: typeof Flame;
}

function RiskCard({ title, risk, icon: Icon }: RiskCardProps) {
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

interface RiskCardsProps {
  heatRisk: RiskLevel;
  dehydrationRisk: RiskLevel;
}

export default function RiskCards({ heatRisk, dehydrationRisk }: RiskCardsProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <RiskCard title="Heat Stress Risk" risk={heatRisk} icon={Flame} />
      <RiskCard title="Dehydration Risk" risk={dehydrationRisk} icon={Droplets} />
    </div>
  );
}
