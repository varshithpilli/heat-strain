import type { LucideIcon } from "lucide-react";

interface MetricBoxProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  subtitle?: string;
}

export default function MetricBox({
  label,
  value,
  icon: Icon,
  subtitle,
}: MetricBoxProps) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-card p-4">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-accent">
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="min-w-0">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-sm font-semibold">{value}</p>
        {subtitle && (
          <p className="text-[11px] text-muted-foreground">{subtitle}</p>
        )}
      </div>
    </div>
  );
}
