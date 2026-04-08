import { motion } from "framer-motion";

interface GaugeProps {
  value: number; // 0–1
  size?: number;
  strokeWidth?: number;
  label?: string;
}

export default function Gauge({
  value,
  size = 110,
  strokeWidth = 8,
  label,
}: GaugeProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clampedValue = Math.max(0, Math.min(1, value));

  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="-rotate-90"
      >
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-accent"
        />

        {/* Fill */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{
            strokeDashoffset: circumference * (1 - clampedValue),
          }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-foreground"
        />

        {/* Center text */}
        <text
          x={size / 2}
          y={size / 2}
          textAnchor="middle"
          dominantBaseline="central"
          className="rotate-90 fill-foreground text-lg font-semibold"
          style={{ transformOrigin: "center" }}
        >
          {Math.round(clampedValue * 100)}%
        </text>
      </svg>
      {label && (
        <span className="text-xs text-muted-foreground">
          {label}
        </span>
      )}
    </div>
  );
}
