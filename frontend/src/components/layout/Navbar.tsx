import { useLocation } from "react-router-dom";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Brain, Menu } from "lucide-react";

/** Maps backend model key → display name */
const MODELS: { key: string; label: string }[] = [
  { key: "custom_nn", label: "Custom Neural Network" },
  { key: "gat", label: "GAT Neural Network" },
  { key: "random_forest", label: "Random Forest" },
  { key: "gradient_boost", label: "Gradient Boosting" },
  { key: "svm", label: "SVM" },
  { key: "logistic_reg", label: "Logistic Regression" },
];

const PAGE_TITLES: Record<string, string> = {
  "/": "Live Prediction",
  "/comparison": "Model Comparison",
  "/cv": "Cross Validation",
  "/about": "About",
};

interface NavbarProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
  onMenuToggle?: () => void;
}

export default function Navbar({ selectedModel, onModelChange, onMenuToggle }: NavbarProps) {
  const location = useLocation();
  const title = PAGE_TITLES[location.pathname] ?? "HeatGuard AI";
  const activeLabel = MODELS.find((m) => m.key === selectedModel)?.label ?? selectedModel;

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-border bg-background/90 px-4 backdrop-blur-sm md:px-6">
      <div className="flex items-center gap-3">
        {onMenuToggle && (
          <button
            onClick={onMenuToggle}
            className="cursor-pointer rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground md:hidden"
          >
            <Menu className="h-5 w-5" />
          </button>
        )}
        <h2 className="font-heading text-base font-medium text-foreground">{title}</h2>
      </div>

      {/* Model selector */}
      <div className="flex items-center gap-3 rounded-xl border border-border bg-card px-4 py-2.5">
        <Brain className="h-4.5 w-4.5 text-muted-foreground" />
        <div className="hidden flex-col sm:flex">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Active Model</span>
          <Select value={selectedModel} onValueChange={onModelChange}>
            <SelectTrigger
              id="model-selector"
              className="h-auto w-auto min-w-[160px] cursor-pointer border-0 bg-transparent p-0 text-sm font-medium text-foreground shadow-none focus:ring-0"
            >
              <SelectValue>{activeLabel}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {MODELS.map((m) => (
                <SelectItem key={m.key} value={m.key} className="cursor-pointer p-3">
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="sm:hidden">
          <Select value={selectedModel} onValueChange={onModelChange}>
            <SelectTrigger className="h-auto w-auto min-w-[120px] cursor-pointer border-0 bg-transparent p-0 text-sm font-medium text-foreground shadow-none focus:ring-0">
              <SelectValue>{activeLabel}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {MODELS.map((m) => (
                <SelectItem key={m.key} value={m.key} className="cursor-pointer p-3">
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </header>
  );
}
