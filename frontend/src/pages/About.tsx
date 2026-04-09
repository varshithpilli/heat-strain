import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ShieldCheck, MessageSquare } from "lucide-react";

const RISK_CLASSES = [
  { level: 0, label: "Normal", action: "Routine monitoring" },
  { level: 1, label: "Moderate", action: "Rest + hydrate, monitor" },
  { level: 2, label: "High", action: "Immediate medical help" },
];

const INPUT_FEATURES = [
  "Body & Ambient Temperature",
  "Heart Rate & Skin Resistance",
  "Humidity & Respiration Rate",
  "Movement Level",
  "Derived: Heat Index, THI, ΔT",
];

const NN_DETAILS = [
  { label: "Architecture", value: "4 layers (128→64→32→3)" },
  { label: "Activations", value: "ReLU + Softmax" },
  { label: "Optimiser", value: "Adam with LR decay" },
  { label: "Regularisation", value: "L2 + Dropout + Batch Norm" },
  { label: "Training", value: "Early stopping (patience=20)" },
];

export default function About() {
  return (
    <div className="mx-auto max-w-4xl space-y-8">
      {/* Hero */}
      <div className="text-center">
        <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-lg bg-accent">
          <ShieldCheck className="h-6 w-6 text-muted-foreground" />
        </div>
        <h1 className="mb-2 font-heading text-xl font-semibold tracking-tight">
          About HeatGuard AI
        </h1>
        <p className="mx-auto max-w-md text-sm leading-relaxed text-muted-foreground">
          An intelligent heat stress and dehydration risk prediction system
          powered by multiple machine learning models. Designed for real-time
          monitoring and early warning.
        </p>
      </div>

      {/* Two-column content */}
      {/* <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 items-start"> */}
      <div className="columns-1 lg:columns-2 gap-6 space-y-6">
        {/* Left column */}


        <div className="flex flex-col gap-6">
          
          {/* Custom Neural Network */}
          <div className="break-inside-avoid">
            <h2 className="mb-3 font-heading text-sm font-semibold">Custom Neural Network</h2>
            <Card className="border-border bg-card">
              <CardContent className="space-y-3">
                <ul className="space-y-2">
                  {NN_DETAILS.map((d) => (
                    <li key={d.label} className="flex items-start gap-2 text-sm">
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                      <span>
                        <span className="font-medium">{d.label}:</span>{" "}
                        <span className="text-muted-foreground">{d.value}</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>


        {/* Input Features */}
        <div className="break-inside-avoid mb-6">
          <h2 className="mb-3 font-heading text-sm font-semibold">Input Features</h2>
          <Card className="border-border bg-card">
            <CardContent className="">
              <ul className="space-y-2">
                {INPUT_FEATURES.map((f) => (
                  <li key={f} className="flex items-center gap-2 text-sm text-muted-foreground">
                    <span className="h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                    {f}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>

          {/* Mobile Alert */}
          <div>
            <h2 className="mb-3 font-heading text-sm font-semibold flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-muted-foreground" />
              Mobile Alert
            </h2>
            <Card className="border-border bg-card">
              <CardContent className="">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Sends a WhatsApp alert to the configured mobile number(s) when the prediction
                  results in <Badge variant="outline" className="border-border text-xs text-muted-foreground mx-0.5">Moderate (1)</Badge> or{" "}
                  <Badge variant="outline" className="border-border text-xs text-muted-foreground mx-0.5">High (2)</Badge> risk, along with the
                  percentage probability of the risk, and a few quick precautions.
                </p>
              </CardContent>
            </Card>
          </div>









                  {/* Graph Attention Network */}
          <div className="break-inside-avoid mb-6">
            <h2 className="mb-3 font-heading text-sm font-semibold">Graph Attention Network</h2>
            <Card className="border-border bg-card">
              <CardContent className="space-y-3">
                <ul className="space-y-2">
                  <li className="flex items-start gap-2 text-sm">
                    <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                    <span><span className="font-medium">Architecture:</span> <span className="text-muted-foreground">GATConv(3→512, heads=8) → GATConv(512→128, heads=4) → skip-add → GlobalMeanPool → Dense(32) → n_classes</span></span>
                  </li>
                  <li className="flex items-start gap-2 text-sm">
                    <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                    <span><span className="font-medium">Graph structure:</span> <span className="text-muted-foreground">Correlation-based k-NN edges (k=4)</span></span>
                  </li>
                  <li className="flex items-start gap-2 text-sm">
                    <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                    <span><span className="font-medium">Activations:</span> <span className="text-muted-foreground">ReLU + Softmax</span></span>
                  </li>
                  <li className="flex items-start gap-2 text-sm">
                    <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                    <span><span className="font-medium">Optimiser:</span> <span className="text-muted-foreground">Adam (lr=0.001)</span></span>
                  </li>
                  <li className="flex items-start gap-2 text-sm">
                    <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                    <span><span className="font-medium">Regularisation:</span> <span className="text-muted-foreground">Dropout (0.3 / 0.3) + Batch Norm</span></span>
                  </li>
                  <li className="flex items-start gap-2 text-sm">
                    <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                    <span><span className="font-medium">Training:</span> <span className="text-muted-foreground">60 epochs + 5-fold CV + threshold calibration</span></span>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>


          {/* Classification System */}
          <div className="break-inside-avoid mb-6">
            <h2 className="mb-3 font-heading text-sm font-semibold">Classification System</h2>
            <Card className="border-border bg-card">
              <CardContent className="">
                <Table>
                  <TableHeader>
                    <TableRow className="border-border">
                      <TableHead className="text-xs text-muted-foreground">Class</TableHead>
                      <TableHead className="text-xs text-muted-foreground">Risk Level</TableHead>
                      <TableHead className="text-xs text-muted-foreground">Action</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {RISK_CLASSES.map((cls) => (
                      <TableRow key={cls.level} className="border-border">
                        <TableCell className="text-sm tabular-nums">{cls.level}</TableCell>
                        <TableCell className="text-sm font-medium">{cls.label}</TableCell>
                        <TableCell className="text-sm text-muted-foreground">{cls.action}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>







        {/* Right column */}
        <div className="space-y-6">



        </div>
      </div>
    </div>
  );
}
