Below is a **complete frontend project template** you can use to build a **sleek, modern dark-mode UI** for your HeatGuard system. It mirrors the **functionality and structure of your Streamlit app** (Live Prediction, Model Comparison, Cross-Validation, About, sensor inputs, predictions, etc.)  but designed as a **modern React dashboard**.

This is **not just design ideas** — it’s a **practical project structure + UI blueprint** you can implement directly with **React + Vite + Tailwind**.

---

# Recommended Stack

```
React
Vite
TailwindCSS
ShadCN UI
Recharts
Axios
Framer Motion
```

Why:

* **ShadCN** → modern component system
* **Recharts** → probability + model charts
* **Framer Motion** → smooth animations
* **Axios** → API communication

---

# Project Folder Structure

```
heatguard-frontend/

├─ public/
│
├─ src/
│
│  ├─ components/
│  │   ├─ layout/
│  │   │   ├─ Sidebar.jsx
│  │   │   ├─ Navbar.jsx
│  │   │   └─ PageContainer.jsx
│  │   │
│  │   ├─ prediction/
│  │   │   ├─ SensorInputs.jsx
│  │   │   ├─ RiskCards.jsx
│  │   │   ├─ ProbabilityChart.jsx
│  │   │   └─ ConfidenceMetrics.jsx
│  │   │
│  │   ├─ charts/
│  │   │   ├─ ModelComparisonChart.jsx
│  │   │   └─ CVChart.jsx
│  │   │
│  │   └─ ui/
│  │       ├─ Card.jsx
│  │       ├─ Gauge.jsx
│  │       └─ MetricBox.jsx
│  │
│  ├─ pages/
│  │   ├─ Dashboard.jsx
│  │   ├─ ModelComparison.jsx
│  │   ├─ CrossValidation.jsx
│  │   └─ About.jsx
│  │
│  ├─ api/
│  │   └─ heatguardAPI.js
│  │
│  ├─ hooks/
│  │   └─ usePrediction.js
│  │
│  ├─ styles/
│  │   └─ globals.css
│  │
│  ├─ App.jsx
│  └─ main.jsx
│
├─ tailwind.config.js
└─ package.json
```

---

# Dark Theme Design

### Background

```
#0B0F14
```

### Card

```
#111827
```

### Border

```
#1F2937
```

### Text

```
Primary: #F9FAFB
Secondary: #9CA3AF
```

### Risk Colors

```
Normal    → #22C55E
Moderate  → #F59E0B
High      → #EF4444
```

---

# Main Layout

```
┌────────────── Sidebar ───────────────┐
│                                      │
│ HeatGuard AI                         │
│                                      │
│ Dashboard                            │
│ Model Comparison                     │
│ Cross Validation                     │
│ About                                │
│                                      │
└───────────────┬──────────────────────┘
                │
                ▼

         ┌──────── Navbar ─────────┐
         │ Model selector | status │
         └─────────────────────────┘

         ┌──────── Main Content ───┐
         │                         │
         │ Dashboard / Charts     │
         │                         │
         └─────────────────────────┘
```

---

# Sidebar Component

`Sidebar.jsx`

```jsx
export default function Sidebar() {
  return (
    <div className="h-screen w-64 bg-slate-900 border-r border-slate-800 flex flex-col">

      <div className="p-6 text-xl font-bold text-white">
        HeatGuard AI
      </div>

      <nav className="flex flex-col gap-2 px-4">

        <a className="sidebar-link">Dashboard</a>
        <a className="sidebar-link">Model Comparison</a>
        <a className="sidebar-link">Cross Validation</a>
        <a className="sidebar-link">About</a>

      </nav>

    </div>
  );
}
```

Tailwind class:

```
.sidebar-link {
  @apply px-4 py-2 rounded-lg text-gray-300 hover:bg-slate-800 transition;
}
```

---

# Dashboard Page

`Dashboard.jsx`

Layout mirrors your Streamlit **Live Prediction tab**.

```
┌───────────────────────────────────────┐
│ Sensor Inputs | Prediction Results    │
└───────────────────────────────────────┘
```

```jsx
import SensorInputs from "../components/prediction/SensorInputs";
import RiskCards from "../components/prediction/RiskCards";
import ProbabilityChart from "../components/prediction/ProbabilityChart";

export default function Dashboard() {

  return (
    <div className="grid grid-cols-2 gap-6">

      <SensorInputs />

      <div className="flex flex-col gap-6">

        <RiskCards />

        <ProbabilityChart />

      </div>

    </div>
  );
}
```

---

# Sensor Input Panel

Matches your **Streamlit sliders**.

`SensorInputs.jsx`

```jsx
export default function SensorInputs() {

  return (

    <div className="card">

      <h2 className="card-title">Sensor Inputs</h2>

      <div className="space-y-4">

        <InputSlider label="Body Temperature" />
        <InputSlider label="Ambient Temperature" />
        <InputSlider label="Heart Rate" />
        <InputSlider label="Humidity" />
        <InputSlider label="Skin Resistance" />
        <InputSlider label="Respiration Rate" />
        <InputSlider label="Movement Level" />

      </div>

    </div>

  );
}
```

---

# Risk Cards

Matches your Streamlit **risk status boxes**.

```
Normal
Moderate
High
```

`RiskCards.jsx`

```jsx
export default function RiskCards({ heatRisk, dehydrationRisk }) {

  return (

    <div className="grid grid-cols-2 gap-4">

      <RiskCard title="Heat Stress" risk={heatRisk} />
      <RiskCard title="Dehydration" risk={dehydrationRisk} />

    </div>

  );

}
```

Card style:

```
.card {
 @apply bg-slate-900 border border-slate-800 rounded-xl p-6;
}
```

---

# Probability Chart

Mirrors your **Streamlit probability breakdown chart**.

Use **Recharts**.

`ProbabilityChart.jsx`

```jsx
import { BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

export default function ProbabilityChart({ data }) {

  return (

    <div className="card">

      <h3 className="card-title">Probability Breakdown</h3>

      <BarChart width={400} height={250} data={data}>

        <XAxis dataKey="label" />
        <YAxis />
        <Tooltip />

        <Bar dataKey="heat" fill="#ef4444" />
        <Bar dataKey="dehydration" fill="#22c55e" />

      </BarChart>

    </div>

  );
}
```

---

# Model Comparison Page

Mirrors **Streamlit comparison tab**.

Layout:

```
┌──────────────────────────────┐
│ Performance Table            │
└──────────────────────────────┘

┌──────────────────────────────┐
│ Accuracy / F1 / ROC charts   │
└──────────────────────────────┘
```

Charts:

* Accuracy comparison
* ROC curves
* Confusion matrix images

---

# Cross Validation Page

Shows:

```
CV Mean
CV Std
Model ranking
```

Charts:

* Error bars
* CV distribution

---

# About Page

Display information similar to Streamlit:

```
HeatGuard AI

Classification System

0 → Normal
1 → Moderate
2 → High

Model Architecture
Custom Neural Network
Random Forest
Gradient Boosting
SVM
Logistic Regression
```

---

# API Client

`heatguardAPI.js`

```javascript
import axios from "axios";

const API = "http://localhost:8000";

export async function predict(data) {

  const res = await axios.post(`${API}/predict`, data);

  return res.data;
}
```

---

# Dark Mode Styling

`globals.css`

```
body {
  background: #0B0F14;
  color: #F9FAFB;
}
```

---

# Animations

Use **Framer Motion** for:

* risk cards appearing
* charts updating
* alerts flashing

Example:

```
animate-pulse
transition-all
duration-300
```

---

# Final Result UI

Your UI will look like a **modern AI dashboard**:

```
HeatGuard AI

┌─────────────┬────────────────────┐
│ Sensors     │ Heat Risk   HIGH   │
│             │ Dehyd Risk  MOD    │
│ sliders     │                    │
│             │ Probability Chart  │
└─────────────┴────────────────────┘
```

Much cleaner than Streamlit.

---

# If you want, I can also give you a **complete starter repo template** for this project that includes:

* FastAPI backend
* React dashboard
* Tailwind + ShadCN
* Charts
* Dark mode
* Prediction integration

Basically a **full production-grade ML dashboard** you can run in **~5 minutes**.
