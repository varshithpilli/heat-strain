import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import PageContainer from "@/components/layout/PageContainer";
import Dashboard from "@/pages/Dashboard";
import ModelComparison from "@/pages/ModelComparison";
import CrossValidation from "@/pages/CrossValidation";
import About from "@/pages/About";

export function App() {
  const [selectedModel, setSelectedModel] = useState("random_forest");

  return (
    <PageContainer
      selectedModel={selectedModel}
      onModelChange={setSelectedModel}
    >
      <Routes>
        <Route path="/" element={<Dashboard selectedModel={selectedModel} />} />
        <Route path="/comparison" element={<ModelComparison />} />
        <Route path="/cv" element={<CrossValidation />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </PageContainer>
  );
}

export default App;
