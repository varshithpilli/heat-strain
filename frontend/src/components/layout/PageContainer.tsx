import { useState, useCallback, type ReactNode } from "react";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";

interface PageContainerProps {
  children: ReactNode;
  selectedModel: string;
  onModelChange: (model: string) => void;
}

export default function PageContainer({
  children,
  selectedModel,
  onModelChange,
}: PageContainerProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const toggleSidebar = useCallback(() => setCollapsed((c) => !c), []);
  const toggleMobile = useCallback(() => setMobileOpen((o) => !o), []);

  return (
    <div className="flex min-h-screen">
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar — hidden on mobile by default, shown via overlay */}
      <div
        className={`fixed z-40 md:relative md:block ${
          mobileOpen ? "block" : "hidden md:block"
        }`}
      >
        <Sidebar collapsed={collapsed} onToggle={toggleSidebar} />
      </div>

      {/* Main content */}
      <div
        className={`flex flex-1 flex-col transition-[margin] duration-200 ${
          collapsed ? "md:ml-16" : "md:ml-60"
        }`}
      >
        <Navbar
          selectedModel={selectedModel}
          onModelChange={onModelChange}
          onMenuToggle={toggleMobile}
        />
        <main className="flex-1 overflow-y-auto p-4 md:p-6">{children}</main>
      </div>
    </div>
  );
}
