import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  BarChart3,
  FlaskConical,
  Info,
  ShieldCheck,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/comparison", label: "Model Comparison", icon: BarChart3 },
  { to: "/cv", label: "Cross Validation", icon: FlaskConical },
  { to: "/about", label: "About", icon: Info },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={`fixed left-0 top-0 z-40 flex h-screen flex-col border-r border-border bg-sidebar transition-[width] duration-200 ${
        collapsed ? "w-16" : "w-60"
      }`}
    >
      {/* Logo — click to toggle sidebar */}
      <button
        onClick={onToggle}
        className="flex h-14 w-full cursor-pointer items-center gap-2.5 border-b border-border px-4 transition-colors hover:bg-accent/50"
      >
        <ShieldCheck className="h-5 w-5 shrink-0 text-foreground" />
        {!collapsed && (
          <span className="text-sm font-semibold tracking-tight text-foreground">
            HeatGuard AI
          </span>
        )}
      </button>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col gap-0.5 px-2 py-3">
        {navItems.map((item) => {
          const link = (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center rounded-md px-3 py-2 text-[13px] font-medium transition-colors duration-150 cursor-pointer ${
                  isActive
                    ? "bg-accent text-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                } ${collapsed ? "justify-center px-0 mx-1" : "gap-2.5"}`
              }
            >
              <item.icon className="h-4 w-4 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          );

          if (collapsed) {
            return (
              <Tooltip key={item.to} delayDuration={0}>
                <TooltipTrigger asChild>{link}</TooltipTrigger>
                <TooltipContent side="right" className="text-xs">
                  {item.label}
                </TooltipContent>
              </Tooltip>
            );
          }

          return link;
        })}
      </nav>
    </aside>
  );
}
