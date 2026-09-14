import {
  DatabaseZap,
  FolderKanban,
  HelpCircle,
  Home,
  LockKeyhole,
  UserCircle
} from "lucide-react";
import { PropsWithChildren } from "react";
import { Link, NavLink } from "react-router-dom";
import { workflows } from "../lib/workflows";

const navItems = [
  { to: "/", label: "Home", Icon: Home },
  { to: "/projects", label: "My Twins", Icon: FolderKanban },
  { to: "/help", label: "Help & Guides", Icon: HelpCircle }
];

export function AppShell({ children }: PropsWithChildren) {
  return (
    <div className="app-frame">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <DatabaseZap size={22} />
          </div>
          <div>
            <strong>Synthetic Data Platform</strong>
          </div>
        </div>
        <nav aria-label="Primary navigation" className="side-nav">
          {navItems.map(({ to, label, Icon }) => (
            <NavLink className={({ isActive }) => `side-link ${isActive ? "active" : ""}`} end={to === "/"} key={to} to={to}>
              <Icon size={18} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="nav-section">
          <span>Specialized Workflows</span>
          <nav aria-label="Workflow navigation" className="side-nav workflow-nav">
            {workflows.map(({ key, route, title, Icon, accent }) => (
              <NavLink
                aria-label={`${key} workflow`}
                className={({ isActive }) => `side-link workflow-${accent} ${isActive ? "active" : ""}`}
                key={route}
                to={route}
              >
                <Icon size={18} />
                <span>{title}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div>
            <h1>Synthetic Data Platform</h1>
          </div>
          <div className="top-actions">
            <div className="service-pill">
              <LockKeyhole size={15} />
              <span>Air-Gapped Mode</span>
            </div>
            <div className="user-pill">
              <UserCircle size={22} />
              <span>Local User</span>
            </div>
          </div>
        </header>
        <main className="content">{children}</main>
      </div>
    </div>
  );
}
