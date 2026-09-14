import { AlertCircle, CheckCircle2 } from "lucide-react";

export function QualitySummary({ report }: { report: Record<string, unknown> | null | undefined }) {
  if (!report) return <div className="empty">Quality report pending</div>;
  const status = String(report.status ?? report.validation_status ?? (report.passed ? "passed" : "available"));
  const passed = status.toLowerCase().includes("pass") || report.passed === true;
  return (
    <div className="quality-summary">
      <div className={`validation-badge ${passed ? "passed" : "failed"}`}>
        {passed ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
        <span>{status}</span>
      </div>
      <pre>{JSON.stringify(report, null, 2)}</pre>
    </div>
  );
}
