import { CheckCircle2 } from "lucide-react";

export function WorkflowStepper({ steps, activeIndex }: { steps: string[]; activeIndex: number }) {
  return (
    <nav aria-label="Workflow progress" className="steps">
      {steps.map((step, index) => {
        const done = index < activeIndex;
        return (
          <div className={`step ${index === activeIndex ? "active" : ""} ${done ? "done" : ""}`} key={step}>
            {done ? <CheckCircle2 size={17} /> : <span className="dot" />}
            <span>{step}</span>
          </div>
        );
      })}
    </nav>
  );
}
