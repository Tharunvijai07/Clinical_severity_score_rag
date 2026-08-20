import { Check } from "lucide-react";
import clsx from "clsx";

interface Step {
  label: string;
  description: string;
}

interface StepIndicatorProps {
  steps: Step[];
  currentStep: number;
}

export default function StepIndicator({ steps, currentStep }: StepIndicatorProps) {
  return (
    <nav className="flex items-center gap-1 sm:gap-2 w-full overflow-x-auto py-2" id="step-indicator" aria-label="Analysis progress">
      {steps.map((step, index) => {
        const isCompleted = index < currentStep;
        const isActive = index === currentStep;
        const isPending = index > currentStep;

        return (
          <div key={step.label} className="flex items-center gap-1 sm:gap-2 flex-1 min-w-0">
            {/* Step circle */}
            <div
              className={clsx(
                "flex items-center justify-center w-8 h-8 rounded-full text-xs font-semibold flex-shrink-0 transition-all duration-300",
                isCompleted && "bg-teal-500/20 text-teal-400 border border-teal-500/40",
                isActive && "bg-teal-500/30 text-teal-300 border-2 border-teal-400 shadow-[0_0_12px_hsl(174_50%_55%/0.25)]",
                isPending && "border border-[hsl(var(--border))] text-[hsl(var(--text-muted))]",
              )}
            >
              {isCompleted ? <Check className="w-4 h-4" /> : index + 1}
            </div>

            {/* Label */}
            <div className="min-w-0 hidden sm:block">
              <p
                className={clsx(
                  "text-xs font-semibold truncate transition-colors duration-300",
                  isActive ? "text-teal-300" : isCompleted ? "text-teal-400/70" : "text-[hsl(var(--text-muted))]",
                )}
              >
                {step.label}
              </p>
              <p className="text-[10px] truncate" style={{ color: "hsl(var(--text-muted))" }}>
                {step.description}
              </p>
            </div>

            {/* Connector line */}
            {index < steps.length - 1 && (
              <div className="flex-1 min-w-4 h-px mx-1" style={{
                background: isCompleted
                  ? "linear-gradient(90deg, hsl(174 42% 42%), hsl(174 50% 55%))"
                  : "hsl(var(--border))",
              }} />
            )}
          </div>
        );
      })}
    </nav>
  );
}
