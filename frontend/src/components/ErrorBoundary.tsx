import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Unhandled UI error", error, info);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="rounded border border-red-200 bg-red-50 p-4 text-sm text-red-950" role="alert">
          <p className="font-semibold">The interface could not render this section.</p>
          <p className="mt-1">{this.state.error.message}</p>
        </div>
      );
    }

    return this.props.children;
  }
}
