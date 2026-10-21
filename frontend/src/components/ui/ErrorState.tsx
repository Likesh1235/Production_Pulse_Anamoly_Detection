import { RefreshCw } from "lucide-react";
import { API_BASE_URL } from "../../lib/config";
import { describeError } from "../../lib/api/errors";
import { Banner, Button } from "./primitives";

/** Renders an API failure with its real message, hint and any per-field details. */
export function ErrorState({ error, onRetry, context }: { error: unknown; onRetry?: () => void; context?: string }) {
  const d = describeError(error, API_BASE_URL);
  return (
    <Banner tone="danger" title={context ? `${context}: ${d.title}` : d.title}
      actions={onRetry && <Button size="sm" variant="secondary" icon={<RefreshCw className="h-3.5 w-3.5" aria-hidden />} onClick={onRetry}>Retry</Button>}>
      <p>{d.message}</p>
      {d.items.length > 0 && (
        <ul className="mt-1.5 list-disc space-y-0.5 pl-4 font-mono text-2xs text-ink">
          {d.items.map((item) => <li key={item}>{item}</li>)}
        </ul>
      )}
      {d.hint && <p className="mt-1.5 text-ink-faint">{d.hint}</p>}
    </Banner>
  );
}
