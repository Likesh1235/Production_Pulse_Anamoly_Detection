import clsx from "clsx";
import { FileSpreadsheet, UploadCloud } from "lucide-react";
import { useId, useRef, useState, type DragEvent } from "react";

export const MAX_FILE_BYTES = 80 * 1024 * 1024;

export interface LoadedFile { name: string; size: number; text: string }

/** Keyboard- and drag-friendly CSV picker. Files are read locally in the browser only. */
export function CsvDropZone({ onLoaded, onError, file, label = "CSV file", disabled }: {
  onLoaded: (f: LoadedFile) => void; onError: (message: string) => void; file?: LoadedFile | null; label?: string; disabled?: boolean;
}) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [reading, setReading] = useState(false);

  const read = async (f: File | undefined) => {
    if (!f) return;
    if (!/\.(csv|txt)$/i.test(f.name)) { onError(`"${f.name}" is not a .csv file.`); return; }
    if (f.size > MAX_FILE_BYTES) { onError(`"${f.name}" is ${(f.size / 1e6).toFixed(1)} MB; the browser limit here is ${MAX_FILE_BYTES / 1e6} MB. Split the file.`); return; }
    setReading(true);
    try { onLoaded({ name: f.name, size: f.size, text: await f.text() }); }
    catch { onError(`Could not read "${f.name}".`); }
    finally { setReading(false); if (inputRef.current) inputRef.current.value = ""; }
  };
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDragging(false); if (!disabled) void read(e.dataTransfer.files?.[0]); };

  return (
    <div>
      <label htmlFor={inputId} className="field-label">{label}</label>
      <div onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}
        className={clsx("grid-bg flex flex-col items-center gap-2 rounded-panel border border-dashed px-4 py-6 text-center transition-colors",
          dragging ? "border-accent bg-accent/5" : "border-line-strong", disabled && "opacity-50")}>
        <UploadCloud className="h-6 w-6 text-accent" aria-hidden />
        <p className="text-xs text-ink-muted">Drag a CSV here, or</p>
        <input ref={inputRef} id={inputId} type="file" accept=".csv,text/csv,.txt" className="sr-only" disabled={disabled}
          onChange={(e) => void read(e.target.files?.[0])} />
        <button type="button" className="btn-secondary btn-sm" disabled={disabled || reading} onClick={() => inputRef.current?.click()}>
          {reading ? "Reading…" : "Choose file"}
        </button>
        <p className="text-2xs text-ink-faint">Header row with the 52 feature names (any order). Optional: sample_id, timestamp, faultNumber, simulationRun, sample.</p>
      </div>
      {file && (
        <p className="mt-2 flex items-center gap-2 text-xs text-ink-muted">
          <FileSpreadsheet className="h-3.5 w-3.5 text-accent" aria-hidden />
          <span className="truncate font-medium text-ink">{file.name}</span>
          <span className="num text-ink-faint">{(file.size / 1024).toFixed(1)} KB</span>
        </p>
      )}
    </div>
  );
}
