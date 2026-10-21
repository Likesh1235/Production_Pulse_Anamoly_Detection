import clsx from "clsx";
import { useRef, type KeyboardEvent, type ReactNode } from "react";

export interface TabItem<T extends string> { id: T; label: string; icon?: ReactNode }

/** WAI-ARIA tabs with roving tabindex: Left/Right/Home/End move between tabs. */
export function Tabs<T extends string>({ items, value, onChange, label, idPrefix }: {
  items: TabItem<T>[]; value: T; onChange: (id: T) => void; label: string; idPrefix: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next = index;
    if (e.key === "ArrowRight") next = (index + 1) % items.length;
    else if (e.key === "ArrowLeft") next = (index - 1 + items.length) % items.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = items.length - 1;
    else return;
    e.preventDefault();
    onChange(items[next].id);
    refs.current[next]?.focus();
  };
  return (
    <div role="tablist" aria-label={label} className="inline-flex rounded-control border border-line bg-surface-1 p-0.5">
      {items.map((item, i) => {
        const selected = item.id === value;
        return (
          <button key={item.id} ref={(el) => { refs.current[i] = el; }} type="button" role="tab"
            id={`${idPrefix}-tab-${item.id}`} aria-controls={`${idPrefix}-panel-${item.id}`} aria-selected={selected}
            tabIndex={selected ? 0 : -1} onClick={() => onChange(item.id)} onKeyDown={(e) => onKeyDown(e, i)}
            className={clsx("inline-flex items-center gap-1.5 rounded-[5px] px-3 py-1.5 text-xs font-medium transition-colors",
              selected ? "bg-surface-3 text-accent shadow-sm" : "text-ink-muted hover:text-ink")}>
            {item.icon}{item.label}
          </button>
        );
      })}
    </div>
  );
}
