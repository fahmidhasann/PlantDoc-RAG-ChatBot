'use client';

import { useEffect, useRef } from 'react';
import type { Source } from '@/lib/types';

/**
 * The passage drawer.
 *
 * The point of a grounded assistant is that a claim can be checked, so the
 * exact text the model was given is one click away — not a page number the
 * reader has to go and trust. Opens as a right-hand drawer on desktop and a
 * bottom sheet on phones.
 */
export function SourcePanel({
  source,
  all,
  onSelect,
  onClose,
}: {
  source: Source | null;
  all: Source[];
  onSelect: (s: Source) => void;
  onClose: () => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const open = Boolean(source);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    panelRef.current?.focus();
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!source) return null;

  const pages =
    source.pageEnd && source.pageEnd !== source.page
      ? `Pages ${source.page}–${source.pageEnd}`
      : `Page ${source.page}`;

  return (
    <>
      <button
        type="button"
        aria-label="Close source"
        onClick={onClose}
        className="fixed inset-0 z-40 bg-[var(--color-ink)]/25 backdrop-blur-[2px] sm:bg-[var(--color-ink)]/15"
      />
      <div
        ref={panelRef}
        tabIndex={-1}
        role="dialog"
        aria-label={`Source ${source.n}`}
        className="rise fixed inset-x-0 bottom-0 z-50 flex max-h-[82vh] flex-col rounded-t-2xl border border-[var(--color-line)] bg-[var(--color-surface)] shadow-[var(--shadow-lift)] outline-none sm:inset-y-0 sm:left-auto sm:right-0 sm:max-h-none sm:w-[26rem] sm:rounded-none sm:rounded-l-2xl sm:border-y-0 sm:border-r-0"
      >
        <header className="flex items-start gap-3 border-b border-[var(--color-line)] px-5 py-4">
          <span className="mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded-md border border-[var(--color-cite)]/25 bg-[var(--color-cite-soft)] px-1.5 text-xs font-semibold tabular-nums text-[var(--color-cite)]">
            {source.n}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-semibold text-[var(--color-ink)]">{pages}</p>
            {source.topic && (
              <p className="mt-0.5 truncate text-xs text-[var(--color-ink-soft)]">{source.topic}</p>
            )}
            {source.chapter && (
              <p className="mt-0.5 truncate text-[11px] text-[var(--color-ink-faint)]">
                {source.chapter}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-mr-1 -mt-1 rounded-lg p-1.5 text-[var(--color-ink-faint)] transition hover:bg-[var(--color-surface-sunken)] hover:text-[var(--color-ink)]"
          >
            <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M5 5l10 10M15 5L5 15" strokeLinecap="round" />
            </svg>
          </button>
        </header>

        <div className="scrollbar-slim flex-1 overflow-y-auto px-5 py-4">
          <div className="mb-3 flex flex-wrap items-center gap-1.5">
            {source.retrievers.map(r => (
              <span
                key={r}
                className="rounded-full border border-[var(--color-line)] bg-[var(--color-surface-sunken)] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-[var(--color-ink-faint)]"
              >
                {r === 'semantic' ? 'semantic match' : 'exact term match'}
              </span>
            ))}
            {source.score !== undefined && (
              <span className="rounded-full border border-[var(--color-line)] bg-[var(--color-surface-sunken)] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-[var(--color-ink-faint)]">
                relevance {source.score.toFixed(2)}
              </span>
            )}
          </div>

          <p className="font-serif text-[0.95rem] leading-[1.75] text-[var(--color-ink-soft)] whitespace-pre-wrap">
            {source.excerpt}
          </p>
        </div>

        {all.length > 1 && (
          <footer className="border-t border-[var(--color-line)] px-5 py-3">
            <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
              All passages for this answer
            </p>
            <div className="flex flex-wrap gap-1.5">
              {all.map(s => (
                <button
                  key={s.n}
                  type="button"
                  onClick={() => onSelect(s)}
                  className={`rounded-md border px-2 py-1 text-xs tabular-nums transition ${
                    s.n === source.n
                      ? 'border-[var(--color-cite)]/40 bg-[var(--color-cite-soft)] text-[var(--color-cite)]'
                      : 'border-[var(--color-line)] text-[var(--color-ink-soft)] hover:border-[var(--color-line-strong)]'
                  }`}
                >
                  {s.n} · p.{s.page}
                </button>
              ))}
            </div>
          </footer>
        )}
      </div>
    </>
  );
}
