'use client';

import type { RetrievalStage, Source } from '@/lib/types';

/**
 * The retrieval trace shown while an answer is being produced, and collapsed
 * into a one-line summary afterwards.
 *
 * Most chat UIs show a spinner here. A spinner says "wait"; this says what is
 * being waited on — which is the difference between a slow answer and an app
 * that looks broken. It also makes the pipeline legible: anyone looking at this
 * project can see that a question is rewritten, searched two ways, fused and
 * reranked before a model ever sees it.
 */

const STAGES: Array<{ id: RetrievalStage; label: string }> = [
  { id: 'understanding', label: 'Reading the question' },
  { id: 'searching', label: 'Searching the book' },
  { id: 'ranking', label: 'Ranking passages' },
  { id: 'writing', label: 'Writing the answer' },
];

export function PipelineTrace({ stage, detail }: { stage: RetrievalStage; detail?: string }) {
  const current = STAGES.findIndex(s => s.id === stage);

  return (
    <div className="rise rounded-xl border border-[var(--color-line)] bg-[var(--color-surface)] px-4 py-3">
      <ol className="space-y-1.5">
        {STAGES.map((s, i) => {
          const done = i < current;
          const active = i === current;
          return (
            <li key={s.id} className="flex items-center gap-2.5 text-xs">
              <span
                aria-hidden
                className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full transition ${
                  done
                    ? 'bg-[var(--color-leaf-400)]'
                    : active
                      ? 'animate-pulse bg-[var(--color-leaf-500)]'
                      : 'bg-[var(--color-line-strong)]'
                }`}
              />
              <span
                className={
                  active
                    ? 'font-medium text-[var(--color-ink)]'
                    : done
                      ? 'text-[var(--color-ink-faint)]'
                      : 'text-[var(--color-ink-faint)]/60'
                }
              >
                {s.label}
              </span>
            </li>
          );
        })}
      </ol>
      {detail && (
        <p className="mt-2.5 border-t border-[var(--color-line)] pt-2 font-serif text-xs italic leading-relaxed text-[var(--color-ink-faint)]">
          {detail}
        </p>
      )}
    </div>
  );
}

export function SourceStrip({
  sources,
  reranked,
  onSelect,
}: {
  sources: Source[];
  reranked?: boolean;
  onSelect: (s: Source) => void;
}) {
  if (!sources.length) return null;

  const both = sources.filter(s => s.retrievers.length > 1).length;

  return (
    <div className="mt-4 border-t border-[var(--color-line)] pt-3">
      <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
        {sources.length} passage{sources.length === 1 ? '' : 's'} from the book
        {both > 0 && ` · ${both} matched both ways`}
        {reranked && ' · reranked'}
      </p>
      <div className="flex flex-wrap gap-1.5">
        {sources.map(s => (
          <button
            key={s.n}
            type="button"
            onClick={() => onSelect(s)}
            title={s.topic || undefined}
            className="group flex max-w-full items-center gap-1.5 rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] py-1 pl-1.5 pr-2.5 text-left transition hover:border-[var(--color-line-strong)] hover:shadow-[var(--shadow-soft)]"
          >
            <span className="inline-flex h-4 min-w-4 items-center justify-center rounded-[4px] bg-[var(--color-cite-soft)] px-1 text-[10px] font-semibold tabular-nums text-[var(--color-cite)]">
              {s.n}
            </span>
            <span className="truncate text-[11px] text-[var(--color-ink-soft)]">
              p.{s.page}
              {s.topic ? ` · ${s.topic}` : ''}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
