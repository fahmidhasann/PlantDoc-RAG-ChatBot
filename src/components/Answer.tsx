'use client';

import React, { Fragment, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Source } from '@/lib/types';

/**
 * Renders a streamed markdown answer and turns inline `[n]` markers into
 * clickable citation chips.
 *
 * The markers are rewritten after markdown parsing rather than before, so a
 * `[2]` that lands inside a bullet, a table cell or a bold run is still caught,
 * while a literal bracket in a code span is left alone. A marker whose number
 * has no matching source is left as plain text — the model occasionally cites
 * past the end of the list, and a dead chip is worse than no chip.
 */

// Matches [3], [3][7] and the grouped form [3, 7] the model also produces.
const MARKER = /\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g;

function CitationChip({ n, source, onSelect }: { n: number; source: Source; onSelect: (s: Source) => void }) {
  return (
    <button
      type="button"
      onClick={() => onSelect(source)}
      title={`Page ${source.page}${source.topic ? ` — ${source.topic}` : ''}`}
      aria-label={`Source ${n}, page ${source.page}`}
      className="mx-[0.1em] inline-flex h-[1.15em] min-w-[1.15em] translate-y-[-0.15em] items-center justify-center rounded-[0.3em] border border-[var(--color-cite)]/25 bg-[var(--color-cite-soft)] px-[0.3em] align-middle text-[0.68em] font-semibold text-[var(--color-cite)] tabular-nums transition hover:border-[var(--color-cite)]/60 hover:brightness-[0.97]"
    >
      {n}
    </button>
  );
}

/** Replaces `[n]` inside every string leaf of a rendered node tree. */
function linkCitations(
  node: React.ReactNode,
  sources: Source[],
  onSelect: (s: Source) => void,
  keyPrefix = 'c'
): React.ReactNode {
  if (typeof node === 'string') {
    if (!node.includes('[')) return node;

    const out: React.ReactNode[] = [];
    let cursor = 0;
    let match: RegExpExecArray | null;
    MARKER.lastIndex = 0;

    while ((match = MARKER.exec(node)) !== null) {
      const cited = match[1]
        .split(',')
        .map(part => Number(part.trim()))
        .map(n => ({ n, source: sources.find(s => s.n === n) }))
        .filter((c): c is { n: number; source: Source } => Boolean(c.source));

      // A marker that cites nothing we hold is left as literal text rather
      // than rendered as a chip that opens nothing.
      if (!cited.length) continue;

      if (match.index > cursor) out.push(node.slice(cursor, match.index));
      for (const { n, source } of cited) {
        out.push(
          <CitationChip
            key={`${keyPrefix}-${match.index}-${n}`}
            n={n}
            source={source}
            onSelect={onSelect}
          />
        );
      }
      cursor = match.index + match[0].length;
    }

    if (!out.length) return node;
    if (cursor < node.length) out.push(node.slice(cursor));
    return <>{out}</>;
  }

  if (Array.isArray(node)) {
    return node.map((child, i) => (
      <Fragment key={i}>{linkCitations(child, sources, onSelect, `${keyPrefix}-${i}`)}</Fragment>
    ));
  }

  if (React.isValidElement(node)) {
    const element = node as React.ReactElement<{ children?: React.ReactNode }>;
    // Code spans keep their literal brackets.
    if (element.type === 'code') return node;
    const children = element.props.children;
    if (children === undefined) return node;
    return React.cloneElement(element, {
      children: linkCitations(children, sources, onSelect, keyPrefix),
    });
  }

  return node;
}

export function Answer({
  text,
  sources,
  streaming,
  onSelectSource,
}: {
  text: string;
  sources: Source[];
  streaming?: boolean;
  onSelectSource: (s: Source) => void;
}) {
  const components = useMemo(() => {
    const wrap =
      <T extends keyof React.JSX.IntrinsicElements>(Tag: T) =>
      ({ children, ...rest }: { children?: React.ReactNode }) =>
        React.createElement(Tag, rest, linkCitations(children, sources, onSelectSource));

    return {
      p: wrap('p'),
      li: wrap('li'),
      td: wrap('td'),
      th: wrap('th'),
      h1: wrap('h2'), // the answer lives under the page's own h1
      h2: wrap('h2'),
      h3: wrap('h3'),
      h4: wrap('h3'),
      blockquote: wrap('blockquote'),
      a: ({ children, ...rest }: { children?: React.ReactNode; href?: string }) =>
        React.createElement(
          'a',
          { ...rest, target: '_blank', rel: 'noreferrer noopener' },
          children
        ),
    } as React.ComponentProps<typeof ReactMarkdown>['components'];
  }, [sources, onSelectSource]);

  return (
    <div className={`answer ${streaming ? 'caret' : ''}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
