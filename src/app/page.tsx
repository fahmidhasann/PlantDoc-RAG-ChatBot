'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Answer } from '@/components/Answer';
import { Composer } from '@/components/Composer';
import { PipelineTrace, SourceStrip } from '@/components/Pipeline';
import { SourcePanel } from '@/components/SourcePanel';
import { BOOK, STARTER_QUESTIONS } from '@/lib/book';
import { readChatStream } from '@/lib/client/stream';
import type { PreparedImage } from '@/lib/client/image';
import type { ChatTurn, RetrievalStage, Source } from '@/lib/types';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  imageUrl?: string;
  sources?: Source[];
  reranked?: boolean;
  stage?: RetrievalStage;
  stageDetail?: string;
  error?: string;
  model?: string;
  elapsedMs?: number;
  streaming?: boolean;
}

let counter = 0;
const nextId = () => `m${++counter}-${Date.now()}`;

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [selected, setSelected] = useState<Source | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  // Follow the stream, but stop following the moment the reader scrolls up.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const onScroll = () => {
      pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    if (pinnedRef.current) bottomRef.current?.scrollIntoView({ block: 'end' });
  }, [messages]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const patch = useCallback((id: string, update: Partial<Message>) => {
    setMessages(prev => prev.map(m => (m.id === id ? { ...m, ...update } : m)));
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
    setMessages(prev =>
      prev.map(m => (m.streaming ? { ...m, streaming: false, stage: undefined } : m))
    );
  }, []);

  const send = useCallback(
    async (text: string, image: PreparedImage | null) => {
      setNotice(null);

      // History is taken before the new turn is appended, and only completed
      // assistant turns are replayed.
      const history: ChatTurn[] = messages
        .filter(m => !m.error && m.text.trim() && !m.streaming)
        .map(m => ({ role: m.role, content: m.text }));

      const userMessage: Message = {
        id: nextId(),
        role: 'user',
        text,
        imageUrl: image?.previewUrl,
      };
      const assistantId = nextId();
      const assistantMessage: Message = {
        id: assistantId,
        role: 'assistant',
        text: '',
        streaming: true,
        stage: 'understanding',
      };

      setMessages(prev => [...prev, userMessage, assistantMessage]);
      pinnedRef.current = true;
      setBusy(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            message: text,
            history,
            image: image ? { data: image.data, mimeType: image.mimeType } : undefined,
          }),
        });

        let buffered = '';
        let flushTimer: ReturnType<typeof setTimeout> | null = null;
        // Batch deltas into animation-frame-ish flushes: a token-per-render
        // stream re-parses the whole markdown tree on every chunk.
        const flush = () => {
          flushTimer = null;
          if (!buffered) return;
          const chunk = buffered;
          buffered = '';
          setMessages(prev =>
            prev.map(m => (m.id === assistantId ? { ...m, text: m.text + chunk } : m))
          );
        };

        for await (const event of readChatStream(response, controller.signal)) {
          switch (event.type) {
            case 'status':
              patch(assistantId, { stage: event.stage, stageDetail: event.detail });
              break;
            case 'sources':
              patch(assistantId, { sources: event.sources, reranked: true });
              break;
            case 'delta':
              buffered += event.text;
              if (!flushTimer) flushTimer = setTimeout(flush, 60);
              break;
            case 'done':
              if (flushTimer) clearTimeout(flushTimer);
              flush();
              patch(assistantId, {
                streaming: false,
                stage: undefined,
                model: event.model,
                elapsedMs: event.elapsedMs,
              });
              break;
            case 'error':
              if (flushTimer) clearTimeout(flushTimer);
              flush();
              patch(assistantId, { streaming: false, stage: undefined, error: event.message });
              break;
          }
        }

        if (flushTimer) clearTimeout(flushTimer);
        flush();
        setMessages(prev =>
          prev.map(m => (m.id === assistantId ? { ...m, streaming: false, stage: undefined } : m))
        );
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') {
          patch(assistantId, { streaming: false, stage: undefined });
        } else {
          patch(assistantId, {
            streaming: false,
            stage: undefined,
            error:
              'The connection to the server dropped. Check your network and try the question again.',
          });
        }
      } finally {
        abortRef.current = null;
        setBusy(false);
      }
    },
    [messages, patch]
  );

  const empty = messages.length === 0;

  return (
    <div className="flex h-dvh flex-col">
      <header className="sticky top-0 z-30 border-b border-[var(--color-line)] bg-[var(--color-canvas)]/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex min-w-0 items-center gap-2.5">
            <span
              aria-hidden
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[var(--color-leaf-100)] text-sm"
            >
              🌿
            </span>
            <div className="min-w-0">
              <h1 className="text-sm font-semibold leading-tight text-[var(--color-ink)]">PlantDoc</h1>
              <p className="truncate text-[11px] leading-tight text-[var(--color-ink-faint)]">
                grounded in {BOOK.title}
              </p>
            </div>
          </div>
          {!empty && (
            <button
              type="button"
              onClick={() => {
                stop();
                setMessages([]);
                setSelected(null);
                setNotice(null);
              }}
              className="shrink-0 rounded-lg border border-[var(--color-line)] px-2.5 py-1.5 text-xs text-[var(--color-ink-soft)] transition hover:border-[var(--color-line-strong)] hover:text-[var(--color-ink)]"
            >
              New chat
            </button>
          )}
        </div>
      </header>

      <div ref={scrollerRef} className="scrollbar-slim flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 sm:px-6">
          {empty ? (
            <section className="flex min-h-[58vh] flex-col justify-center py-10">
              <h2 className="font-serif text-[1.75rem] leading-snug text-[var(--color-ink)] sm:text-[2rem]">
                Ask the textbook directly.
              </h2>
              <p className="mt-3 max-w-xl text-sm leading-relaxed text-[var(--color-ink-soft)]">
                Every answer is built from passages retrieved out of{' '}
                <span className="font-serif italic">{BOOK.title}</span>
                {BOOK.subtitle ? ` (${BOOK.subtitle})` : ''}, and cites the page it came from. You
                can open any citation to read the original wording. Photographs of affected plants
                are read alongside your question.
              </p>

              <div className="mt-7 grid gap-2 sm:grid-cols-2">
                {STARTER_QUESTIONS.map(q => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => void send(q, null)}
                    className="rounded-xl border border-[var(--color-line)] bg-[var(--color-surface)] px-4 py-3 text-left text-sm leading-snug text-[var(--color-ink-soft)] transition hover:border-[var(--color-leaf-200)] hover:text-[var(--color-ink)] hover:shadow-[var(--shadow-soft)]"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <div className="space-y-7 py-6">
              {messages.map(message =>
                message.role === 'user' ? (
                  <div key={message.id} className="flex justify-end">
                    <div className="max-w-[85%] space-y-2">
                      {message.imageUrl && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={message.imageUrl}
                          alt="Photo you attached"
                          className="ml-auto max-h-56 rounded-xl border border-[var(--color-line)] object-cover"
                        />
                      )}
                      {message.text && (
                        <p className="rounded-2xl rounded-br-md bg-[var(--color-leaf-100)] px-4 py-2.5 text-[0.9375rem] leading-relaxed text-[var(--color-ink)]">
                          {message.text}
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <article key={message.id} className="rise">
                    {message.streaming && !message.text && message.stage && (
                      <PipelineTrace stage={message.stage} detail={message.stageDetail} />
                    )}

                    {message.text && (
                      <Answer
                        text={message.text}
                        sources={message.sources ?? []}
                        streaming={message.streaming}
                        onSelectSource={setSelected}
                      />
                    )}

                    {message.error && (
                      <div className="rounded-xl border border-[var(--color-danger)]/25 bg-[var(--color-danger-soft)] px-4 py-3 text-sm leading-relaxed text-[var(--color-danger)]">
                        {message.error}
                      </div>
                    )}

                    {!message.streaming && message.sources?.length ? (
                      <SourceStrip
                        sources={message.sources}
                        reranked={message.reranked}
                        onSelect={setSelected}
                      />
                    ) : null}

                    {!message.streaming && message.model && (
                      <p className="mt-2 text-[10px] text-[var(--color-ink-faint)]">
                        {message.model}
                        {message.elapsedMs ? ` · ${(message.elapsedMs / 1000).toFixed(1)}s` : ''}
                      </p>
                    )}
                  </article>
                )
              )}
              <div ref={bottomRef} className="h-px" />
            </div>
          )}
        </div>
      </div>

      <div className="sticky bottom-0 border-t border-[var(--color-line)] bg-[var(--color-canvas)]/85 backdrop-blur-md">
        <div className="mx-auto max-w-3xl px-4 pb-4 pt-3 sm:px-6">
          {notice && (
            <p className="mb-2 rounded-lg border border-[var(--color-danger)]/25 bg-[var(--color-danger-soft)] px-3 py-2 text-xs text-[var(--color-danger)]">
              {notice}
            </p>
          )}
          <Composer busy={busy} onSend={send} onStop={stop} onError={setNotice} />
          <p className="mt-2 text-center text-[11px] leading-relaxed text-[var(--color-ink-faint)]">
            Answers come from the indexed book only. Chemical recommendations are textbook
            guidance, not a product label — check local registration before applying anything.
          </p>
        </div>
      </div>

      <SourcePanel
        source={selected}
        all={selected ? (messages.find(m => m.sources?.some(s => s.n === selected.n))?.sources ?? []) : []}
        onSelect={setSelected}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}
