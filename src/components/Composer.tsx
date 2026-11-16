'use client';

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { prepareImage, type PreparedImage } from '@/lib/client/image';

export function Composer({
  busy,
  onSend,
  onStop,
  onError,
}: {
  busy: boolean;
  onSend: (text: string, image: PreparedImage | null) => void;
  onStop: () => void;
  onError: (message: string) => void;
}) {
  const [value, setValue] = useState('');
  const [image, setImage] = useState<PreparedImage | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Grow with the content, up to a ceiling, then scroll.
  //
  // Measured in a layout effect so it never runs against a half-laid-out box,
  // and re-measured when the textarea's own size changes — the wrap point
  // moves with the width, and a stale measurement leaves the composer stuck
  // at whatever height it happened to compute once.
  const resize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    // An empty textarea measures its PLACEHOLDER, not its value, so a long
    // placeholder that wraps on a narrow screen would leave the composer
    // several lines tall with nothing typed in it. Clearing the inline height
    // hands it back to rows={1}, which is one line whatever the placeholder does.
    if (!el.value) {
      el.style.height = '';
      return;
    }
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  useLayoutEffect(resize, [value, image, resize]);

  useLayoutEffect(() => {
    const row = textareaRef.current?.parentElement;
    if (!row || typeof ResizeObserver === 'undefined') return;
    // The row is observed rather than the textarea itself: resizing the
    // textarea is what this callback does, so watching it would feed back on
    // itself. Only a width change matters — that is what moves the wrap point.
    let lastWidth = row.clientWidth;
    const observer = new ResizeObserver(() => {
      if (row.clientWidth === lastWidth) return;
      lastWidth = row.clientWidth;
      resize();
    });
    observer.observe(row);
    return () => observer.disconnect();
  }, [resize]);

  useEffect(() => {
    if (!busy) textareaRef.current?.focus();
  }, [busy]);

  const clearImage = () => {
    setImage(current => {
      if (current) URL.revokeObjectURL(current.previewUrl);
      return null;
    });
    if (fileRef.current) fileRef.current.value = '';
  };

  const accept = async (file: File | undefined | null) => {
    if (!file) return;
    setPreparing(true);
    try {
      const prepared = await prepareImage(file);
      setImage(current => {
        if (current) URL.revokeObjectURL(current.previewUrl);
        return prepared;
      });
    } catch (err) {
      onError((err as Error).message);
    } finally {
      setPreparing(false);
    }
  };

  const submit = () => {
    const text = value.trim();
    if (busy || preparing || (!text && !image)) return;
    onSend(text, image);
    setValue('');
    // The preview URL now belongs to the sent message, so hand it over rather
    // than revoking it here.
    setImage(null);
    if (fileRef.current) fileRef.current.value = '';
  };

  const canSend = (value.trim().length > 0 || Boolean(image)) && !preparing;

  return (
    <div
      onDragOver={e => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => {
        e.preventDefault();
        setDragging(false);
        void accept(e.dataTransfer.files?.[0]);
      }}
      className={`rounded-2xl border bg-[var(--color-surface)] shadow-[var(--shadow-soft)] transition ${
        dragging ? 'border-[var(--color-leaf-400)]' : 'border-[var(--color-line)]'
      }`}
    >
      {image && (
        <div className="flex items-center gap-3 border-b border-[var(--color-line)] px-3 py-2.5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={image.previewUrl}
            alt="Attached leaf photo"
            className="h-12 w-12 rounded-lg border border-[var(--color-line)] object-cover"
          />
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-[var(--color-ink)]">Photo attached</p>
            <p className="text-[11px] text-[var(--color-ink-faint)]">
              {image.width}×{image.height} · will be read alongside your question
            </p>
          </div>
          <button
            type="button"
            onClick={clearImage}
            className="rounded-lg px-2 py-1 text-xs text-[var(--color-ink-faint)] transition hover:bg-[var(--color-surface-sunken)] hover:text-[var(--color-ink)]"
          >
            Remove
          </button>
        </div>
      )}

      <div className="flex items-end gap-2 p-2">
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="sr-only"
          onChange={e => void accept(e.target.files?.[0])}
        />
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={preparing}
          aria-label="Attach a leaf photo"
          title="Attach a leaf photo"
          className="mb-0.5 shrink-0 rounded-xl p-2.5 text-[var(--color-ink-faint)] transition hover:bg-[var(--color-surface-sunken)] hover:text-[var(--color-ink)] disabled:opacity-40"
        >
          <svg viewBox="0 0 20 20" className="h-[18px] w-[18px]" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2.5" y="3.5" width="15" height="13" rx="2.5" />
            <circle cx="7" cy="8" r="1.4" />
            <path d="M3 13.5l4-3.5 3.5 3 2.5-2 4 3.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>

        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={e => setValue(e.target.value)}
          onPaste={e => {
            const file = Array.from(e.clipboardData.files)[0];
            if (file?.type.startsWith('image/')) {
              e.preventDefault();
              void accept(file);
            }
          }}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          // Kept short so it fits one line at phone width; the empty state
          // above already explains what can be asked.
          placeholder={preparing ? 'Preparing photo…' : 'Ask about a disease…'}
          className="scrollbar-slim max-h-[200px] flex-1 resize-none bg-transparent py-2.5 text-[0.9375rem] leading-6 text-[var(--color-ink)] outline-none placeholder:text-[var(--color-ink-faint)]"
        />

        {busy ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop generating"
            className="mb-0.5 shrink-0 rounded-xl bg-[var(--color-surface-sunken)] p-2.5 text-[var(--color-ink-soft)] transition hover:text-[var(--color-ink)]"
          >
            <svg viewBox="0 0 20 20" className="h-[18px] w-[18px]" fill="currentColor">
              <rect x="5.5" y="5.5" width="9" height="9" rx="1.5" />
            </svg>
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!canSend}
            aria-label="Send"
            className="mb-0.5 shrink-0 rounded-xl bg-[var(--color-leaf-600)] p-2.5 text-white transition disabled:cursor-not-allowed disabled:bg-[var(--color-surface-sunken)] disabled:text-[var(--color-ink-faint)] dark:text-[var(--color-canvas)]"
          >
            <svg viewBox="0 0 20 20" className="h-[18px] w-[18px]" fill="none" stroke="currentColor" strokeWidth="1.7">
              <path d="M10 16V4M10 4l-5 5M10 4l5 5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
