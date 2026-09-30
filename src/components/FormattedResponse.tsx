import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface FormattedResponseProps {
  content: string;
}

export function FormattedResponse({ content }: FormattedResponseProps) {
  return (
    <div className="formatted-response text-slate-800 text-sm leading-relaxed space-y-3">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-xl font-bold text-slate-900 mt-4 mb-2 flex items-center gap-2 border-b border-slate-200 pb-1.5">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-lg font-bold text-slate-900 mt-4 mb-2 flex items-center gap-2 border-b border-slate-100 pb-1">
              {children}
            </h2>
          ),
          h3: ({ children }) => {
            const textStr = String(children);
            let badgeColor = "bg-emerald-50 border-emerald-200 text-emerald-900";
            if (textStr.includes("Diagnosis")) {
              badgeColor = "bg-blue-50 border-blue-200 text-blue-900";
            } else if (textStr.includes("Symptom")) {
              badgeColor = "bg-amber-50 border-amber-200 text-amber-900";
            } else if (textStr.includes("Management")) {
              badgeColor = "bg-emerald-50 border-emerald-200 text-emerald-900";
            } else if (textStr.includes("Reference")) {
              badgeColor = "bg-purple-50 border-purple-200 text-purple-900";
            }

            return (
              <div className={`text-sm font-semibold tracking-wide px-3 py-1.5 rounded-lg border mt-5 mb-2.5 flex items-center gap-2 ${badgeColor}`}>
                {children}
              </div>
            );
          },
          h4: ({ children }) => (
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mt-3 mb-1.5">
              {children}
            </h4>
          ),
          p: ({ children }) => (
            <p className="text-slate-700 leading-relaxed mb-2.5 last:mb-0">
              {children}
            </p>
          ),
          ul: ({ children }) => (
            <ul className="space-y-1.5 my-2 pl-2">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal space-y-1.5 my-2 pl-5 text-slate-700">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="flex items-start gap-2 text-slate-700">
              <span className="text-emerald-500 font-bold mt-1 text-xs select-none">•</span>
              <span className="flex-1">{children}</span>
            </li>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-slate-900">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic font-serif text-slate-800">
              {children}
            </em>
          ),
          blockquote: ({ children }) => (
            <div className="border-l-4 border-emerald-400 bg-emerald-50/60 p-3 rounded-r-lg my-2 text-slate-700 text-xs italic">
              {children}
            </div>
          ),
          code: ({ children }) => (
            <code className="bg-slate-100 border border-slate-200 text-slate-800 px-1.5 py-0.5 rounded text-xs font-mono">
              {children}
            </code>
          ),
          hr: () => <hr className="border-t border-slate-200/80 my-4" />
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
