import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'PlantDoc — Multimodal Plant Pathology AI',
  description: 'AI-powered plant disease diagnosis and RAG system trained on Agrios Plant Pathology textbook with page citations.',
  icons: {
    icon: 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌱</text></svg>',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased bg-slate-50 text-slate-900 selection:bg-emerald-500/20 selection:text-emerald-900">
        {children}
      </body>
    </html>
  );
}
