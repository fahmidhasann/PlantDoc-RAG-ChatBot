/**
 * The corpus the app is grounded in.
 *
 * Swapping books means re-running ingestion and changing these two env vars —
 * nothing in the retrieval or chat code knows what the book is about.
 */
export const BOOK = {
  title: process.env.NEXT_PUBLIC_BOOK_TITLE || 'Plant Pathology',
  subtitle: process.env.NEXT_PUBLIC_BOOK_SUBTITLE || 'Agrios, 5th edition',
};

/** Opening prompts. Deliberately varied: definition, comparison, diagnosis, control. */
export const STARTER_QUESTIONS = [
  'How do I identify and manage late blight in tomato?',
  'What is the difference between downy mildew and powdery mildew?',
  'Explain the disease triangle and why it matters in the field.',
  'Which fungicide groups are at highest risk of resistance, and why?',
];
