'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  Send,
  Sparkles,
  BookOpen,
  X,
  AlertCircle,
  RefreshCw,
  FileCheck2,
  ExternalLink
} from 'lucide-react';
import { FormattedResponse } from '@/components/FormattedResponse';

interface Source {
  page: number;
  chapter: string;
  topic: string;
}

interface Message {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  image?: string;
  sources?: Source[];
  timestamp: string;
}

const SAMPLE_PROMPTS = [
  'How do I identify and manage Late Blight in tomato?',
  'What causes bird’s-eye spots on tomato fruit?',
  'Difference between downy mildew and powdery mildew?',
  'What fungicide protocols work best against Rice Blast?'
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: `### Welcome to PlantDoc AI 🌱
I am your **Multimodal Plant Pathology Assistant**, powered by **Gemini** and grounded in **Agrios' Plant Pathology (5th Edition)** via serverless vector retrieval.

**What you can do:**
* 📸 **Upload a diseased leaf photo** for visual symptom diagnosis.
* 💬 **Ask any pathology question** regarding symptoms, pathogen life-cycles, or chemical and cultural control measures.
* 📖 Every response includes **direct textbook page citations** to eliminate hallucinations.`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  const [inputQuery, setInputQuery] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedMimeType, setSelectedMimeType] = useState<string>('image/jpeg');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setErrorMessage('Please select a valid image file (JPEG, PNG, WebP).');
      return;
    }

    if (file.size > 8 * 1024 * 1024) {
      setErrorMessage('Image size should be less than 8MB.');
      return;
    }

    setSelectedMimeType(file.type);
    const reader = new FileReader();
    reader.onload = () => {
      const base64String = (reader.result as string).split(',')[1];
      setSelectedImage(base64String);
      setErrorMessage(null);
    };
    reader.readAsDataURL(file);
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async (queryText?: string) => {
    const textToSend = (queryText || inputQuery).trim();
    if (!textToSend && !selectedImage) return;

    setErrorMessage(null);
    const userMsgId = Date.now().toString();
    const newMsg: Message = {
      id: userMsgId,
      sender: 'user',
      text: textToSend,
      image: selectedImage ? `data:${selectedMimeType};base64,${selectedImage}` : undefined,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, newMsg]);
    setInputQuery('');
    const currentImage = selectedImage;
    const currentMime = selectedMimeType;
    setSelectedImage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: textToSend,
          image: currentImage,
          mimeType: currentMime
        })
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to analyze plant pathology request.');
      }

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: data.answer,
        sources: data.sources,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      setErrorMessage(err.message || 'Network connection failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 text-slate-800">
      {/* Top Navbar (Clean Light Theme) */}
      <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/85 backdrop-blur-md px-4 sm:px-8 py-3.5 shadow-xs">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-green-500 flex items-center justify-center shadow-md shadow-emerald-500/20 text-white font-bold text-xl">
              🌱
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-lg text-slate-900 tracking-tight">PlantDoc</h1>
                <span className="text-[10px] uppercase font-semibold tracking-wider px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200">
                  Multimodal RAG
                </span>
              </div>
              <p className="text-xs text-slate-500">Agrios Plant Pathology 5th Ed. • Gemini Flash • Pinecone</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>100% Free Serverless</span>
            </div>
            <a
              href="https://github.com/fahmidhasann/PlantDoc-RAG-ChatBot"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-100 text-slate-700 transition flex items-center gap-1.5 shadow-2xs"
            >
              <span>GitHub</span>
              <ExternalLink className="w-3 h-3 text-slate-400" />
            </a>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-6 flex flex-col">
        {/* Messages Feed */}
        <div className="flex-1 space-y-6 mb-4">
          {messages.map(msg => (
            <div
              key={msg.id}
              className={`flex gap-3.5 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.sender === 'assistant' && (
                <div className="w-9 h-9 rounded-xl bg-emerald-100 border border-emerald-200 flex items-center justify-center text-emerald-800 text-sm font-bold flex-shrink-0 mt-1 shadow-2xs">
                  🌿
                </div>
              )}

              <div
                className={`max-w-[88%] sm:max-w-[80%] rounded-2xl p-5 ${
                  msg.sender === 'user'
                    ? 'bg-emerald-600 text-white rounded-tr-none shadow-md shadow-emerald-600/10'
                    : 'bg-white border border-slate-200/90 rounded-tl-none shadow-sm shadow-slate-200/40'
                }`}
              >
                {/* Uploaded User Image Thumbnail */}
                {msg.image && (
                  <div className="mb-3 overflow-hidden rounded-xl border border-slate-200 max-w-xs shadow-xs">
                    <img src={msg.image} alt="User plant leaf" className="w-full object-cover max-h-56" />
                  </div>
                )}

                {/* Message Body with Rich Markdown & Formatting */}
                {msg.sender === 'user' ? (
                  <div className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
                    {msg.text}
                  </div>
                ) : (
                  <FormattedResponse content={msg.text} />
                )}

                {/* Sources & Citations Drawer */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-5 pt-3.5 border-t border-slate-100">
                    <div className="flex items-center gap-1.5 font-semibold text-slate-700 text-xs mb-2.5">
                      <BookOpen className="w-3.5 h-3.5 text-emerald-600" />
                      <span>Verified Textbook Citations</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {msg.sources.map((src, idx) => (
                        <div
                          key={idx}
                          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200 text-slate-700 text-xs shadow-2xs hover:bg-slate-100 transition"
                        >
                          <span className="font-bold text-emerald-700">P. {src.page}</span>
                          <span className="text-slate-300">•</span>
                          <span className="truncate max-w-[180px] font-medium">{src.topic || src.chapter}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div
                  className={`mt-2.5 text-[10px] ${
                    msg.sender === 'user' ? 'text-emerald-100 text-right' : 'text-slate-400'
                  }`}
                >
                  {msg.timestamp}
                </div>
              </div>

              {msg.sender === 'user' && (
                <div className="w-9 h-9 rounded-xl bg-slate-800 flex items-center justify-center text-white text-xs font-semibold flex-shrink-0 mt-1 shadow-2xs">
                  You
                </div>
              )}
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-3.5 justify-start">
              <div className="w-9 h-9 rounded-xl bg-emerald-100 border border-emerald-200 flex items-center justify-center text-emerald-800 text-sm font-bold flex-shrink-0 shadow-2xs">
                🌿
              </div>
              <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-none p-4 text-sm text-slate-600 flex items-center gap-3 shadow-xs">
                <RefreshCw className="w-4 h-4 animate-spin text-emerald-600" />
                <span>Consulting textbook index & synthesizing diagnosis...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Error Banner */}
        {errorMessage && (
          <div className="mb-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2 shadow-2xs">
            <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
            <p className="flex-1 font-medium">{errorMessage}</p>
            <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-700">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Sample Prompt Chips (Visible initially) */}
        {messages.length <= 2 && (
          <div className="mb-4">
            <p className="text-xs text-slate-500 mb-2 flex items-center gap-1.5 font-medium">
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
              <span>Suggested pathology queries:</span>
            </p>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_PROMPTS.map((prompt, i) => (
                <button
                  key={i}
                  onClick={() => handleSubmit(prompt)}
                  disabled={isLoading}
                  className="text-xs px-3.5 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 hover:border-emerald-400 hover:text-emerald-700 hover:bg-emerald-50/40 transition disabled:opacity-50 text-left shadow-2xs font-medium"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Selected Image Preview */}
        {selectedImage && (
          <div className="mb-3 p-2.5 rounded-xl bg-white border border-emerald-300 shadow-sm flex items-center justify-between max-w-sm">
            <div className="flex items-center gap-3 overflow-hidden">
              <img
                src={`data:${selectedMimeType};base64,${selectedImage}`}
                alt="Selected preview"
                className="w-12 h-12 rounded-lg object-cover border border-slate-200 shadow-2xs"
              />
              <div className="truncate text-xs">
                <p className="font-semibold text-slate-800">Leaf Image Selected</p>
                <p className="text-[11px] text-emerald-600">Ready for visual diagnosis</p>
              </div>
            </div>
            <button
              onClick={handleRemoveImage}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Input Bar (Clean White Card) */}
        <div className="sticky bottom-4 z-20">
          <form
            onSubmit={e => {
              e.preventDefault();
              handleSubmit();
            }}
            className="flex items-center gap-2 p-2 rounded-2xl bg-white border border-slate-300 shadow-xl shadow-slate-200/80 focus-within:border-emerald-500 focus-within:ring-3 focus-within:ring-emerald-500/10 transition"
          >
            {/* Hidden File Input */}
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              onChange={handleImageSelect}
              className="hidden"
            />

            {/* Image Upload Button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              title="Upload leaf photo for diagnosis"
              className="p-2.5 rounded-xl text-slate-500 hover:bg-emerald-50 hover:text-emerald-700 transition flex items-center justify-center flex-shrink-0"
            >
              <UploadCloud className="w-5 h-5" />
            </button>

            {/* Text Input */}
            <input
              type="text"
              value={inputQuery}
              onChange={e => setInputQuery(e.target.value)}
              placeholder={
                selectedImage
                  ? 'Ask a specific question about this leaf, or press Send...'
                  : 'Ask a plant disease question or upload a leaf photo...'
              }
              className="flex-1 bg-transparent px-2 text-sm text-slate-800 placeholder-slate-400 focus:outline-none"
              disabled={isLoading}
            />

            {/* Send Button */}
            <button
              type="submit"
              disabled={isLoading || (!inputQuery.trim() && !selectedImage)}
              className="p-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-medium disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center justify-center shadow-sm flex-shrink-0"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="mt-2 text-center text-[11px] text-slate-400">
            PlantDoc RAG v2 • Agrios Plant Pathology (5th Ed.) • Zero cold-start on Vercel Serverless
          </div>
        </div>
      </main>
    </div>
  );
}
