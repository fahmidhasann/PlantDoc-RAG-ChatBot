'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  Send,
  Sparkles,
  BookOpen,
  Image as ImageIcon,
  X,
  AlertCircle,
  Leaf,
  CheckCircle2,
  RefreshCw,
  HelpCircle,
  ShieldCheck
} from 'lucide-react';

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
I am your **Multimodal Plant Pathology Assistant**, powered by **Gemini 3.8 Flash** and grounded in **Agrios' Plant Pathology (5th Edition)** via serverless vector retrieval.

**How to use:**
1. 📸 **Upload a diseased leaf photo** for visual diagnostic symptom extraction.
2. 💬 **Ask any pathology or agronomy question** (symptoms, life-cycle, or control measures).
3. 📖 Every diagnosis includes **direct textbook page citations** to eliminate hallucinations.`,
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
    <div className="flex flex-col min-h-screen">
      {/* Top Navbar */}
      <header className="sticky top-0 z-30 border-b border-emerald-900/40 bg-black/40 backdrop-blur-md px-4 sm:px-8 py-3.5">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-green-400 flex items-center justify-center shadow-lg shadow-emerald-500/20 text-white font-bold text-xl">
              🌱
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-lg text-emerald-100 tracking-tight">PlantDoc</h1>
                <span className="text-[10px] uppercase font-semibold tracking-wider px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  Multimodal RAG
                </span>
              </div>
              <p className="text-xs text-emerald-400/80">Agrios Plant Pathology 5th Ed. • Gemini 3.8 Flash • Pinecone</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/40 text-xs text-emerald-300">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>100% Free Serverless</span>
            </div>
            <a
              href="https://github.com/fahmidhasann/PlantDoc-RAG-ChatBot"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs px-3 py-1.5 rounded-lg border border-emerald-700/50 hover:bg-emerald-900/30 text-emerald-200 transition"
            >
              GitHub Repo →
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
                <div className="w-8 h-8 rounded-lg bg-emerald-800/40 border border-emerald-700/30 flex items-center justify-center text-emerald-300 text-sm flex-shrink-0 mt-1">
                  🌿
                </div>
              )}

              <div
                className={`max-w-[85%] sm:max-w-[78%] rounded-2xl p-4 sm:p-5 text-sm leading-relaxed ${
                  msg.sender === 'user'
                    ? 'bg-emerald-600 text-white rounded-tr-none shadow-md shadow-emerald-950'
                    : 'bg-emerald-950/40 border border-emerald-800/40 text-emerald-100 rounded-tl-none shadow-sm'
                }`}
              >
                {/* Uploaded User Image Thumbnail */}
                {msg.image && (
                  <div className="mb-3 overflow-hidden rounded-xl border border-emerald-400/30 max-w-xs">
                    <img src={msg.image} alt="User plant leaf" className="w-full object-cover max-h-56" />
                  </div>
                )}

                {/* Message Body */}
                <div className="whitespace-pre-wrap font-sans space-y-2 prose-invert">
                  {msg.text}
                </div>

                {/* Sources & Citations */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-emerald-800/50 text-xs">
                    <div className="flex items-center gap-1.5 font-semibold text-emerald-300 mb-2">
                      <BookOpen className="w-3.5 h-3.5" />
                      <span>Verified Textbook Citations</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {msg.sources.map((src, idx) => (
                        <div
                          key={idx}
                          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-900/40 border border-emerald-700/40 text-emerald-200"
                        >
                          <span className="font-bold text-emerald-400">P. {src.page}</span>
                          <span className="text-emerald-400/50">•</span>
                          <span className="truncate max-w-[160px]">{src.topic || src.chapter}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div
                  className={`mt-2 text-[10px] ${
                    msg.sender === 'user' ? 'text-emerald-200/70 text-right' : 'text-emerald-400/60'
                  }`}
                >
                  {msg.timestamp}
                </div>
              </div>

              {msg.sender === 'user' && (
                <div className="w-8 h-8 rounded-lg bg-emerald-700 flex items-center justify-center text-white text-xs font-semibold flex-shrink-0 mt-1">
                  You
                </div>
              )}
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-3.5 justify-start">
              <div className="w-8 h-8 rounded-lg bg-emerald-800/40 border border-emerald-700/30 flex items-center justify-center text-emerald-300 text-sm flex-shrink-0">
                🌿
              </div>
              <div className="bg-emerald-950/40 border border-emerald-800/40 rounded-2xl rounded-tl-none p-4 text-sm text-emerald-300 flex items-center gap-3">
                <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
                <span>Consulting textbook index & generating diagnosis with Gemini 3.8 Flash...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Error Banner */}
        {errorMessage && (
          <div className="mb-4 p-3 rounded-xl bg-red-950/50 border border-red-800/50 text-red-200 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <p className="flex-1">{errorMessage}</p>
            <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-200">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Sample Prompt Chips (Visible when few messages) */}
        {messages.length <= 2 && (
          <div className="mb-3">
            <p className="text-xs text-emerald-400/70 mb-2 flex items-center gap-1.5 font-medium">
              <Sparkles className="w-3 h-3 text-emerald-400" />
              <span>Suggested pathology queries:</span>
            </p>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_PROMPTS.map((prompt, i) => (
                <button
                  key={i}
                  onClick={() => handleSubmit(prompt)}
                  disabled={isLoading}
                  className="text-xs px-3 py-1.5 rounded-lg bg-emerald-950/50 border border-emerald-800/40 text-emerald-300 hover:bg-emerald-900/40 hover:border-emerald-600 transition disabled:opacity-50 text-left"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Selected Image Preview */}
        {selectedImage && (
          <div className="mb-3 p-2.5 rounded-xl bg-emerald-950/60 border border-emerald-700/50 flex items-center justify-between max-w-sm">
            <div className="flex items-center gap-3 overflow-hidden">
              <img
                src={`data:${selectedMimeType};base64,${selectedImage}`}
                alt="Selected preview"
                className="w-12 h-12 rounded-lg object-cover border border-emerald-600/40"
              />
              <div className="truncate text-xs">
                <p className="font-medium text-emerald-200">Leaf Image Selected</p>
                <p className="text-[11px] text-emerald-400/70">Ready for Multimodal diagnosis</p>
              </div>
            </div>
            <button
              onClick={handleRemoveImage}
              className="p-1 rounded-md text-emerald-400 hover:text-white hover:bg-emerald-900/60"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Input Bar */}
        <div className="sticky bottom-4 z-20">
          <form
            onSubmit={e => {
              e.preventDefault();
              handleSubmit();
            }}
            className="flex items-center gap-2 p-2 rounded-2xl bg-emerald-950/90 border border-emerald-700/50 shadow-2xl backdrop-blur-lg focus-within:border-emerald-500 transition"
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
              className="p-2.5 rounded-xl text-emerald-300 hover:bg-emerald-900/60 hover:text-white transition flex items-center justify-center flex-shrink-0"
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
                  ? 'Ask a specific question about this leaf, or press Send for diagnosis...'
                  : 'Ask a plant disease question or upload a leaf photo...'
              }
              className="flex-1 bg-transparent px-2 text-sm text-emerald-100 placeholder-emerald-500/60 focus:outline-none"
              disabled={isLoading}
            />

            {/* Send Button */}
            <button
              type="submit"
              disabled={isLoading || (!inputQuery.trim() && !selectedImage)}
              className="p-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-green-500 text-white font-medium hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center justify-center shadow-md shadow-emerald-900/40 flex-shrink-0"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="mt-2 text-center text-[11px] text-emerald-500/60">
            PlantDoc RAG v2 • Agrios Plant Pathology (5th Ed.) • Zero-cold start on Vercel Serverless
          </div>
        </div>
      </main>
    </div>
  );
}
