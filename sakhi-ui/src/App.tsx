import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Scale,
  User,
  Bot,
  AlertCircle,
  Sparkles,
  Shield,
  MessageCircle,
} from "lucide-react";

interface ChatMessage {
  id: string;
  role: "user" | "sakhi";
  content: string;
}

const SUGGESTIONS = [
  { icon: Shield, text: "What are my rights if police stops me?" },
  { icon: MessageCircle, text: "My landlord won't return my deposit" },
  { icon: Scale, text: "Can my employer fire me without notice?" },
];

const SakhiApp: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "greeting",
      role: "sakhi",
      content:
        "Namaste 🙏 I am **Sakhi**, your AI legal companion.\n\nI'm here to help you understand your rights and the law in simple, clear language — without judgment, without jargon.\n\nHow can I assist you today?",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 128) + "px";
    }
  }, [inputValue]);

  const sendMessage = async (query: string) => {
    if (!query.trim()) return;

    const newUserMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: query.trim(),
    };

    setMessages((prev) => [...prev, newUserMsg]);
    setInputValue("");
    setIsLoading(true);
    setError(null);

    try {
      const chatHistory = [...messages, newUserMsg]
        .filter((msg) => msg.id !== "greeting")
        .map((msg) => ({
          role: msg.role === "sakhi" ? "assistant" : "user",
          content: msg.content,
        }));

      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          chat_history: chatHistory,
        }),
      });

      if (!response.ok) throw new Error(`API error: ${response.status}`);

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "sakhi",
          content: data.answer,
        },
      ]);
    } catch (err: any) {
      console.error(err);
      setError(
        "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputValue);
  };

  const showSuggestions = messages.length <= 1 && !isLoading;

  return (
    <div className="flex flex-col h-screen animated-bg text-slate-200">
      {/* ── Header ────────────────────────────────────────────── */}
      <header className="glass sticky top-0 z-20 px-5 py-3.5 flex items-center gap-3">
        <div className="avatar-shimmer p-2.5 rounded-xl shadow-lg">
          <Scale size={22} className="text-white drop-shadow" />
        </div>
        <div className="flex-1">
          <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-1.5">
            Sakhi
            <Sparkles size={14} className="text-amber-400 opacity-80" />
          </h1>
          <p className="text-[11px] text-slate-400 font-medium tracking-wide uppercase">
            AI Legal Companion • India
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] text-emerald-400 font-medium">
            Online
          </span>
        </div>
      </header>

      {/* ── Chat Area ─────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto px-4 py-6 sm:px-6 flex flex-col items-center">
        <div className="w-full max-w-2xl flex flex-col gap-5">
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 16, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{
                  duration: 0.35,
                  ease: [0.16, 1, 0.3, 1],
                }}
                className={`flex w-full ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`flex max-w-[88%] sm:max-w-[80%] gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
                >
                  {/* Avatar */}
                  <div
                    className={`flex-shrink-0 h-8 w-8 rounded-lg flex items-center justify-center mt-1 shadow-md transition-transform hover:scale-110
                    ${
                      msg.role === "user"
                        ? "bg-gradient-to-br from-violet-500 to-indigo-600"
                        : "avatar-shimmer"
                    }`}
                  >
                    {msg.role === "user" ? (
                      <User size={15} className="text-white" />
                    ) : (
                      <Bot size={15} className="text-white" />
                    )}
                  </div>

                  {/* Bubble */}
                  <div
                    className={`px-4 py-3.5 rounded-2xl transition-all
                    ${
                      msg.role === "user"
                        ? "bg-gradient-to-br from-violet-600 to-indigo-700 text-white rounded-tr-sm shadow-lg shadow-indigo-500/20"
                        : "glass-light text-slate-200 rounded-tl-sm glow-saffron"
                    }`}
                  >
                    {msg.role === "user" ? (
                      <p className="text-[14.5px] leading-relaxed whitespace-pre-wrap font-normal">
                        {msg.content}
                      </p>
                    ) : (
                      <div className="prose prose-sm prose-dark max-w-none prose-p:leading-relaxed prose-headings:text-white prose-headings:font-semibold prose-a:text-amber-400 prose-strong:text-slate-100 prose-li:my-0.5 marker:text-amber-500 prose-code:text-amber-300 prose-code:bg-black/20 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}

            {/* Loading */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex w-full justify-start"
              >
                <div className="flex gap-2.5 max-w-[80%]">
                  <div className="flex-shrink-0 h-8 w-8 rounded-lg flex items-center justify-center mt-1 avatar-shimmer shadow-md">
                    <Bot size={15} className="text-white" />
                  </div>
                  <div className="glass-light px-5 py-4 rounded-2xl rounded-tl-sm flex items-center gap-2">
                    {[0, 0.2, 0.4].map((delay) => (
                      <motion.div
                        key={delay}
                        className="typing-dot"
                        animate={{
                          y: [0, -6, 0],
                          opacity: [0.4, 1, 0.4],
                        }}
                        transition={{
                          repeat: Infinity,
                          duration: 1.2,
                          delay,
                          ease: "easeInOut",
                        }}
                      />
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {/* Error */}
            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex w-full justify-center my-2"
              >
                <div className="flex items-center gap-2 text-red-300 bg-red-500/10 border border-red-500/20 px-4 py-2.5 rounded-xl text-sm font-medium backdrop-blur-sm">
                  <AlertCircle size={16} />
                  <p>{error}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Suggestions ───────────────────────────────────── */}
          {showSuggestions && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.4 }}
              className="flex flex-col gap-2.5 mt-2"
            >
              <p className="text-xs text-slate-500 font-medium uppercase tracking-wider pl-1">
                Try asking...
              </p>
              {SUGGESTIONS.map(({ icon: Icon, text }) => (
                <button
                  key={text}
                  onClick={() => sendMessage(text)}
                  className="glass-light text-left px-4 py-3 rounded-xl text-sm text-slate-300 hover:text-white hover:border-amber-500/30 transition-all duration-200 flex items-center gap-3 group"
                >
                  <Icon
                    size={16}
                    className="text-amber-500/70 group-hover:text-amber-400 transition-colors flex-shrink-0"
                  />
                  <span>{text}</span>
                </button>
              ))}
            </motion.div>
          )}

          <div ref={messagesEndRef} className="h-4" />
        </div>
      </main>

      {/* ── Input ─────────────────────────────────────────────── */}
      <footer className="glass sticky bottom-0 z-20 px-4 py-3 border-t border-white/5">
        <div className="max-w-2xl mx-auto w-full">
          <form
            onSubmit={handleSubmit}
            className="relative flex items-end glass-light rounded-2xl glow-input transition-all duration-300"
          >
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask Sakhi about your rights..."
              className="w-full bg-transparent resize-none outline-none max-h-32 min-h-[52px] py-3.5 pl-4 pr-14 text-slate-200 placeholder-slate-500 text-[14.5px] leading-relaxed"
              rows={1}
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="absolute right-2 bottom-2 p-2.5 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 text-white hover:from-orange-400 hover:to-amber-400 disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500 transition-all duration-200 shadow-lg shadow-orange-500/20 hover:shadow-orange-500/30 disabled:shadow-none flex items-center justify-center"
            >
              <Send size={17} />
            </button>
          </form>
          <p className="text-center text-[10.5px] text-slate-600 mt-2.5 font-medium">
            Sakhi is an AI legal companion. Not a substitute for professional
            legal advice.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default SakhiApp;
