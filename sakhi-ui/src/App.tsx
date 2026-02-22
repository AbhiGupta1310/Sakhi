import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Scale, User, Bot, AlertCircle } from "lucide-react";

interface ChatMessage {
  id: string;
  role: "user" | "sakhi";
  content: string;
}

const SakhiApp: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "greeting",
      role: "sakhi",
      content:
        "Namaste 🙏 I am Sakhi, your AI legal companion. I'm here to help you understand your rights and the law in simple, clear language. How can I assist you today?",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const query = inputValue.trim();
    const newUserMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: query,
    };

    setMessages((prev) => [...prev, newUserMsg]);
    setInputValue("");
    setIsLoading(true);
    setError(null);

    try {
      // Build chat_history from all previous messages (excluding the greeting)
      const chatHistory = [...messages, newUserMsg]
        .filter((msg) => msg.id !== "greeting")
        .map((msg) => ({
          role: msg.role === "sakhi" ? "assistant" : "user",
          content: msg.content,
        }));
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, chat_history: chatHistory }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      const newSakhiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "sakhi",
        content: data.answer,
      };

      setMessages((prev) => [...prev, newSakhiMsg]);
    } catch (err: any) {
      console.error(err);
      setError(
        "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 text-slate-800 font-sans">
      {/* Header */}
      <header className="bg-white px-6 py-4 shadow-sm border-b border-slate-200 flex items-center gap-3 z-10 sticky top-0">
        <div className="bg-orange-100 p-2 rounded-full text-orange-600">
          <Scale size={24} />
        </div>
        <div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-orange-600 to-amber-500 bg-clip-text text-transparent">
            Sakhi
          </h1>
          <p className="text-xs text-slate-500 font-medium">
            Your trusted AI legal companion
          </p>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex-1 overflow-y-auto w-full p-4 sm:p-6 lg:p-8 flex flex-col items-center">
        <div className="w-full max-w-3xl flex flex-col gap-6">
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className={`flex w-full ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`flex max-w-[85%] md:max-w-[75%] gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
                >
                  {/* Avatar */}
                  <div
                    className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center mt-1 
                    ${msg.role === "user" ? "bg-indigo-100 text-indigo-600" : "bg-orange-100 text-orange-600"}`}
                  >
                    {msg.role === "user" ? (
                      <User size={16} />
                    ) : (
                      <Bot size={16} />
                    )}
                  </div>

                  {/* Message Bubble */}
                  <div
                    className={`px-5 py-4 rounded-2xl shadow-sm
                    ${
                      msg.role === "user"
                        ? "bg-indigo-600 text-white rounded-tr-sm"
                        : "bg-white border border-slate-100 text-slate-700 rounded-tl-sm"
                    }
                  `}
                  >
                    {msg.role === "user" ? (
                      <p className="text-[15px] leading-relaxed whitespace-pre-wrap">
                        {msg.content}
                      </p>
                    ) : (
                      <div
                        className="prose prose-sm md:prose-base prose-slate max-w-none 
                        prose-headings:text-slate-800 prose-headings:font-bold 
                        prose-a:text-orange-600 prose-a:no-underline hover:prose-a:underline
                        prose-strong:text-slate-900 prose-strong:font-semibold
                        prose-li:my-1 marker:text-orange-500"
                      >
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}

            {/* Loading Indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex w-full justify-start"
              >
                <div className="flex gap-3 flex-row max-w-[75%]">
                  <div className="flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center mt-1 bg-orange-100 text-orange-600">
                    <Bot size={16} />
                  </div>
                  <div className="px-5 py-4 rounded-2xl shadow-sm bg-white border border-slate-100 rounded-tl-sm flex items-center gap-1.5 h-[56px]">
                    <motion.div
                      className="w-2 h-2 rounded-full bg-slate-300"
                      animate={{ y: [0, -5, 0] }}
                      transition={{ repeat: Infinity, duration: 1, delay: 0 }}
                    />
                    <motion.div
                      className="w-2 h-2 rounded-full bg-slate-300"
                      animate={{ y: [0, -5, 0] }}
                      transition={{ repeat: Infinity, duration: 1, delay: 0.2 }}
                    />
                    <motion.div
                      className="w-2 h-2 rounded-full bg-slate-300"
                      animate={{ y: [0, -5, 0] }}
                      transition={{ repeat: Infinity, duration: 1, delay: 0.4 }}
                    />
                  </div>
                </div>
              </motion.div>
            )}

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex w-full justify-center my-2"
              >
                <div className="flex items-center gap-2 text-red-500 bg-red-50 px-4 py-2 rounded-lg text-sm font-medium border border-red-100 shadow-sm">
                  <AlertCircle size={16} />
                  <p>{error}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={messagesEndRef} className="h-4" /> {/* Spacer */}
        </div>
      </main>

      {/* Input Area */}
      <footer className="bg-white border-t border-slate-200 p-4 sticky bottom-0 z-10">
        <div className="max-w-3xl mx-auto w-full">
          <form
            onSubmit={handleSendMessage}
            className="relative flex items-end shadow-sm border border-slate-300 rounded-2xl bg-slate-50 focus-within:ring-2 focus-within:ring-indigo-500 focus-within:border-indigo-500 transition-shadow"
          >
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e);
                }
              }}
              placeholder="Ask Sakhi about your rights, legal procedures, or a specific situation..."
              className="w-full bg-transparent resize-none outline-none max-h-32 min-h-[56px] py-4 pl-5 pr-14 text-slate-700 placeholder-slate-400 text-[15px]"
              rows={1}
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="absolute right-2 bottom-2 p-2.5 rounded-xl bg-orange-600 text-white hover:bg-orange-700 disabled:bg-slate-300 disabled:text-slate-100 transition-colors shadow-sm flex-shrink-0 flex items-center justify-center"
            >
              <Send size={18} />
            </button>
          </form>
          <p className="text-center text-xs text-slate-400 mt-3 font-medium">
            Sakhi is an AI companion to help you understand Indian law. It does
            not replace professional legal advice.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default SakhiApp;
