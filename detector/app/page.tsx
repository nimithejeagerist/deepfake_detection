"use client";

import { useRef, useState } from "react";
import { Sono } from "next/font/google";

const sono = Sono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export default function DeepfakeDetector() {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ label: string; confidence: number } | null>(null);
  const resetUpload = () => {
  setFile(null);
  setPreview(null);
  setResult(null);
  };


  const handleSelect = (f: File | null) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  };

  const handleDetect = async () => {
    if (!file) return;

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("video", file);

    try {
      const res = await fetch("http://localhost:8000/detect", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      className={`min-h-screen flex items-center justify-center bg-neutral-950 ${sono.className}`}
    >
      <div className="w-full max-w-2xl px-6">
        <div
          className="relative rounded-2xl border border-neutral-800 bg-neutral-900/70 
                     backdrop-blur-xl shadow-2xl p-8 space-y-8
                     transition-all duration-300 hover:border-neutral-700"
        >
          {/* Header */}
          <div className="space-y-2">
            <h1 className="text-3xl font-semibold text-neutral-100">
              Deepfake Detector
            </h1>
            <p className="text-sm text-neutral-400">
              Upload a video and run it through a trained detection model.
            </p>
          </div>

          {/* File input */}
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            hidden
            onChange={(e) => handleSelect(e.target.files?.[0] ?? null)}
          />

          {!file && (
            <button
              onClick={() => inputRef.current?.click()}
              className="group w-full rounded-xl border border-dashed border-neutral-700 
                         py-10 text-neutral-400 transition-all
                         hover:border-neutral-500 hover:text-neutral-200"
            >
              <span className="block text-sm font-medium">
                Click to upload a video
              </span>
              <span className="mt-1 block text-xs opacity-70 group-hover:opacity-100">
                MP4, MOV, or WebM
              </span>
            </button>
          )}

          {/* Preview */}
          {preview && (
            <div className="space-y-4">
              <video
                src={preview}
                controls
                className="w-full rounded-xl border border-neutral-800"
              />
          
              <div className="flex gap-3">
                <button
                  onClick={handleDetect}
                  disabled={loading}
                  className="flex-1 rounded-xl bg-neutral-100 py-3 text-sm font-medium text-neutral-900
                             transition-all hover:bg-white hover:shadow
                             disabled:opacity-50"
                >
                  {loading ? "Running detection…" : "Run detection"}
                </button>
          
                <button
                  onClick={resetUpload}
                  disabled={loading}
                  className="rounded-xl border border-neutral-700 px-4 text-sm text-neutral-400
                             transition-all hover:border-neutral-500 hover:text-neutral-200"
                >
                  Change
                </button>
              </div>
            </div>
          )}


          {/* Result */}
          {result && (
            <div
              className="pt-6 border-t border-neutral-800 space-y-3
                         transition-all duration-500"
            >
              <div className="flex items-center justify-between">
                <span className="text-sm text-neutral-400">
                  Detection result
                </span>
                <span
                  className={`text-sm font-semibold tracking-wide
                    ${result.label === "fake" ? "text-red-400" : "text-emerald-400"}`}
                >
                  {result.label.toUpperCase()}
                </span>
              </div>

              <p className="text-sm text-neutral-300">
                Confidence{" "}
                <span className="font-medium text-neutral-100">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
