"use client";

import { useRef, useState } from "react";

export default function DeepfakeDetector() {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ label: string; confidence: number } | null>(null);

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
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-100 to-slate-200">
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-xl p-8 space-y-6">
        <h1 className="text-2xl font-semibold text-gray-800">
          Deepfake Detector
        </h1>

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
            className="w-full py-3 rounded-lg border border-dashed border-gray-400 text-gray-600 hover:bg-gray-50"
          >
            Upload a video
          </button>
        )}

        {preview && (
          <video
            src={preview}
            controls
            className="w-full rounded-lg"
          />
        )}

        {file && (
          <button
            onClick={handleDetect}
            disabled={loading}
            className="w-full py-3 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-50"
          >
            {loading ? "Detecting…" : "Detect"}
          </button>
        )}

        {result && (
          <div className="pt-4 border-t">
            <p className="text-lg">
              Result:{" "}
              <span className="font-semibold capitalize">
                {result.label}
              </span>
            </p>
            <p className="text-gray-600">
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
