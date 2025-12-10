import React, { useEffect, useState } from "react";
import FileDrop from "../components/FileDrop";
import AudioPlayer from "../components/AudioPlayer";
import api from "../api/client";

/**
 * Audio Inference page:
 * - upload audio (field name 'file')
 * - preview/play locally via object URL
 * - POST to /audio-infer
 * - show result: label, score, gradcam image (timeline)
 */

const ALLOWED_EXT = [".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"];
const MAX_MB = 40; // client-side limit (adjust if you want)

function getExt(filename = "") {
  const i = filename.lastIndexOf(".");
  return i >= 0 ? filename.slice(i).toLowerCase() : "";
}

export default function AudioInference() {
  const [file, setFile] = useState(null);
  const [objectUrl, setObjectUrl] = useState(null);
  const [durationSec, setDurationSec] = useState(null);

  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // create object URL for playback
  useEffect(() => {
    if (!file) {
      setObjectUrl(null);
      setDurationSec(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => {
      URL.revokeObjectURL(url);
      setObjectUrl(null);
    };
  }, [file]);

  // handle file selection (from FileDrop)
  function handleFileSelected(f) {
    setError(null);
    setResult(null);

    if (!f) {
      setFile(null);
      return;
    }

    // validate extension
    const ext = getExt(f.name);
    if (!ALLOWED_EXT.includes(ext)) {
      setError(`Unsupported audio type: ${ext}. Allowed: ${ALLOWED_EXT.join(", ")}`);
      return;
    }

    // size
    const mb = f.size / (1024 * 1024);
    if (mb > MAX_MB) {
      setError(`File too large (${mb.toFixed(1)} MB). Limit is ${MAX_MB} MB.`);
      return;
    }

    setFile(f);
  }

  async function handlePredict() {
    setError(null);
    setResult(null);

    if (!file) {
      setError("Please upload an audio file first.");
      return;
    }

    const form = new FormData();
    form.append("file", file, file.name);

    try {
      setLoading(true);
      setProgress(0);

      const resp = await api.post("/audio-infer", form, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((e.loaded / e.total) * 100));
        },
      });

      // expected: { label, score, gradcam_url }
      setResult(resp.data);
    } catch (err) {
      console.error(err);
      const msg = err?.response?.data?.detail || err?.message || "Upload failed";
      setError(String(msg));
    } finally {
      setLoading(false);
    }
  }

  const handleAudioLoaded = (secs) => {
    setDurationSec(secs);
  };

  return (
    <div className="bg-white p-6 rounded shadow">
      <h2 className="text-xl font-bold">Audio Inference</h2>
      <p className="text-slate-600 mt-1">
        Upload an audio file. The model will return Real / Fake and a Grad-CAM timeline image.
      </p>

      <div className="mt-4">
        <FileDrop accept="audio/*" onFile={handleFileSelected} />
      </div>

      {file && (
        <div className="mt-4 bg-slate-50 p-3 rounded">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium">{file.name}</div>
              <div className="text-sm text-slate-500">{(file.size / 1024).toFixed(1)} KB</div>
            </div>
            <div>
              <button
                onClick={() => {
                  setFile(null);
                  setResult(null);
                  setError(null);
                }}
                className="px-3 py-1 border rounded"
              >
                Remove
              </button>
            </div>
          </div>

          <div className="mt-3">
            {objectUrl ? (
              <AudioPlayer src={objectUrl} onLoaded={handleAudioLoaded} />
            ) : (
              <div className="text-sm text-slate-500">Audio preview not available</div>
            )}
            <div className="text-xs text-slate-400 mt-2">
              {durationSec ? `Duration: ${Math.round(durationSec)}s` : ""}
            </div>
          </div>
        </div>
      )}

      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={handlePredict}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
        >
          {loading ? `Uploading... ${progress}%` : "Predict"}
        </button>

        <button
          onClick={() => {
            setFile(null);
            setResult(null);
            setError(null);
          }}
          className="px-3 py-2 border rounded"
        >
          Clear
        </button>

        {loading && <div className="text-sm text-slate-500">Upload: {progress}%</div>}
      </div>

      {error && <div className="mt-4 text-red-600">{error}</div>}

      {result && (
        <div className="mt-6 bg-slate-50 p-4 rounded">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-lg font-semibold">
                Result:{" "}
                <span
                  className={`inline-block ml-2 px-2 py-1 rounded ${
                    result.label === "real"
                      ? "bg-emerald-200 text-emerald-800"
                      : "bg-rose-200 text-rose-800"
                  }`}
                >
                  {String(result.label).toUpperCase()}
                </span>
              </div>
              <div className="text-sm text-slate-600 mt-1">
                Confidence: {(result.score * 100).toFixed(2)}%
              </div>
            </div>

            <div className="text-right">
              {result.gradcam_url && (
                <a
                  href={result.gradcam_url}
                  target="_blank"
                  rel="noreferrer"
                  className="underline text-sm"
                >
                  Open Grad-CAM
                </a>
              )}
            </div>
          </div>

          {result.gradcam_url && (
            <div className="mt-4 bg-white p-3 rounded shadow-sm">
              <div className="font-semibold mb-2">Grad-CAM Timeline</div>
              {/* timeline is an image returned from backend */}
              <img
                src={result.gradcam_url}
                alt="Audio Grad-CAM Timeline"
                className="w-full object-contain border rounded"
                style={{ maxHeight: 420 }}
              />
              <div className="mt-2 flex gap-2">
                <a
                  href={result.gradcam_url}
                  download
                  className="px-3 py-1 border rounded text-sm"
                >
                  Download Grad-CAM
                </a>
                <button
                  onClick={() => {
                    // try to open image in new tab
                    window.open(result.gradcam_url, "_blank", "noopener");
                  }}
                  className="px-3 py-1 border rounded text-sm"
                >
                  Open in new tab
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
