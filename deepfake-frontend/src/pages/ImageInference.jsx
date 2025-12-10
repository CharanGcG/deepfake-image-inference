import React, { useState, useEffect } from "react";
import FileDrop from "../components/FileDrop";
import ImagePreview from "../components/ImagePreview";
import GradcamViewer from "../components/GradcamViewer";
import api from "../api/client";

export default function ImageInference() {
  const [file, setFile] = useState(null);
  const [objectUrl, setObjectUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function predict() {
    if (!file) return setError("Please upload an image.");

    setError(null);
    setResult(null);
    setLoading(true);

    const form = new FormData();
    form.append("file", file);

    try {
      const response = await api.post("/image-infer", form, {
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((e.loaded / e.total) * 100));
        },
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white p-6 rounded shadow">
      <h2 className="text-xl font-bold">Image Inference</h2>

      <div className="mt-4">
        <FileDrop accept="image/*" onFile={setFile} />
        <ImagePreview file={file} objectUrl={objectUrl} />
      </div>

      <button
        onClick={predict}
        disabled={loading}
        className="mt-4 bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? `Uploading… ${progress}%` : "Predict"}
      </button>

      {error && <div className="text-red-600 mt-3">{error}</div>}

      {result && (
        <div className="mt-6 bg-gray-100 p-4 rounded">
          <div className="text-lg font-semibold">
            Result:{" "}
            <span
              className={`px-2 py-1 rounded ${
                result.label === "real"
                  ? "bg-green-200 text-green-900"
                  : "bg-red-200 text-red-900"
              }`}
            >
              {result.label.toUpperCase()}
            </span>
          </div>
          <div className="text-sm mt-1 text-gray-600">
            Confidence: {(result.score * 100).toFixed(2)}%
          </div>

          <GradcamViewer
            gradcamB64={result.gradcam_b64}
            gradcamUrl={result.gradcam_url}
          />
        </div>
      )}
    </div>
  );
}
