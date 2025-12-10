export default function GradcamViewer({ gradcamB64, gradcamUrl }) {
  const src = gradcamB64 || gradcamUrl;
  if (!src) return null;

  return (
    <div className="mt-6 bg-white p-4 shadow rounded">
      <div className="font-semibold mb-2">Grad-CAM Output</div>
      <img src={src} alt="GradCAM" className="w-full rounded border" />
    </div>
  );
}
