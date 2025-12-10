import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="bg-white p-6 rounded shadow">
      <h1 className="text-2xl font-bold">Deepfake Detection</h1>
      <p className="text-gray-600 mt-1">
        Upload images or audio and get Real/Fake predictions with Grad-CAM.
      </p>

      <div className="flex gap-3 mt-4">
        <Link to="/image" className="bg-blue-600 text-white px-4 py-2 rounded">
          Image Inference
        </Link>
        <Link to="/audio" className="border px-4 py-2 rounded">
          Audio Inference
        </Link>
      </div>
    </div>
  );
}
