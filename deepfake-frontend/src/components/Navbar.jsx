import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const loc = useLocation();

  const linkClass = (path) =>
    `px-3 py-2 rounded hover:bg-gray-100 ${
      loc.pathname === path ? "bg-gray-200 font-semibold" : ""
    }`;

  return (
    <nav className="bg-white shadow">
      <div className="container flex justify-between items-center py-3">
        <Link to="/" className="text-xl font-bold">
          DeepfakeDetect
        </Link>

        <div className="flex gap-2">
          <Link className={linkClass("/")} to="/">Home</Link>
          <Link className={linkClass("/image")} to="/image">Image</Link>
          <Link className={linkClass("/audio")} to="/audio">Audio</Link>
          <Link className={linkClass("/models")} to="/models">Models</Link>
          <Link className={linkClass("/gradcam")} to="/gradcam">GradCAM</Link>
          <Link className={linkClass("/how")} to="/how">How To</Link>
          <Link className={linkClass("/about")} to="/about">About</Link>
        </div>
      </div>
    </nav>
  );
}
