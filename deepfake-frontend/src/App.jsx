import React from "react";
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

import Home from "./pages/Home";
import ImageInference from "./pages/ImageInference";
import AudioInference from "./pages/AudioInference";
import Models from "./pages/Models";
import GradcamGuide from "./pages/GradcamGuide";
import HowTo from "./pages/HowTo";
import About from "./pages/About";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="container flex-grow">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/image" element={<ImageInference />} />
          <Route path="/audio" element={<AudioInference />} />
          <Route path="/models" element={<Models />} />
          <Route path="/gradcam" element={<GradcamGuide />} />
          <Route path="/how" element={<HowTo />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>

      <Footer />
    </div>
  );
}
