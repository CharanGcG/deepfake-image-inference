import React, { useRef, useState, useEffect } from "react";

/**
 * Simple audio player wrapper using native <audio>.
 * Props:
 * - src: object URL or remote URL
 * - onLoaded (durationSeconds) optional
 */
export default function AudioPlayer({ src, onLoaded }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [pos, setPos] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;

    const onTime = () => setPos(a.currentTime);
    const onLoadedMeta = () => {
      setDuration(a.duration || 0);
      if (onLoaded) onLoaded(a.duration || 0);
    };
    const onEnded = () => setPlaying(false);

    a.addEventListener("timeupdate", onTime);
    a.addEventListener("loadedmetadata", onLoadedMeta);
    a.addEventListener("ended", onEnded);

    return () => {
      a.removeEventListener("timeupdate", onTime);
      a.removeEventListener("loadedmetadata", onLoadedMeta);
      a.removeEventListener("ended", onEnded);
    };
  }, [src, onLoaded]);

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    if (playing) {
      // try play
      const p = a.play();
      if (p && p.catch) p.catch(() => setPlaying(false));
    } else {
      a.pause();
    }
  }, [playing]);

  const togglePlay = () => setPlaying((p) => !p);

  const onSeek = (e) => {
    const a = audioRef.current;
    const val = Number(e.target.value);
    if (!a || isNaN(val)) return;
    a.currentTime = val;
    setPos(val);
  };

  const formatTime = (s) => {
    if (!isFinite(s)) return "00:00";
    const mm = Math.floor(s / 60)
      .toString()
      .padStart(2, "0");
    const ss = Math.floor(s % 60)
      .toString()
      .padStart(2, "0");
    return `${mm}:${ss}`;
  };

  return (
    <div className="bg-white p-3 rounded shadow-sm">
      <audio ref={audioRef} src={src} preload="metadata" />
      <div className="flex items-center gap-3">
        <button
          aria-label={playing ? "Pause" : "Play"}
          onClick={togglePlay}
          className="px-3 py-1 rounded bg-slate-100 hover:bg-slate-200"
        >
          {playing ? "Pause" : "Play"}
        </button>

        <div className="flex-1">
          <input
            type="range"
            min={0}
            max={duration || 0}
            value={pos}
            onChange={onSeek}
            step="0.01"
            className="w-full"
          />
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <div>{formatTime(pos)}</div>
            <div>{formatTime(duration)}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
