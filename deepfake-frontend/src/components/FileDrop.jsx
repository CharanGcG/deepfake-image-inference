import React, { useRef } from "react";

export default function FileDrop({ accept, onFile }) {
  const inputRef = useRef();

  const handleSelect = (files) => {
    if (files && files[0]) onFile(files[0]);
  };

  return (
    <div
      className="border-2 border-dashed border-gray-300 p-6 rounded text-center bg-white cursor-pointer"
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        handleSelect(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handleSelect(e.target.files)}
      />

      <div className="text-gray-600">Drag & drop or click to select file</div>
      <div className="text-xs text-gray-400 mt-1">{accept}</div>
    </div>
  );
}
