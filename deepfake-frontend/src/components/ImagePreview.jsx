export default function ImagePreview({ file, objectUrl }) {
  if (!file) return null;

  return (
    <div className="mt-4 bg-white shadow-sm rounded p-4">
      <img
        src={objectUrl}
        alt={file.name}
        className="w-40 h-40 object-contain border rounded"
      />

      <div className="mt-2">
        <div className="font-medium">{file.name}</div>
        <div className="text-sm text-gray-500">
          {(file.size / 1024).toFixed(1)} KB
        </div>
      </div>
    </div>
  );
}
