export default function Footer() {
  return (
    <footer className="bg-white border-t mt-8">
      <div className="container text-center py-4 text-sm text-gray-500">
        © {new Date().getFullYear()} DeepfakeDetect — For research/demo only
      </div>
    </footer>
  );
}
