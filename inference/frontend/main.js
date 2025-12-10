const SERVER = window.SERVER_BASE || "http://127.0.0.1:8000";
const fileInput = document.getElementById("file-input");
const dropArea = document.getElementById("drop-area");
const uploadBtn = document.getElementById("upload-btn");
const status = document.getElementById("status");
const results = document.getElementById("results");
const origImg = document.getElementById("orig-img");
const gradcamImg = document.getElementById("gradcam-img");
const labelEl = document.getElementById("label");
const scoreEl = document.getElementById("score");
const dlGrad = document.getElementById("download-gradcam");
const dlOrig = document.getElementById("download-orig");
const serverInfo = document.getElementById("server-info");
serverInfo.textContent = SERVER;

let selectedFile = null;

function setStatus(t){ status.textContent = t; }

fileInput.addEventListener("change", (e)=>{
  if(e.target.files && e.target.files[0]){
    handleFile(e.target.files[0]);
  }
});

dropArea.addEventListener("dragover", (e)=>{
  e.preventDefault();
  dropArea.style.borderColor = "#4f46e5";
});
dropArea.addEventListener("dragleave", (e)=>{
  dropArea.style.borderColor = "rgba(255,255,255,0.06)";
});
dropArea.addEventListener("drop", (e)=>{
  e.preventDefault();
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if(f) handleFile(f);
});

function handleFile(file){
  if(!file.type.startsWith("image/")) {
    setStatus("Please select an image file");
    return;
  }
  selectedFile = file;
  uploadBtn.disabled = false;
  // preview original
  const url = URL.createObjectURL(file);
  origImg.src = url;
  dlOrig.href = url;
  dlOrig.download = "original_" + file.name;
  results.hidden = false;
  setStatus("Ready to upload");
  labelEl.textContent = "—";
  scoreEl.textContent = "Score: —";
  gradcamImg.src = "";
}

uploadBtn.addEventListener("click", async ()=>{
  if(!selectedFile) return;
  setStatus("Uploading...");
  uploadBtn.disabled = true;
  try{
    const fd = new FormData();
    fd.append("file", selectedFile);
    const res = await fetch(`${SERVER}/infer`, { method: "POST", body: fd });
    if(!res.ok){
      const txt = await res.text();
      setStatus("Server error: " + res.status + " — " + txt.slice(0,200));
      uploadBtn.disabled = false;
      return;
    }
    const data = await res.json();
    // data: { label, score, gradcam_url }
    const label = data.label || "—";
    const score = (typeof data.score === "number") ? data.score.toFixed(4) : data.score;
    labelEl.textContent = label.toUpperCase();
    labelEl.style.background = label.toLowerCase() === "real" ? "linear-gradient(90deg,#10b981,#059669)" : "linear-gradient(90deg,#ef4444,#f97316)";
    scoreEl.textContent = `Score: ${score}`;
    // Grad-CAM path: server returns path like "/static/gradcam_xxx.png"
        // Show gradcam: prefer gradcam_b64 (safer), fallback to gradcam_url
    let gradUrl = null;
    if (data.gradcam_b64) {
      gradUrl = data.gradcam_b64;
    } else if (data.gradcam_url) {
      gradUrl = data.gradcam_url.startsWith("http") ? data.gradcam_url : (SERVER + data.gradcam_url);
    }

    if (gradUrl) {
      gradcamImg.src = gradUrl;
      dlGrad.href = gradUrl;
      dlGrad.download = "gradcam.png";
    } else {
      gradcamImg.src = "";
      dlGrad.href = "";
    }
    setStatus("Done");
  }catch(err){
    console.error(err);
    setStatus("Upload failed: " + (err.message || err));
  } finally {
    uploadBtn.disabled = false;
  }
});
