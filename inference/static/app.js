// static/app.js
(async function () {
    // helper selectors
    const qs = s => document.querySelector(s);

    // Image UI
    const imgInput = qs('#image-file');
    const imgFname = qs('#image-fname');
    const imgRun = qs('#image-run');
    const imageOutput = qs('#image-output');
    const imageLabel = qs('#image-label');
    const imageScore = qs('#image-score');
    const imageGradcam = qs('#image-gradcam');
    const imageError = qs('#image-error');

    // Audio UI
    const audInput = qs('#audio-file');
    const audFname = qs('#audio-fname');
    const audRun = qs('#audio-run');
    const audioOutput = qs('#audio-output');
    const audioLabel = qs('#audio-label');
    const audioScore = qs('#audio-score');
    const audioGradcam = qs('#audio-gradcam');
    const audioError = qs('#audio-error');

    // show file name
    imgInput.addEventListener('change', e => {
        const f = e.target.files[0];
        imgFname.textContent = f ? f.name : 'No file chosen';
        imageOutput.classList.add('hide');
        imageError.classList.add('hide');
    });
    audInput.addEventListener('change', e => {
        const f = e.target.files[0];
        audFname.textContent = f ? f.name : 'No file chosen';
        audioOutput.classList.add('hide');
        audioError.classList.add('hide');
    });

    // POST helper
    async function postFile(url, file) {
        const fd = new FormData();
        fd.append('file', file);
        const resp = await fetch(url, { method: 'POST', body: fd });
        if (!resp.ok) {
            const txt = await resp.text();
            throw new Error(`${resp.status} ${resp.statusText}: ${txt}`);
        }
        return resp.json();
    }

    // IMAGE
    imgRun.addEventListener('click', async () => {
        const f = imgInput.files[0];
        imageError.classList.add('hide');
        if (!f) { imageError.textContent = 'Please choose an image file.'; imageError.classList.remove('hide'); return; }

        imgRun.disabled = true; imgRun.textContent = 'Processing...';
        try {
            const data = await postFile('/image-infer', f);
            // label & score
            imageLabel.textContent = data.label ?? 'n/a';
            imageScore.textContent = typeof data.score === 'number' ? data.score.toFixed(4) : data.score;

            // gradcam: prefer b64 if available, else url
            if (data.gradcam_b64) {
                imageGradcam.src = data.gradcam_b64;
            } else if (data.gradcam_url) {
                imageGradcam.src = data.gradcam_url;
            } else {
                imageGradcam.src = '';
            }

            imageOutput.classList.remove('hide');
        } catch (err) {
            imageError.textContent = err.message || 'Inference failed';
            imageError.classList.remove('hide');
        } finally {
            imgRun.disabled = false; imgRun.textContent = 'Run Image Inference';
        }
    });

    // AUDIO
    audRun.addEventListener('click', async () => {
        const f = audInput.files[0];
        audioError.classList.add('hide');
        if (!f) { audioError.textContent = 'Please choose an audio file.'; audioError.classList.remove('hide'); return; }

        audRun.disabled = true; audRun.textContent = 'Processing...';
        try {
            const data = await postFile('/audio-infer', f);
            audioLabel.textContent = data.label ?? 'n/a';
            audioScore.textContent = typeof data.score === 'number' ? data.score.toFixed(4) : data.score;

            if (data.gradcam_url) {
                // audio gradcam is served as a static image; set src
                audioGradcam.src = data.gradcam_url;
            } else {
                audioGradcam.src = '';
            }

            audioOutput.classList.remove('hide');
        } catch (err) {
            audioError.textContent = err.message || 'Inference failed';
            audioError.classList.remove('hide');
        } finally {
            audRun.disabled = false; audRun.textContent = 'Run Audio Inference';
        }
    });

})();