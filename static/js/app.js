/* global Chart */
'use strict';

// ── Cached DOM references ──────────────────────────────────────────────────
const form        = document.getElementById('analyze-form');
const analyzeBtn  = document.getElementById('analyze-btn');
const btnText     = document.getElementById('btn-text');
const btnSpinner  = document.getElementById('btn-spinner');
const errorBox    = document.getElementById('error-box');
const resultsSection = document.getElementById('results-section');

// File-drop labels
const inputFileEl = document.getElementById('input-audio');
const refFileEl   = document.getElementById('reference-audio');
const labelInput  = document.getElementById('label-input');
const labelRef    = document.getElementById('label-ref');

// Mic-mode elements
const tabUpload       = document.getElementById('tab-upload');
const tabMic          = document.getElementById('tab-mic');
const panelUpload     = document.getElementById('panel-upload');
const panelMic        = document.getElementById('panel-mic');
const btnRecordStart  = document.getElementById('btn-record-start');
const btnRecordStop   = document.getElementById('btn-record-stop');
const micDot          = document.getElementById('mic-dot');
const micStatusText   = document.getElementById('mic-status-text');
const micPlayback     = document.getElementById('mic-playback');
const micAudio        = document.getElementById('mic-audio');
const micFileInput    = document.getElementById('mic-file-input');

// Chart instances (kept so we can destroy & recreate on new analysis)
let freqChart = null;
let waveChart = null;

// ── Input-mode state ───────────────────────────────────────────────────────
let micMode = false;           // true → use mic recording as input_audio
let recordedWavFile = null;    // File object set after recording stops

// ── Tab switching ──────────────────────────────────────────────────────────
tabUpload.addEventListener('click', () => {
  micMode = false;
  tabUpload.classList.add('tab-active');
  tabUpload.setAttribute('aria-selected', 'true');
  tabMic.classList.remove('tab-active');
  tabMic.setAttribute('aria-selected', 'false');
  panelUpload.classList.remove('hidden');
  panelMic.classList.add('hidden');
  // Restore "required" on the file input so the browser validates it
  inputFileEl.required = true;
});

tabMic.addEventListener('click', () => {
  micMode = true;
  tabMic.classList.add('tab-active');
  tabMic.setAttribute('aria-selected', 'true');
  tabUpload.classList.remove('tab-active');
  tabUpload.setAttribute('aria-selected', 'false');
  panelMic.classList.remove('hidden');
  panelUpload.classList.add('hidden');
  // Remove "required" – we validate manually in the submit handler
  inputFileEl.required = false;
});

// ── File label helpers ─────────────────────────────────────────────────────
function setupFileLabel(inputEl, labelEl) {
  inputEl.addEventListener('change', () => {
    if (inputEl.files.length > 0) {
      labelEl.textContent = inputEl.files[0].name;
      labelEl.classList.add('has-file');
    } else {
      labelEl.innerHTML = 'Drop file or click to browse<br/><small>.wav · .mp3 · .flac · .ogg</small>';
      labelEl.classList.remove('has-file');
    }
  });
}

setupFileLabel(inputFileEl, labelInput);
setupFileLabel(refFileEl,   labelRef);

// Drag-over visual feedback
['drop-input', 'drop-ref'].forEach(id => {
  const zone = document.getElementById(id);
  if (!zone) return;
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('dragover'));
  zone.addEventListener('drop',      ()  => zone.classList.remove('dragover'));
});

// ── Loading state ──────────────────────────────────────────────────────────
function setLoading(loading) {
  analyzeBtn.disabled = loading;
  btnText.textContent  = loading ? 'Analyzing…' : 'Analyze';
  btnSpinner.classList.toggle('hidden', !loading);
}

// ── Error display ──────────────────────────────────────────────────────────
function showError(msg) {
  errorBox.textContent = `⚠ ${msg}`;
  errorBox.classList.remove('hidden');
}
function clearError() {
  errorBox.textContent = '';
  errorBox.classList.add('hidden');
}

// ── Form submit ────────────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearError();
  resultsSection.classList.add('hidden');

  // Mic-mode validation
  if (micMode && !recordedWavFile) {
    showError('Please record audio from the microphone first.');
    return;
  }

  setLoading(true);

  const formData = new FormData(form);

  // In mic mode, replace (or add) the input_audio field with the recorded WAV
  if (micMode && recordedWavFile) {
    formData.set('input_audio', recordedWavFile, recordedWavFile.name);
  }

  try {
    const res  = await fetch('/api/analyze', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || `Server error: ${res.status}`);
      return;
    }

    renderResults(data);
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    showError(`Network error: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

// ── Render all results ─────────────────────────────────────────────────────
function renderResults(data) {
  renderScore(data.overall_score, data.quality_rating);
  renderFeedback(data.feedback);
  renderMetrics(data);
  renderFreqChart(data.frequency);
  renderWaveChart(data.time_domain);
}

// ── Score gauge ────────────────────────────────────────────────────────────
function renderScore(score, rating) {
  const rounded = Math.round(score);
  document.getElementById('score-text').textContent = rounded;

  // Gauge arc: total arc ≈ 251.2 px (half-circle, r=80)
  const arcLen = 251.2;
  const filled = (score / 100) * arcLen;
  const gaugeFill = document.getElementById('gauge-fill');
  gaugeFill.setAttribute('stroke-dasharray', `${filled} ${arcLen - filled}`);

  // Colour the gauge fill based on score
  gaugeFill.style.stroke =
    score >= 80 ? '#48c78e' :
    score >= 60 ? '#53d8fb' :
    score >= 40 ? '#ffbd00' : '#e94560';

  // Quality badge
  const badge = document.getElementById('quality-badge');
  badge.textContent = rating;
  badge.className   = `quality-badge badge-${rating.toLowerCase()}`;
}

// ── Feedback bullets ───────────────────────────────────────────────────────
function renderFeedback(feedback) {
  const list = document.getElementById('feedback-list');
  list.innerHTML = '';
  feedback.forEach(msg => {
    const li = document.createElement('li');
    li.textContent = msg;
    list.appendChild(li);
  });
}

// ── Metric cards ───────────────────────────────────────────────────────────
function renderMetrics(data) {
  const snr = data.snr;
  document.getElementById('m-snr').textContent       = `${snr.snr_db.toFixed(1)} dB`;
  document.getElementById('m-snr-rating').textContent = snr.quality_rating;

  const dr = data.dynamic_range;
  document.getElementById('m-dr').textContent  = `${dr.input_dynamic_range_db.toFixed(1)} dB`;
  document.getElementById('m-dr-sub').textContent =
    dr.overcompression_detected ? '⚠ Over-compressed' : 'Normal';

  const ph = data.phase;
  document.getElementById('m-phase').textContent  = ph.phase_coherence.toFixed(3);
  document.getElementById('m-phase-sub').textContent =
    `Delay: ${ph.estimated_delay_ms.toFixed(1)} ms`;

  const sp = data.spectral_similarity;
  document.getElementById('m-spec').textContent  = sp.mfcc_cosine_similarity.toFixed(3);
  document.getElementById('m-spec-sub').textContent =
    `MFCC dist: ${sp.mfcc_distance.toFixed(2)}`;

  const xc = data.cross_correlation;
  document.getElementById('m-xcorr').textContent  = xc.max_correlation.toFixed(3);
  document.getElementById('m-xcorr-sub').textContent =
    `Lag: ${xc.lag_ms.toFixed(1)} ms`;
}

// ── Chart helpers ──────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 600 },
  plugins: {
    legend: { labels: { color: '#8892a4', font: { size: 11 } } },
    tooltip: { mode: 'index', intersect: false },
  },
  scales: {
    x: {
      ticks: { color: '#8892a4', maxTicksLimit: 10 },
      grid:  { color: 'rgba(255,255,255,.05)' },
    },
    y: {
      ticks: { color: '#8892a4' },
      grid:  { color: 'rgba(255,255,255,.05)' },
    },
  },
};

function destroyChart(chartRef) {
  if (chartRef) { chartRef.destroy(); }
  return null;
}

// ── Frequency chart ────────────────────────────────────────────────────────
function renderFreqChart(freqData) {
  freqChart = destroyChart(freqChart);

  // Downsample to at most 512 points for chart performance
  const step     = Math.max(1, Math.floor(freqData.input_freqs.length / 512));
  const freqs    = freqData.input_freqs.filter((_, i) => i % step === 0);
  const inputMag = freqData.input_magnitudes.filter((_, i) => i % step === 0);
  const refMag   = freqData.ref_magnitudes.filter((_, i) => i % step === 0);

  // Convert to dB for display
  const eps  = 1e-10;
  const toDb = v => 20 * Math.log10(v + eps);

  const ctx = document.getElementById('freq-chart').getContext('2d');
  freqChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: freqs.map(f => f.toFixed(0)),
      datasets: [
        {
          label: 'Input',
          data:  inputMag.map(toDb),
          borderColor:  '#e94560',
          backgroundColor: 'rgba(233,69,96,.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'Reference',
          data:  refMag.map(toDb),
          borderColor:  '#53d8fb',
          backgroundColor: 'rgba(83,216,251,.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        ...CHART_DEFAULTS.scales,
        x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Frequency (Hz)', color: '#8892a4' } },
        y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'Magnitude (dB)', color: '#8892a4' } },
      },
    },
  });
}

// ── Waveform chart ─────────────────────────────────────────────────────────
function renderWaveChart(tdData) {
  waveChart = destroyChart(waveChart);

  // Further downsample to 1024 points for waveform
  const targetLen = 1024;
  function downsample(arr) {
    if (arr.length <= targetLen) return arr;
    const step = Math.floor(arr.length / targetLen);
    return arr.filter((_, i) => i % step === 0).slice(0, targetLen);
  }

  const inputWave = downsample(tdData.input_waveform);
  const refWave   = downsample(tdData.ref_waveform);
  const labels    = inputWave.map((_, i) => i);

  const ctx = document.getElementById('wave-chart').getContext('2d');
  waveChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Input',
          data:  inputWave,
          borderColor:  '#e94560',
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
        },
        {
          label: 'Reference',
          data:  refWave,
          borderColor:  '#53d8fb',
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        ...CHART_DEFAULTS.scales,
        x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Sample', color: '#8892a4' } },
        y: {
          ...CHART_DEFAULTS.scales.y,
          title: { display: true, text: 'Amplitude', color: '#8892a4' },
          min: -1,
          max:  1,
        },
      },
    },
  });
}

// ══════════════════════════════════════════════════════════════════════════
// MICROPHONE RECORDING
// Uses the Web Audio API to capture raw PCM, then encodes to 16-bit WAV so
// that the server can process it without needing any extra codecs.
// ══════════════════════════════════════════════════════════════════════════

let mediaStream      = null;  // MediaStream from getUserMedia
let audioContext     = null;  // AudioContext used for PCM capture
let scriptProcessor  = null;  // ScriptProcessorNode
let pcmChunks        = [];    // Float32Array samples collected during recording

/**
 * Encode an array of Float32 PCM samples as a 16-bit mono WAV Blob.
 * @param {Float32Array[]} chunks  - raw PCM chunks (mono, same sample rate)
 * @param {number}         sampleRate
 * @returns {Blob}
 */
function encodeWav(chunks, sampleRate) {
  // Flatten all chunks into one buffer
  const totalLen = chunks.reduce((n, c) => n + c.length, 0);
  const pcm = new Float32Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    pcm.set(chunk, offset);
    offset += chunk.length;
  }

  // Convert float32 → int16
  const int16 = new Int16Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  // Build WAV file
  const dataBytes   = int16.buffer.byteLength;
  const buffer      = new ArrayBuffer(44 + dataBytes);
  const view        = new DataView(buffer);
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate    = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign  = numChannels * (bitsPerSample / 8);

  const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
  writeStr(0,  'RIFF');
  view.setUint32(4,  36 + dataBytes, true);
  writeStr(8,  'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);           // chunk size
  view.setUint16(20, 1,  true);           // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, 'data');
  view.setUint32(40, dataBytes, true);
  new Int16Array(buffer, 44).set(int16);

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Stop and release the current media stream and audio context.
 */
function releaseAudioResources() {
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor.onaudioprocess = null;
    scriptProcessor = null;
  }
  if (audioContext) {
    audioContext.close().catch((err) => { console.warn('AudioContext close error:', err); });
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
}

// ── Start recording ────────────────────────────────────────────────────────
btnRecordStart.addEventListener('click', async () => {
  clearError();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Microphone access is not supported in this browser or context. Please use HTTPS.');
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    showError(`Microphone access denied: ${err.message}`);
    return;
  }

  // Set up audio context
  audioContext    = new (window.AudioContext || window.webkitAudioContext)();
  const source    = audioContext.createMediaStreamSource(mediaStream);
  // ScriptProcessorNode is deprecated but remains the only option without a
  // separate AudioWorklet JS file. It has universal browser support for now.
  const bufSize   = 4096;
  scriptProcessor = audioContext.createScriptProcessor(bufSize, 1, 1);
  pcmChunks       = [];

  scriptProcessor.onaudioprocess = (evt) => {
    // Copy the input buffer so it doesn't get recycled
    const data = evt.inputBuffer.getChannelData(0);
    pcmChunks.push(new Float32Array(data));
  };

  source.connect(scriptProcessor);
  scriptProcessor.connect(audioContext.destination);

  // Update UI
  micDot.classList.add('recording');
  micStatusText.textContent = 'Recording…';
  btnRecordStart.classList.add('hidden');
  btnRecordStop.classList.remove('hidden');
  micPlayback.classList.add('hidden');
  recordedWavFile = null;
});

// ── Stop recording ─────────────────────────────────────────────────────────
btnRecordStop.addEventListener('click', () => {
  if (!audioContext) return;

  const sampleRate = audioContext.sampleRate;
  releaseAudioResources();

  // Encode to WAV
  const wavBlob = encodeWav(pcmChunks, sampleRate);
  recordedWavFile = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });

  // Playback preview
  micAudio.src = URL.createObjectURL(wavBlob);
  micPlayback.classList.remove('hidden');

  // Update UI
  micDot.classList.remove('recording');
  micStatusText.textContent = 'Recording complete';
  btnRecordStop.classList.add('hidden');
  btnRecordStart.classList.remove('hidden');
  btnRecordStart.textContent = '⏺ Record Again';
});
