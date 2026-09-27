const form = document.getElementById('predictForm');
const ccInput = document.getElementById('ccInput');
const mloInput = document.getElementById('mloInput');
const analyseButton = document.getElementById('analyseButton');
const clearButton = document.getElementById('clearButton');
const resultPanel = document.getElementById('resultPanel');
const toast = document.getElementById('toast');
const statusPill = document.getElementById('statusPill');
const statusText = document.getElementById('statusText');

function showToast(message) { toast.textContent = message; toast.hidden = false; clearTimeout(window.__toastTimer); window.__toastTimer = setTimeout(() => toast.hidden = true, 6500); }
function formatPercent(value) { return `${(Number(value) * 100).toFixed(1)}%`; }
function updateButton() { analyseButton.disabled = !(ccInput.files?.length && mloInput.files?.length); }

function setupUpload(input, card, previewId, nameId) {
  const preview = document.getElementById(previewId), fileName = document.getElementById(nameId);
  const empty = card.querySelector('[data-empty]'), previewWrap = card.querySelector('[data-preview]');
  input.addEventListener('change', () => { const file = input.files?.[0]; if (!file) return; const reader = new FileReader(); reader.onload = e => { preview.src = e.target.result; fileName.textContent = file.name; empty.hidden = true; previewWrap.hidden = false; updateButton(); }; reader.readAsDataURL(file); });
}
setupUpload(ccInput, document.getElementById('ccCard'), 'ccPreview', 'ccName');
setupUpload(mloInput, document.getElementById('mloCard'), 'mloPreview', 'mloName');

clearButton.addEventListener('click', () => {
  for (const [input, card, preview] of [[ccInput, document.getElementById('ccCard'), 'ccPreview'], [mloInput, document.getElementById('mloCard'), 'mloPreview']]) {
    input.value = ''; document.getElementById(preview).removeAttribute('src'); card.querySelector('[data-empty]').hidden = false; card.querySelector('[data-preview]').hidden = true;
  }
  resultPanel.hidden = true; updateButton();
});

async function checkHealth() {
  try { const response = await fetch('/api/health'); const data = await response.json();
    if (data.model_available) { statusPill.classList.add('ready'); statusText.textContent = 'Trained model ready'; document.getElementById('modelName').textContent = data.selected_model || 'Selected CNN'; }
    else { statusText.textContent = 'No trained model selected yet'; }
  } catch (_) { statusText.textContent = 'Server unavailable'; }
}
checkHealth();

form.addEventListener('submit', async event => {
  event.preventDefault(); if (!ccInput.files?.[0] || !mloInput.files?.[0]) return;
  const body = new FormData(); body.append('cc_file', ccInput.files[0]); body.append('mlo_file', mloInput.files[0]);
  analyseButton.disabled = true; analyseButton.classList.add('loading'); analyseButton.querySelector('.button-text').textContent = 'Analysing…';
  try {
    const response = await fetch('/api/predict', { method: 'POST', body }); const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || 'Prediction failed.');
    const prediction = String(payload.prediction).toLowerCase(); const badge = document.getElementById('resultBadge'); badge.textContent = prediction; badge.className = `result-badge ${prediction}`;
    document.getElementById('predictedClass').textContent = prediction.charAt(0).toUpperCase() + prediction.slice(1);
    document.getElementById('malignantProbability').textContent = formatPercent(payload.malignant_probability);
    document.getElementById('confidenceValue').textContent = formatPercent(payload.confidence);
    document.getElementById('ccProbability').textContent = formatPercent(payload.cc_probability);
    document.getElementById('mloProbability').textContent = formatPercent(payload.mlo_probability);
    document.getElementById('thresholdLabel').textContent = `Validation threshold ${formatPercent(payload.threshold)}`;
    document.getElementById('meterFill').style.width = formatPercent(payload.malignant_probability);
    document.getElementById('modelName').textContent = payload.selected_model || 'Selected CNN';
    resultPanel.hidden = false; resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) { showToast(error.message || String(error)); }
  finally { analyseButton.classList.remove('loading'); analyseButton.querySelector('.button-text').textContent = 'Analyse pair'; updateButton(); }
});
