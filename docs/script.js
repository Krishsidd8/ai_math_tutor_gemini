const API_BASE_URL = 'https://aimathtutorgemini-production.up.railway.app';

const darkModeToggle = document.getElementById('darkModeToggle');
const body = document.body;
const imageInput = document.getElementById('imageInput');
const solveBtn = document.getElementById('solveBtn');
const chatSection = document.getElementById('chatSection');
const uploadBox = document.getElementById('uploadBox');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');

let uploadedImageURL = null;

function updateToggleText() {
  darkModeToggle.textContent = body.classList.contains('dark-mode')
    ? '☀️ Light Mode'
    : '🌙 Dark Mode';
}

darkModeToggle.addEventListener('click', () => {
  body.classList.toggle('dark-mode');
  updateToggleText();
});
updateToggleText();

uploadBox.addEventListener('click', () => imageInput.click());

uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
  uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    imageInput.files = files;
    imageInput.dispatchEvent(new Event('change'));
  }
});

imageInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    alert("Please upload a valid image file.");
    imageInput.value = "";
    return;
  }

  const maxSizeMB = 5;
  if (file.size > maxSizeMB * 1024 * 1024) {
    alert(`File size exceeds ${maxSizeMB}MB. Please upload a smaller image.`);
    imageInput.value = "";
    return;
  }

  const reader = new FileReader();
  reader.onload = function(e) {
    uploadedImageURL = e.target.result;
    imagePreviewContainer.innerHTML = `<img src="${uploadedImageURL}" alt="Uploaded Preview" />`;
  };
  reader.readAsDataURL(file);
});

solveBtn.addEventListener('click', () => {
  const file = imageInput.files[0];

  if (!file) {
    alert("Please upload an image.");
    return;
  }

  if (!file.type.startsWith("image/")) {
    alert("The selected file is not an image.");
    return;
  }

  const maxSizeMB = 5;
  if (file.size > maxSizeMB * 1024 * 1024) {
    alert(`File size exceeds ${maxSizeMB}MB.`);
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  solveBtn.disabled = true;
  solveBtn.textContent = "Solving...";

  fetch(`${API_BASE_URL}/solve`, {
    method: "POST",
    body: formData,
  })
    .then(res => res.json())
    .then(data => {
      console.log("Backend response:", data);

      if (data.error) {
        alert("Error from backend: " + data.error);
        return;
      }

      if (!data.steps || !Array.isArray(data.steps)) {
        alert("No steps found in backend response. Full response: " + JSON.stringify(data));
        return;
      }

      const botMsg = document.createElement('div');
      botMsg.className = 'chat-message bot';
      botMsg.innerHTML = `
        <strong>Predicted LaTeX:</strong> ${data.latex || "N/A"}<br/>
        <strong>Step-by-Step Solution:</strong>
        <ol>${data.steps.map(s => `<li>${s.step || ""}<br/><code>${s.detail || ""}</code></li>`).join('')}</ol>
      `;
      chatSection.appendChild(botMsg);
      chatSection.scrollTop = chatSection.scrollHeight;
    })
    .catch(err => {
      alert("Failed to solve: " + err);
    })
    .finally(() => {
      solveBtn.disabled = false;
      solveBtn.textContent = "Solve";
    });
});