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

function renderMathInElement(element) {
  if (window.MathJax) {
    MathJax.typesetPromise([element]).catch(err => {
      console.error('MathJax typeset failed:', err.message);
    });
  }
}

function addBotMessage(htmlContent) {
  const botMsg = document.createElement('div');
  botMsg.className = 'chat-message bot';
  botMsg.innerHTML = htmlContent;
  chatSection.appendChild(botMsg);
  renderMathInElement(botMsg);
  chatSection.scrollTop = chatSection.scrollHeight;
}

function cleanLatex(latex) {
  if (!latex) return "";
  latex = latex.trim();

  latex = latex.replace(/^'''|'''$/g, '')
               .replace(/^"""/g, '')
               .replace(/"""$/g, '')
               .replace(/^'/, '')
               .replace(/'$/, '')
               .replace(/^"/, '')
               .replace(/"$/, '');

  return latex.trim();
}

uploadBox.addEventListener('click', () => imageInput.click());
uploadBox.addEventListener('dragover', (e) => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    imageInput.files = files;
    imageInput.dispatchEvent(new Event('change'));
  }
});

imageInput.addEventListener('change', async (event) => {
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

  imagePreviewContainer.innerHTML = '';
  const prevPreviewMessages = chatSection.querySelectorAll('.chat-message.predicted-preview');
  prevPreviewMessages.forEach(msg => msg.remove());

  const reader = new FileReader();
  reader.onload = function(e) {
    uploadedImageURL = e.target.result;
    imagePreviewContainer.innerHTML = `<img src="${uploadedImageURL}" alt="Uploaded Preview" />`;
  };
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE_URL}/predict/`, { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      addBotMessage(`Error from backend: ${data.error}`);
      return;
    }

    if (data.latex) {
      const cleaned = cleanLatex(data.latex);
      const htmlContent = `<strong>Predicted Equation (Preview):</strong><br>$$${cleaned}$$`;
      const botMsg = document.createElement('div');
      botMsg.className = 'chat-message bot predicted-preview';
      botMsg.innerHTML = htmlContent;
      chatSection.appendChild(botMsg);
      renderMathInElement(botMsg);
      chatSection.scrollTop = chatSection.scrollHeight;
    }

  } catch (err) {
    addBotMessage(`Failed to get equation preview: ${err.message}`);
  }
});

solveBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) {
    alert("Please upload an image.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  solveBtn.disabled = true;
  solveBtn.textContent = "Solving...";

  try {
    const res = await fetch(`${API_BASE_URL}/solve`, { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      addBotMessage(`Error from backend: ${data.error}`);
      return;
    }

    if (data.latex) {
      const cleanedLatex = cleanLatex(data.latex);
      addBotMessage(`<strong>Predicted Equation:</strong><br>$$${cleanedLatex}$$`);
    } else {
      addBotMessage(`<strong>Predicted Equation:</strong> N/A`);
    }

    if (data.steps && Array.isArray(data.steps)) {
      const stepsHtml = data.steps.map(s => {
        const mathjaxRaw = s.mathjax || s.detail || "";
        const mathjax = cleanLatex(mathjaxRaw);
        return `<li><strong>${s.step || ""}:</strong><br>$$${mathjax}$$</li>`;
      }).join('');
      addBotMessage(`<strong>Step-by-Step Solution:</strong><ol>${stepsHtml}</ol>`);
    }

    if (data.note) {
      addBotMessage(`<em>Note: ${data.note}</em>`);
    }

  } catch (err) {
    addBotMessage(`Failed to solve: ${err.message}`);
  } finally {
    solveBtn.disabled = false;
    solveBtn.textContent = "Solve";
  }
});
