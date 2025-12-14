// ============================================
// RLHF News Summarization - Frontend Logic
// Updated to use modular summarization pipeline
// ============================================

const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const textInput = document.getElementById('textInput');
const urlInput = document.getElementById('urlInput');
const fileInput = document.getElementById('fileInput');
const fileUploadArea = document.getElementById('fileUploadArea');
const fileSelected = document.getElementById('fileSelected');
const fileName = document.getElementById('fileName');
const fileRemove = document.getElementById('fileRemove');
const instructionsInput = document.getElementById('instructionsInput');
const summarizeButton = document.getElementById('summarizeButton');
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

// State
let isProcessing = false;
let currentTab = 'text';
let selectedFile = null;

// ============================================
// Tab Switching
// ============================================

tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        const tab = button.dataset.tab;
        switchTab(tab);
    });
});

function switchTab(tab) {
    currentTab = tab;

    // Update tab buttons
    tabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Update tab contents
    tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tab}-tab`);
    });

    // Update button state
    updateButtonState();
}

// ============================================
// Input Validation
// ============================================

textInput.addEventListener('input', updateButtonState);
urlInput.addEventListener('input', updateButtonState);

function updateButtonState() {
    let hasInput = false;

    if (currentTab === 'text') {
        hasInput = textInput.value.trim().length > 0;
    } else if (currentTab === 'url') {
        hasInput = urlInput.value.trim().length > 0;
    } else if (currentTab === 'file') {
        hasInput = selectedFile !== null;
    }

    summarizeButton.disabled = !hasInput || isProcessing;
}

// ============================================
// File Upload Handling
// ============================================

fileUploadArea.addEventListener('click', () => {
    if (!isProcessing) {
        fileInput.click();
    }
});

fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('dragover');
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('dragover');
});

fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove('dragover');

    if (e.dataTransfer.files.length > 0) {
        // Handle multiple files from drag and drop
        handleFilesSelect(e.dataTransfer.files);
    }
});

// File input handling
fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleFilesSelect(files);
    }
});

fileRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFileSelection();
});

// New function to handle multiple files
function handleFilesSelect(files) {
    let allValid = true;
    for (const file of files) {
        // Validate file type
        const validTypes = ['.pdf', '.docx', '.doc', '.txt'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();

        if (!validTypes.includes(fileExt)) {
            alert(`File "${file.name}": Please upload a PDF, DOCX, or TXT file`);
            allValid = false;
            break;
        }

        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            alert(`File "${file.name}": File size must be less than 10MB`);
            allValid = false;
            break;
        }
    }

    if (!allValid) {
        clearFileSelection(); // Clear any partially selected files if validation fails
        return;
    }

    // If all files are valid, update display
    fileUploadArea.querySelector('.file-upload-content').style.display = 'none';
    fileSelected.style.display = 'flex';

    // Display file names
    if (files.length === 1) {
        fileName.textContent = files[0].name;
        selectedFile = files[0]; // Keep selectedFile for single file backward compatibility
    } else {
        fileName.textContent = `${files.length} files selected`;
        selectedFile = files[0]; // Still set selectedFile to the first file for consistency, though fileInput.files is primary
    }

    updateButtonState();
}

// Original handleFileSelect (now unused by fileInput.addEventListener, but kept for reference or other potential uses)
function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['.pdf', '.docx', '.doc', '.txt'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(fileExt)) {
        alert('Please upload a PDF, DOCX, or TXT file');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSelected.style.display = 'flex';
    fileUploadArea.querySelector('.file-upload-content').style.display = 'none';

    updateButtonState();
}

function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    fileSelected.style.display = 'none';
    fileUploadArea.querySelector('.file-upload-content').style.display = 'flex';
    updateButtonState();
}

// ============================================
// Main Summarization Handler
// ============================================

summarizeButton.addEventListener('click', handleSummarize);

async function handleSummarize() {
    if (isProcessing) return;

    isProcessing = true;
    summarizeButton.disabled = true;

    const instructions = instructionsInput.value.trim() || null;
    const loadingId = addLoadingMessage();

    try {
        let result;
        let userMessage = '';

        if (currentTab === 'text') {
            result = await summarizeText(textInput.value.trim(), instructions);
            userMessage = `Summarize text (${textInput.value.length} characters)`;
        } else if (currentTab === 'url') {
            const urlText = urlInput.value.trim();
            // Parse URLs - split by newlines and filter empty lines
            const urls = urlText.split('\n').map(u => u.trim()).filter(u => u.length > 0);

            if (urls.length === 0) {
                throw new Error('Please enter at least one URL');
            } else if (urls.length === 1) {
                // Single URL
                result = await summarizeURL(urls[0], instructions);
                userMessage = `Summarize URL: ${urls[0]}`;
            } else {
                // Multiple URLs
                result = await summarizeMultipleURLs(urls, instructions);
                userMessage = `Summarize ${urls.length} URLs`;
            }
        } else if (currentTab === 'file') {
            // Check if multiple files selected
            const files = fileInput.files;
            if (files.length === 0) {
                throw new Error('Please select at least one file');
            } else if (files.length === 1) {
                // Single file
                result = await summarizeFile(files[0], instructions);
                userMessage = `Summarize file: ${files[0].name}`;
            } else {
                // Multiple files
                result = await summarizeMultipleFiles(files, instructions);
                userMessage = `Summarize ${files.length} files`;
            }
        }

        // Add instructions to user message if provided
        if (instructions) {
            userMessage += `\nInstructions: ${instructions}`;
        }

        addUserMessage(userMessage);

        removeMessage(loadingId);
        addSummaryMessage(result);

        // Clear inputs
        if (currentTab === 'text') {
            textInput.value = '';
        } else if (currentTab === 'url') {
            urlInput.value = '';
        } else if (currentTab === 'file') {
            clearFileSelection();
        }
        instructionsInput.value = '';

    } catch (error) {
        console.error('Error:', error);
        removeMessage(loadingId);
        addErrorMessage(error.message);
    } finally {
        isProcessing = false;
        updateButtonState();
    }
}

// ============================================
// API Calls
// ============================================

async function summarizeText(text, instructions) {
    const response = await fetch(`${API_BASE_URL}/summarize/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, instructions })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to summarize text');
    }

    return await response.json();
}

async function summarizeURL(url, instructions) {
    const response = await fetch(`${API_BASE_URL}/summarize/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, instructions })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to summarize URL');
    }

    return await response.json();
}

async function summarizeFile(file, instructions) {
    const formData = new FormData();
    formData.append('file', file);
    if (instructions) {
        formData.append('instructions', instructions);
    }

    const response = await fetch(`${API_BASE_URL}/summarize/file`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to summarize file');
    }

    return await response.json();
}

// Summarize multiple URLs
async function summarizeMultipleURLs(urls, instructions = null) {
    const response = await fetch(`${API_BASE_URL}/summarize/urls`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            urls: urls,
            instructions: instructions,
            combine: true
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to summarize URLs');
    }

    return await response.json();
}

// Summarize multiple files
async function summarizeMultipleFiles(files, instructions = null) {
    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    if (instructions) {
        formData.append('instructions', instructions);
    }

    formData.append('combine', 'true');

    const response = await fetch(`${API_BASE_URL}/summarize/files`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to summarize files');
    }

    return await response.json();
}


// ============================================
// Message Display Functions
// ============================================

function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'U';

    const content = document.createElement('div');
    content.className = 'message-content';

    const header = document.createElement('div');
    header.className = 'message-header';

    const author = document.createElement('span');
    author.className = 'message-author';
    author.textContent = 'You';

    header.appendChild(author);

    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    messageText.innerHTML = `<p>${escapeHtml(text)}</p>`;

    content.appendChild(header);
    content.appendChild(messageText);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addSummaryMessage(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';

    const content = document.createElement('div');
    content.className = 'message-content';

    const header = document.createElement('div');
    header.className = 'message-header';

    const author = document.createElement('span');
    author.className = 'message-author';
    author.textContent = 'RLHF Assistant';

    const badge = document.createElement('span');
    badge.className = 'message-badge';
    badge.textContent = 'DPO-Optimized';

    header.appendChild(author);
    header.appendChild(badge);

    const messageText = document.createElement('div');
    messageText.className = 'message-text';

    // Summary
    const summary = document.createElement('div');
    summary.innerHTML = '<strong>Summary:</strong>';

    // Convert newlines to paragraphs
    const summaryText = data.summary;
    const paragraphs = summaryText.split('\n\n');

    paragraphs.forEach(para => {
        if (para.trim()) {
            const p = document.createElement('p');
            p.textContent = para.trim();
            summary.appendChild(p);
        }
    });

    messageText.appendChild(summary);

    // Metadata
    if (data.title) {
        const title = document.createElement('p');
        title.innerHTML = `<strong>Title:</strong> ${escapeHtml(data.title)}`;
        messageText.appendChild(title);
    }

    if (data.author) {
        const author = document.createElement('p');
        author.innerHTML = `<strong>Author:</strong> ${escapeHtml(data.author)}`;
        messageText.appendChild(author);
    }

    // Metrics
    const metrics = document.createElement('div');
    metrics.className = 'metrics';

    const metricsData = [
        { label: 'Chunks', value: data.num_chunks || 1 },
        { label: 'Input', value: `${data.input_length || 0} chars` },
        { label: 'Summary', value: `${data.summary_length || 0} chars` },
        { label: 'Compression', value: `${((1 - (data.summary_length / data.input_length)) * 100).toFixed(0)}%` }
    ];

    metricsData.forEach(metric => {
        const metricItem = document.createElement('div');
        metricItem.className = 'metric-item';
        metricItem.innerHTML = `
            <span class="metric-label">${metric.label}</span>
            <span class="metric-value">${metric.value}</span>
        `;
        metrics.appendChild(metricItem);
    });

    messageText.appendChild(metrics);

    content.appendChild(header);
    content.appendChild(messageText);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    const id = `loading-${Date.now()}`;
    messageDiv.id = id;
    messageDiv.className = 'message assistant-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';

    const content = document.createElement('div');
    content.className = 'message-content';

    const loading = document.createElement('div');
    loading.className = 'loading';
    loading.innerHTML = `
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
    `;

    content.appendChild(loading);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    chatContainer.appendChild(messageDiv);
    scrollToBottom();

    return id;
}

function addErrorMessage(error) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '!';

    const content = document.createElement('div');
    content.className = 'message-content';

    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    messageText.innerHTML = `
        <p><strong>Error:</strong> ${escapeHtml(error)}</p>
        <p>Please make sure the backend server is running at ${API_BASE_URL}</p>
        <p>Run: <code>python web/backend.py</code></p>
    `;

    content.appendChild(messageText);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function removeMessage(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

// ============================================
// Utility Functions
// ============================================

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Initialize
// ============================================

console.log('RLHF News Summarization UI loaded');
console.log('API Base URL:', API_BASE_URL);
console.log('Current tab:', currentTab);

// Check backend health
fetch(`${API_BASE_URL}/health`)
    .then(res => res.json())
    .then(data => {
        console.log('Backend health:', data);
        if (data.status === 'healthy') {
            console.log('[OK] Backend is ready');
        }
    })
    .catch(err => {
        console.warn('Backend not available:', err.message);
        console.log('Start backend with: python web/backend.py');
    });

// ============================================
// Example/Demo Functions
// ============================================

function loadExampleText() {
    const exampleText = `Artificial intelligence has made remarkable progress in recent years, with large language models demonstrating unprecedented capabilities in natural language understanding and generation. These models, trained on vast amounts of text data, can perform a wide range of tasks including translation, summarization, question answering, and creative writing.

However, challenges remain in ensuring these models are aligned with human values and preferences. Techniques like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) have emerged as promising approaches to fine-tune models based on human preferences, making them more helpful, harmless, and honest.

The development of these alignment techniques represents a significant shift in how we approach AI safety. Rather than relying solely on pre-training objectives, researchers now recognize the importance of incorporating human feedback directly into the training process. This allows models to learn not just what is statistically likely, but what humans actually find valuable and appropriate.

One of the key innovations in this space is the use of preference learning, where models are trained on pairs of outputs ranked by human evaluators. This approach has proven more effective than traditional reward modeling, as it directly captures the nuances of human judgment without requiring explicit reward functions.

The field continues to evolve rapidly, with researchers exploring new architectures, training methods, and applications. Recent advances include more efficient fine-tuning techniques like LoRA (Low-Rank Adaptation), which allows large models to be adapted with minimal computational resources, and improvements in evaluation metrics that better capture model quality beyond simple accuracy scores.

As these technologies become more powerful and accessible, questions about their societal impact become increasingly urgent. Issues of bias, fairness, transparency, and accountability must be addressed to ensure that AI systems serve the broader public interest rather than narrow commercial or political goals.

The integration of AI into critical domains like healthcare, education, and governance requires careful consideration of both technical capabilities and ethical implications. Stakeholders from diverse backgrounds must be involved in shaping the development and deployment of these systems to ensure they reflect a wide range of perspectives and values.

Looking forward, the challenge is not just to build more capable AI systems, but to ensure they remain aligned with human values as they grow in power and influence. This requires ongoing research, thoughtful regulation, and sustained public engagement to navigate the complex tradeoffs between innovation, safety, and societal benefit.`;

    textInput.value = exampleText;
    instructionsInput.value = "Please summarize this article";

    // Trigger input event to enable the Generate Summary button
    textInput.dispatchEvent(new Event('input', { bubbles: true }));
    instructionsInput.dispatchEvent(new Event('input', { bubbles: true }));

    // Switch to text tab if not already there
    if (currentTab !== 'text') {
        switchTab('text');
    }

    // Scroll to input
    textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function loadExampleURL() {
    const exampleURL = "https://theconversation.com/more-than-half-of-new-articles-on-the-internet-are-being-written-by-ai-is-human-writing-headed-for-extinction-268354";

    urlInput.value = exampleURL;
    instructionsInput.value = "Please summarize this article";

    // Trigger input event to enable the Generate Summary button
    urlInput.dispatchEvent(new Event('input', { bubbles: true }));
    instructionsInput.dispatchEvent(new Event('input', { bubbles: true }));

    // Switch to URL tab if not already there
    if (currentTab !== 'url') {
        switchTab('url');
    }

    // Scroll to input
    urlInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

