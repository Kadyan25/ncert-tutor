const API_URL = "http://localhost:8000/chat";

const messagesEl = document.getElementById("messages");
const emptyStateEl = document.getElementById("emptyState");
const inputEl = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const exampleChipsEl = document.getElementById("exampleChips");

let history = [];
let isWaiting = false;

const SUBJECT_EXAMPLES = {
  mathematics: "Explain the difference between definite and indefinite integrals.",
  physics: "What is the difference between EMF and terminal voltage?",
  chemistry: "Explain how an electrochemical cell works.",
};

const BOOK_ICON = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4 4h16v12H7l-3 3V4z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
</svg>`;

const TUTOR_AVATAR = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 3L2 8l10 5 8-4.2V15h2V8L12 3z" fill="currentColor"/>
</svg>`;

function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}

function isNearBottom() {
  return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 150;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideEmptyState() {
  if (emptyStateEl) emptyStateEl.remove();
}

function appendUserMessage(text) {
  const el = document.createElement("div");
  el.className = "message user";
  el.innerHTML = `<div class="message-body"><div class="message-content"></div></div>`;
  el.querySelector(".message-content").textContent = text;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function appendTypingIndicator() {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.id = "typingIndicator";
  el.innerHTML = `
    <div class="message-avatar">${TUTOR_AVATAR}</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>
  `;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

function citationLabel(c) {
  return `${c.book} · ${c.section} · p.${c.page}`;
}

function dedupeCitations(citations) {
  const seen = new Set();
  const unique = [];
  for (const c of citations) {
    const key = `${c.book}|${c.section}|${c.page}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(c);
    }
  }
  return unique;
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// No single-asterisk italics: a math/physics/chemistry tutor's answers use "*" as a
// literal multiplication sign constantly (e.g. "k * q1 * q2"), which single-asterisk
// italic parsing would misread as emphasis markers and mangle the formula.
function inlineMarkdown(text) {
  return text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+?)`/g, "<code>$1</code>");
}

// Small dependency-free markdown renderer: handles bold/italic/inline-code and
// bullet lists, which covers what local models actually produce. Not a full parser.
function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  let html = "";
  let inList = false;

  for (const line of lines) {
    const bulletMatch = line.match(/^\s*[*-]\s+(.*)/);
    if (bulletMatch) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${inlineMarkdown(bulletMatch[1])}</li>`;
      continue;
    }
    if (inList) {
      html += "</ul>";
      inList = false;
    }
    html += line.trim() === "" ? "<br>" : `<p>${inlineMarkdown(line)}</p>`;
  }
  if (inList) html += "</ul>";
  return html;
}

function appendAssistantMessage(answer, rawCitations, isOutOfScope) {
  const wasNearBottom = isNearBottom();
  const citations = rawCitations ? dedupeCitations(rawCitations) : [];
  const el = document.createElement("div");
  el.className = "message assistant" + (isOutOfScope ? " out-of-scope" : "");
  el.innerHTML = `
    <div class="message-avatar">${TUTOR_AVATAR}</div>
    <div class="message-body">
      <div class="message-content"></div>
      <div class="citations"></div>
    </div>
  `;
  messagesEl.appendChild(el);
  const contentEl = el.querySelector(".message-content");
  const citationsEl = el.querySelector(".citations");

  revealText(contentEl, answer, () => {
    contentEl.innerHTML = renderMarkdown(answer);
    if (citations.length) {
      citationsEl.innerHTML = citations
        .map((c) => `<span class="citation-pill">${BOOK_ICON}${citationLabel(c)}</span>`)
        .join("");
    }
    if (wasNearBottom) scrollToBottom();
  });

  if (wasNearBottom) scrollToBottom();
}

function appendErrorMessage(text) {
  const el = document.createElement("div");
  el.className = "message assistant error";
  el.innerHTML = `
    <div class="message-avatar">${TUTOR_AVATAR}</div>
    <div class="message-body"><div class="message-content"></div></div>
  `;
  el.querySelector(".message-content").textContent = text;
  messagesEl.appendChild(el);
  scrollToBottom();
}

// Backend returns the full answer in one response (no streaming endpoint), so this
// simulates a token-by-token reveal client-side for better perceived responsiveness.
function revealText(el, fullText, onDone) {
  const words = fullText.split(" ");
  let i = 0;
  const step = () => {
    i += 1;
    el.textContent = words.slice(0, i).join(" ");
    if (isNearBottom()) scrollToBottom();
    if (i < words.length) {
      setTimeout(step, 18);
    } else if (onDone) {
      onDone();
    }
  };
  step();
}

async function sendMessage(text) {
  if (!text.trim() || isWaiting) return;

  hideEmptyState();
  appendUserMessage(text);
  history.push({ role: "user", content: text });

  inputEl.value = "";
  autoResize();
  isWaiting = true;
  sendBtn.disabled = true;
  appendTypingIndicator();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });

    removeTypingIndicator();

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const data = await response.json();
    const isOutOfScope = !data.citations || data.citations.length === 0;
    appendAssistantMessage(data.answer, data.citations, isOutOfScope);
    history.push({ role: "assistant", content: data.answer });
  } catch (err) {
    removeTypingIndicator();
    appendErrorMessage(
      "Couldn't reach the tutor backend. Make sure the FastAPI server is running on localhost:8000."
    );
    history.pop(); // drop the unanswered user turn so retry doesn't duplicate context
  } finally {
    isWaiting = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener("click", () => sendMessage(inputEl.value));

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(inputEl.value);
  }
});

inputEl.addEventListener("input", autoResize);

newChatBtn.addEventListener("click", () => {
  history = [];
  messagesEl.innerHTML = "";
  location.reload();
});

if (exampleChipsEl) {
  exampleChipsEl.addEventListener("click", (e) => {
    const chip = e.target.closest(".example-chip");
    if (chip) sendMessage(chip.textContent);
  });
}

document.querySelectorAll(".subject-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const subject = chip.dataset.subject;
    inputEl.value = SUBJECT_EXAMPLES[subject] || "";
    autoResize();
    inputEl.focus();
  });
});

inputEl.focus();
