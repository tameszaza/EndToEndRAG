const chatForm = document.querySelector("#chatForm");
const questionInput = document.querySelector("#questionInput");
const chatMessages = document.querySelector("#chatMessages");
const apiStatus = document.querySelector("#apiStatus");
const chatMode = document.querySelector("#chatMode");
const quickQuestions = document.querySelectorAll("[data-question]");

function createMessage(role, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "You" : "SP";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  bubble.appendChild(paragraph);

  if (sources.length > 0) {
    const sourceList = document.createElement("div");
    sourceList.className = "source-list";
    sources.forEach((source) => {
      const badge = document.createElement("span");
      badge.textContent = `${source.id} - ${Math.round(source.score * 100)}%`;
      badge.title = source.question;
      sourceList.appendChild(badge);
    });
    bubble.appendChild(sourceList);
  }

  article.appendChild(avatar);
  article.appendChild(bubble);
  chatMessages.appendChild(article);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createTypingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.dataset.typing = "true";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "SP";

  const bubble = document.createElement("div");
  bubble.className = "bubble typing";
  bubble.innerHTML = "<span></span><span></span><span></span>";

  article.appendChild(avatar);
  article.appendChild(bubble);
  chatMessages.appendChild(article);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingMessage() {
  const typing = document.querySelector("[data-typing='true']");
  if (typing) {
    typing.remove();
  }
}

function setLoading(isLoading) {
  const button = chatForm.querySelector("button");
  button.disabled = isLoading;
  questionInput.disabled = isLoading;
}

async function askQuestion(question) {
  createMessage("user", question);
  createTypingMessage();
  setLoading(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }

    const payload = await response.json();
    removeTypingMessage();
    createMessage("assistant", payload.answer, payload.sources || []);
    chatMode.textContent = payload.mode === "local-rag" ? "Local RAG" : payload.mode;
  } catch (error) {
    removeTypingMessage();
    createMessage(
      "assistant",
      "I could not reach the SleepPilot API. Please check that the FastAPI server is running."
    );
  } finally {
    setLoading(false);
    questionInput.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }
  questionInput.value = "";
  questionInput.style.height = "auto";
  askQuestion(question);
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 126)}px`;
});

quickQuestions.forEach((button) => {
  button.addEventListener("click", () => {
    askQuestion(button.dataset.question);
  });
});

async function checkApiHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("Unhealthy API");
    }
    apiStatus.classList.remove("offline");
    apiStatus.classList.add("online");
    apiStatus.querySelector("span:last-child").textContent = "API online";
  } catch (error) {
    apiStatus.classList.remove("online");
    apiStatus.classList.add("offline");
    apiStatus.querySelector("span:last-child").textContent = "API offline";
  }
}

window.addEventListener("load", () => {
  checkApiHealth();
  if (window.lucide) {
    window.lucide.createIcons();
  }
});
