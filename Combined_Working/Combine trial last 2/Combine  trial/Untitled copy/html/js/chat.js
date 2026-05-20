/* Chat display. Pepper-safe ES5 (no template literals). */
function displayChat(sender, message) {
  var container = document.getElementById("chat-box");
  if (!container) return;

  var msgDiv = document.createElement("div");
  var isPepper = String(sender).toLowerCase() === "pepper";
  msgDiv.className = isPepper ? "msg pepper" : "msg student";

  msgDiv.innerHTML =
    '<span class="name">' + sender + "</span>" +
    "<div>" + message + "</div>";

  container.appendChild(msgDiv);

  container.scrollTop = container.scrollHeight;
  window.scrollTo(0, document.body.scrollHeight);

  while (container.children.length > 10) {
    container.removeChild(container.firstChild);
  }
}

window.displayChat = displayChat;
