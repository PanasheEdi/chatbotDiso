document.addEventListener("DOMContentLoaded", function () {
    const chatbox = document.getElementById("chatbox");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    function sendMessage() {
        const userMessage = userInput.value.trim();
        if (userMessage === "") return;
        
        appendMessage("You", userMessage);
        userInput.value = "";
        analyzeSentiment(userMessage);
    }

    function appendMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message");
        messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function analyzeSentiment(text) {
        fetch("/analyze-sentiment", { //might need to come back to change
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: text })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Sentiment Analysis Response:", data); // Debugging
            handleBotResponse(data.dominant_emotion);
        })
        .catch(error => {
            console.error("Error analyzing sentiment:", error);
            appendMessage("Bot", "Sorry, I couldn't understand your emotions. Can you tell me more?");
        });
    }
});
