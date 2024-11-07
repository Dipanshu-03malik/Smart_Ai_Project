document.getElementById("openApp").onclick = function () {
    const width = Math.floor(screen.width / 2);  // Set width to half the screen width
    const height = screen.height; // Set height to full screen height
    const left = Math.floor(screen.width / 2); // Position it on the right side
    const top = 0; // Start from the top of the screen

    // Use chrome.windows.create to open a new window
    chrome.windows.create({
        url: "http://localhost:8501",
        type: "popup",
        width: width,
        height: height,
        top: top,
        left: left
    });
};
