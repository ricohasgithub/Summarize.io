// Unique ID for the className.
var HIGHLIGHT_CLASS = 'highlight';

// Previous dom, that we want to track, so we can remove the previous styling.
var prevDOM = null;

let enabled = false;

chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    if (request.cTabSettings === true) {
      enabled = true;
    } else if (request.cTabSettings === false) {
      enabled = false;
    }
});

// Mouse listener for any move event on the current document.
document.addEventListener('mousemove', function (e) {

      chrome.runtime.sendMessage({sender: "content", selText : "disabled"});

  if (enabled) {

    // Go through highlighting/appending procedure throughout the DOM
    var srcElement = e.target;

    // Lets check if our underlying element is a DIV.
    if (srcElement.nodeName == 'DIV') {

        // For NPE checking, we check safely. We need to remove the class name
        // Since we will be styling the new one after.
        if (prevDOM != null) {
          prevDOM.classList.remove(HIGHLIGHT_CLASS);
        }

        // Add a visited class name to the element. So we can style it.
        srcElement.classList.add(HIGHLIGHT_CLASS);

        // The current element is now the previous. So we can remove the class
        // during the next iteration.
        prevDOM = srcElement;
    }

    console.log("Sending message");

    // Send an update message to the popup.js script
    chrome.runtime.sendMessage({sender: "content", selText : ($(srcElement).text())});


  } else if (!enabled) {

    // Go through unhighlighting/unappending procedure throughout the DOM
    var srcElement = e.target;

    // Lets check if our underlying element is a DIV.
    if (srcElement.nodeName == 'DIV') {

        // For NPE checking, we check safely. We need to remove the class name
        // Since we will be styling the new one after.
        if (prevDOM != null) {
          prevDOM.classList.remove(HIGHLIGHT_CLASS);
        }

        // Add a visited class name to the element. So we can style it.
        srcElement.classList.remove(HIGHLIGHT_CLASS);

        // The current element is now the previous. So we can remove the class
        // during the next iteration.
        prevDOM = srcElement;
    }

    // Send an update message to the popup.js script
    chrome.runtime.sendMessage({sender: "content", selText : "disabled"});

  }

}, false);
