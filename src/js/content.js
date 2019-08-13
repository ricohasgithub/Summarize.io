// Variables for oprhan checking
var orphanMessageId = chrome.runtime.id + 'orphanCheck';
window.dispatchEvent(new Event(orphanMessageId));
window.addEventListener(orphanMessageId, unregisterOrphan);

// register all listeners with named functions to preserve their object reference
chrome.runtime.onMessage.addListener(onMessage);
document.addEventListener('mousemove', onMouseMove);

// the popup script checks it to see if a usable instance of content script is running
window.running = true;

// Unique ID for the className.
var HIGHLIGHT_CLASS = 'highlight';

// Previous dom, that we want to track, so we can remove the previous styling.
var prevDOM = null;

let enabled = false;

function onMessage (request, sender, sendResponse) {
  console.log("message received!");
  if (request.cTabSettings === true) {
    enabled = true;
  } else if (request.cTabSettings === false) {
    enabled = false;
  }
}

function onMouseMove (e) {
  // DOM events still fire in the orphaned content script after the extension
  // was disabled/removed and before it's re-enabled or re-installed
  if (unregisterOrphan()) { return }

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

    // Send an update message to the popup.js script
    chrome.runtime.sendMessage({sender: "content", selText : ($(srcElement).text())}, function () {
      console.log("enabled-success");
    });


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

    // Send an update message to the popup.js script; send summarization request to background script
    chrome.runtime.sendMessage({sender: "content", selText : "disabled"}, function () {
      console.log("disabled-success");
    });

  }
}

function unregisterOrphan() {
  if (chrome.i18n) {
    // someone tried to kick us out but we're not orphaned!
    return;
  }
  console.log(orphanMessageId);
  window.removeEventListener(orphanMessageId, unregisterOrphan);
  document.removeEventListener('mousemove', onMouseMove);
  try {
    // 'try' is needed to avoid an exception being thrown in some cases
    chrome.runtime.onMessage.removeListener(onMessage);
  } catch (e) {}
  return true;
}
