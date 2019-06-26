
import * as tf from '@tensorflow/tfjs';

const model_host = 'https://api.myjson.com/bins/nm2n3';

class Summarizer () {

  constructor () {
    this.textRequests = {};
    this.addListeners();
    this.loadModel();
  }

  addListeners () {
    chrome.runtime.onMessage.addListener(
      function(request, sender, sendResponse) {
        if (request.sender == "content" && request.selText == "disabled") {
          // Disabled for webpage - No action
        } else if (request.sender == "content" && request.selText !== "disabled") {
          // User has selected text that is not "disabled", summarize input text (selText) and send back response
          if (this.model) {
            var summary = summarize(request.selText);
            // Send an update message to the content script
            chrome.runtime.sendMessage({sender: "background", selText : summary}, function () {
              console.log("Returning summary");
            });
          }
        }
    });
  }

  async loadModel () {
    const model = await tf.loadLayersModel(model_host);
  }

  async summarize (text) {
    var inputLength = getTExtLength(text);

  }

  async getTextLength (text) {
    return text.split().length;
  }

}
