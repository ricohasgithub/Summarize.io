
import * as tf from '@tensorflow/tfjs';

const model_host = 'https://api.myjson.com/bins/10mbt5';

const model = await tf.loadLayersModel(model_host);

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
            
          }
        }
    });
  }

  async loadModel () {

  }

  async summarize () {

  }

  async getTextLength (text) {
    return  text.split();
  }

}
