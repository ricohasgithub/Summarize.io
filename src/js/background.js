
import * as tf from '@tensorflow/tfjs';
import { DICTIONARY } from  './dictionary.js';

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
    console.log('Loading Model...');
    this.model = await tf.loadLayersModel(model_host);
    console.log('Model Loaded');
  }

  async summarize (text) {

    if (!this.model) {
      return;
    }

    console.log('Summarizing...');

    // Get text length
    var inputLength = getTextLength(text);

    // Transform the input text data into an integer form
    var inSeq = transformInput(text);
    // Summarize/predict
    var summary = this.model.predict(inSeq);

    console.log("Finished Predicting!");

    summary = convertPredictions(summary);

    return summary;

  }

  async getTextLength (text) {
    return text.split().length;
  }

  async buildDictionary (text) {
    // Get the maximum index/length of the text (number of words) and construct empty dictionary
    var maxIndex = getTextLength(text);
    var dictionary = [];



    return dictionary;

  }

  async transformInput (text) {
    // Transform input text (highlight/selText) into a tensor of integer values (constuct a dictionary)
  }

  async convertPredictions(text) {

  }

}
