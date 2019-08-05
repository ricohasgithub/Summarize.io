
var markovjs = (function() {

  var MarkovModel = function (inputText) {
  }

  MarkovModel.prototype = {

    prep: function (inputText, capLength) {

      this.inputText = inputText;
      this.capLength = capLength;

      let followers = new Array(inputText.length);

      for (i=0; i<followers.length; i++) {
        folowers[i] = new Array(inputText.length - i);
      }

    },

    summarize: function (inputText, capLength) {
      let initWord = Math.random() * inputText.length;

    }

  }

  function getRand (seed) {
    return Math.random() * seed;
  }

  // Exported elements: markov model with highlighted text data prepped for summarization
  let exports = {};
  exports.MarkovModel = MarkovModel;
  return exports;

})();
