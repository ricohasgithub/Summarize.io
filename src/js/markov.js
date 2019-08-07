
var markovjs = (function() {

  var MarkovModel = function (inputText) {
  }

  MarkovModel.prototype = {

    prep: function (inputText, capLength) {

      this.inputText = inputText;
      this.capLength = capLength;

      let followers = new Array(inputText.length);

      for (i=0; i<inputText.length; i++) {
        // Create a new array of length 4 (4 most significant words)
        folowers[i] = new Array(4);
      }

      // Fill the follower array
      for (i=0; i<inputText.length; i++) {

        let currWord = this.parseWord(inputText[i]);


      }


    },

    summarize: function (inputText, capLength) {

      let initWord = Math.random() * inputText.length;

    },

    parseWord: function (word) {
      let newWord = word.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"");
      return newWord;
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
