
var markovjs = (function() {

  var MarkovModel = function (inputText) {
  }

  MarkovModel.prototype = {

    prep: function (inputText, capLength) {

    },

    summarize: function (inputText, capLength) {

    }

  }

  function getRand (seed) {
    return Math.random()*seed;
  }

  // Exported elements: markov model with highlighted text data prepped for summarization
  let exports = {};
  exports.MarkovModel = MarkovModel;
  return exports;

})();
