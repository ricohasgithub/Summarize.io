window.onload = function () {
  loadSettings();
  $('#toggle-enable').click(toggleSettings);
}

// Check for incoming messages from background and content scripts periodcially
chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    console.log(request.selText);
    if (request.sender == "content" && request.selText == "disabled") {
      // Disabled for webpage - No action
    } else if (request.sender == "content" && request.selText !== "disabled") {
      // Get summarized content from content.js; display
    }
});

function loadSettings () {

  chrome.tabs.query({'active': true, 'currentWindow': true}, function (tabs) {

      // Get the current URL and domain name
      var domain = new URL(tabs[0].url).hostname;

      $("#toggle-enable").prop('checked', false);
      $("#toggle-enable").removeClass("enabled").addClass("disabled");
      $("#sitename").text(domain);

  });
}

async function toggleSettings () {

  chrome.tabs.query({'active': true, 'currentWindow': true}, function (tabs) {

        // Get the current URL and domain name of the tab
        var domain = new URL(tabs[0].url).hostname;

        if ($('#toggle-enable').is(":checked") === false) {
          console.log("disabling");
          // Change popup appearance from disabled to enabled
          $("#toggle-enable").removeClass("enabled").addClass("disabled");

          // Send an update message to the content.js script
          chrome.tabs.sendMessage(tabs[0].id, {cTabSettings: false});

        } else if ($('#toggle-enable').is(":checked") === true) {
          console.log("enabling");
          // Change popup appearance from enabled to disabled
          $("#toggle-enable").removeClass("disabled").addClass("enabled");

          if (ensureContentScript(tabs[0].id)) {
            // Send an update message to the content.js script
            chrome.tabs.sendMessage(tabs[0].id, {cTabSettings: true});
          }

        } else {
          console.log("noooooo");
        }

    });
}

function getCurrWebpageSettings (domain) {
  let enabled = $('#toggle-enable').is(":checked");
  return enabled;
}

async function ensureContentScript(tabId) {
  try {
    const [running] = await browser.tabs.executeScript(tabId, {
      code: 'window.running === true',
    });
    if (!running) {
      console.log("injected new script");
      await browser.tabs.executeScript(tabId, {file: 'content.js'});
    }
    return true;
  } catch (e) {}
}
