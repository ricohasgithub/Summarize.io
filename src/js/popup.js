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
      // $("#intro-text").text("Message");
      console.log(request.selText);
    }
});

function loadSettings () {

  chrome.tabs.query({'active': true, 'currentWindow': true}, function (tabs) {

        // Get the current URL and domain name
        var domain = new URL(tabs[0].url).hostname;

        chrome.storage.local.get("settings", function (data) {

              // Get the settings object from the chrome local storage
              var settings = data["settings"];

              // If the local chrome storage has not had initialized a settings query, create an empty object
              if (typeof settings === "undefined") {
                  settings = {};
              }

              if (typeof settings["disabled-hostnames"] === "undefined") {
                  settings["disabled-hostnames"] = [];
              }

              var pageEnableStats = getCurrWebpageSettings(domain);

              // Bind the toggle-enable id object to the settings in the local storage
              $("#toggle-enable").prop('checked', pageEnableStats);

              if (settings["disabled-hostnames"].indexOf(domain) != -1) {
                $("#toggle-enable").removeClass("disabled").addClass("enabled");
              } else {
                $("#toggle-enable").removeClass("enabled").addClass("disabled");
              }

              $("#sitename").text(domain);

        });
    });
}

function updateSettings () {

    chrome.storage.local.get("settings", function (data) {

        // Get the settings from the chrome tab local storage
        var settings = data["settings"];

        if (typeof settings === "undefined"){
            settings = {};
        }

        if (typeof settings["disabled-hostnames"] === "undefined") {
            settings["disabled-hostnames"] = [];
        }

        chrome.storage.local.set({settings: settings}, function (data) {
            loadSettings();
        });

    });

}

function toggleSettings () {

  chrome.tabs.query({'active': true, 'currentWindow': true}, function (tabs) {

        // Get the current URL and domain name of the tab
        var domain = new URL(tabs[0].url).hostname;

        chrome.storage.local.get("settings", function (data) {

              // Get the current settings from the chrome local storage
              let settings = data["settings"];

              // If the local chrome storage has not had initialized a disabled-hostnamess obejct, create an empty one
              if (typeof settings["disabled-hostnames"] === "undefined") {
                settings["disabled-hostnames"] = [];
              }

              if (settings["disabled-hostnames"].indexOf(domain) != -1) {

                // Change popup appearance from disabled to enabled
                $("#toggle-enable").removeClass("enabled").addClass("disabled");

                // Add the current tab's domain name to the disabled-hostnames object-array
                settings["disabled-hostnames"].push(domain);

              } else {

                // Change popup appearance from enabled to disabled
                $("#toggle-enable").removeClass("disabled").addClass("enabled");

                // Remove current tab hostname from disabled-hostnames object-array
                settings["disabled-hostnames"] = $.grep(settings["disabled-hostnames"], function (item) {
                  return item !== domain;
                });

              }

              chrome.storage.local.set({settings: settings}, function (data) {
                // Reload settings after updating local storage values
                loadSettings();
              });

              // Send an update message to the content.js script
              chrome.tabs.sendMessage(tabs[0].id, {cTabSettings: ($('#toggle-enable').is(":checked"))});

        });
    });
}

function getCurrWebpageSettings (domain) {

  var enabled;

  chrome.tabs.query({'active': true, 'currentWindow': true}, function (tabs) {

    chrome.storage.local.get("settings", function (data) {

        // Get the settings from the chrome tab local storage
        var settings = data["settings"];

        disabled = (settings["disabled-hostnames"].indexOf(domain) != -1);

      });

  });

  return enabled;

}
