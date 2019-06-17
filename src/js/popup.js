
window.onload = function () {
  loadSettings();
  $('#toggle-enable').click(toggleSettings);
}

setInterval(function () {
    updateSettings();
}, 500);

function loadSettings () {

  chrome.tabs.query({'active': true, 'lastFocusedWindow': true}, function (tabs) {

        // Get the current URL and domain name
        let domain = new URL(tabs[0].url).hostname;

        chrome.storage.local.get("settings", function (data) {

              // Get the settings object from the chrome local storage
              let settings = data["settings"];

              // If the local chrome storage has not had initialized a settings query, create an empty object
              if (typeof settings === "undefined") {
                  settings = {};
              }
              if (typeof settings["toggle-enable"] === "undefined") {
                  settings["toggle-enable"] = true;
              }
              if (typeof settings["disabled-hostnames"] === "undefined") {
                  settings["disabled-hostnames"] = [];
              }

              // Bind the toggle-enable id object to the settings in the local storage
              $("#toggle-enable").prop('checked', settings["toggle-enable"]);

              if (settings["toggle-enable"]) {
                console.log("enabled");
              } else {
                console.log("disabled");
              }

              if (settings["disabled-hostnames"].indexOf(domain) != -1) {
              } else {
              }

        });
    });
}

function updateSettings () {
  loadSettings();
}

function toggleSettings () {

  // Reload the settings
  loadSettings();

  chrome.tabs.query({'active': true, 'lastFocusedWindow': true}, function (tabs) {

        // Get the current URL and domain name of the tab
        let domain = new URL(tabs[0].url).hostname;

        chrome.storage.local.get("settings", function (data) {

              // Get the current settings from the chrome local storage
              let settings = data["settings"];

              // If the local chrome storage has not had initialized a disabled-hostnamess obejct, create an empty one
              if (typeof settings["disabled-hostnames"] === "undefined") {
                settings["disabled-hostnames"] = [];
              }

              if (settings["toggle-enable"]) {
                console.log("disabling");
                //
                settings["disabled-hostnames"].push(domain);
              } else {
                console.log("enabling");
              }
        });
    });
}
