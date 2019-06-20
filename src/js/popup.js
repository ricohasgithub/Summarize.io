

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
        var domain = new URL(tabs[0].url).hostname;

        chrome.storage.local.get("settings", function (data) {

              // Get the settings object from the chrome local storage
              var settings = data["settings"];

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

              console.log(settings);

              chrome.storage.local.set({settings: settings}, function () {
                console.log("Settings stored");
              });

              // Bind the toggle-enable id object to the settings in the local storage
              $("#toggle-enable").prop('checked', settings["toggle-enable"]);

              if (settings["toggle-enable"]) {
                console.log("enabled");
              } else {
                console.log("disabled");
              }

        });
    });
}

function updateSettings () {

  console.log("Updating settings");

    chrome.storage.local.get("settings", function (data) {

        // Get the settings from the chrome tab local storage
        var settings = data["settings"];

        if (typeof settings === "undefined"){
            settings = {};
        }

        if (typeof settings["disabled-hostnames"] === "undefined") {
            settings["disabled-hostnames"] = [];
        }

        // Get an update to see whether the toggle-enable div has changed status (checked or not)
        settings["toggle-enable"] = $("#toggle-enable").is(':checked');

        chrome.storage.local.set({settings: settings}, function (data) {
            loadSettings();
        });

    });

}

function toggleSettings () {

  // Reload the settings
  loadSettings();

  chrome.tabs.query({'active': true, 'lastFocusedWindow': true}, function (tabs) {

        // Get the current URL and domain name of the tab
        var domain = new URL(tabs[0].url).hostname;

        chrome.storage.local.get("settings", function (data) {

              // Get the current settings from the chrome local storage
              let settings = data["settings"];

              // If the local chrome storage has not had initialized a disabled-hostnamess obejct, create an empty one
              if (typeof settings["disabled-hostnames"] === "undefined") {
                settings["disabled-hostnames"] = [];
              }

              if (settings["toggle-enable"]) {

                console.log("disabling");

                // Change popup appearance from disabled to enabled
                $("#toggle-enable").removeClass("disabled").addClass("enabled");

                // Add the current tab's domain name to the disabled-hostnames object-array
                settings["disabled-hostnames"].push(domain);

              } else {

                console.log("enabling");

                // Change popup appearance from enabled to disabled
                $("#toggle-enable").removeClass("enabled").addClass("disabled");

                // Remove current tab hostname from disabled-hostnames object-array
                settings["disabled-hostnames"] = $.grep(settings["disabled-hostnames"], function (item) {
                  return item !== domain;
                });

                chrome.storage.local.set({settings: settings}, function (data) {
                  // Reload settings after updating local storage values
                  loadSettings();
                });
              }
        });
    });
}
