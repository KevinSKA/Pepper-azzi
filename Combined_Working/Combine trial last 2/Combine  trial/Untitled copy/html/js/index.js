/* Home / menu page. Pepper-safe ES5. */
(function () {
  var memoryService = null;

  function connectToPepper() {
    if (typeof QiSession === "undefined") {
      return;
    }
    try {
      new QiSession(
        function (session) {
          session
            .service("ALMemory")
            .done(function (memory) {
              memoryService = memory;
            })
            .fail(function (error) {
              if (window.console) console.log("ALMemory failed:", error);
            });
        },
        function (error) {
          if (window.console) console.log("QiSession failed:", error);
        }
      );
    } catch (e) {
      if (window.console) console.log("QiMessaging error:", e);
    }
  }

  function attachNextHook() {
    var nextBtn = document.getElementById("btnNext");
    if (!nextBtn) {
      return;
    }
    nextBtn.addEventListener("click", function () {
      if (memoryService) {
        try {
          memoryService.raiseEvent("NextPressed", 1);
        } catch (e) {}
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      connectToPepper();
      attachNextHook();
    });
  } else {
    connectToPepper();
    attachNextHook();
  }
})();
