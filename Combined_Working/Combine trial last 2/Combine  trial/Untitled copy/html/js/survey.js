/* Survey page nav. Pepper-safe ES5. */
(function () {
  function bind(id, href) {
    var el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("click", function () {
      window.location.href = href;
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      bind("btnBack", "index.html");
      bind("btnNext", "next.html");
    });
  } else {
    bind("btnBack", "index.html");
    bind("btnNext", "next.html");
  }
})();
