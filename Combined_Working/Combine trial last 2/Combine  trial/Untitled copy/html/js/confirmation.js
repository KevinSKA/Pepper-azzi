/* Quiz / confirmation page. Pepper-safe ES5. URL comes from
   window.APP_CONFIG (config.js) — no hardcoded IPs. */
(function () {
  var cfg = window.APP_CONFIG || {};
  // Prefer the Flask /get_quiz endpoint; fall back to STUDENTS_BASE_URL.
  var QUIZ_URL =
    cfg.QUIZ_URL ||
    (cfg.STUDENTS_BASE_URL || "http://127.0.0.1:5000") + "/get_quiz";

  var questions = [];
  var currentQuestion = 0;
  var answered = false;
  var score = 0;

  function $(id) {
    return document.getElementById(id);
  }

  function loadQuestion(index) {
    answered = false;
    var q = questions[index];
    $("pageHeading").innerHTML = q.heading;
    $("pageText").innerHTML = q.text;
    $("option1Text").innerHTML = q.options[0];
    $("option2Text").innerHTML = q.options[1];
    $("option3Text").innerHTML = q.options[2];

    // Tell Pepper to read the question out loud.
    try {
      QiSession(function (session) {
        session.service("ALMemory").then(function (mem) {
          var textToSpeak = q.heading + ". " + q.text;
          mem.raiseEvent("PepperSpeakQuestion", textToSpeak);
        });
      });
    } catch (e) {
      if (window.console) console.log("Could not trigger speech: " + e);
    }

    // Reset button styles by replacing with fresh buttons.
    var ids = ["option1", "option2", "option3"];
    for (var i = 0; i < ids.length; i++) {
      var oldBtn = $(ids[i]);
      var newBtn = oldBtn.cloneNode(true);
      newBtn.className = "fa rate-button rounded-rect-btn brown-btn quiz-btn";
      newBtn.removeAttribute("style");
      oldBtn.parentNode.replaceChild(newBtn, oldBtn);
    }
    $("option1").onclick = function () {
      selectAnswer("1");
    };
    $("option2").onclick = function () {
      selectAnswer("2");
    };
    $("option3").onclick = function () {
      selectAnswer("3");
    };
  }

  function selectAnswer(choice) {
    if (answered) return;
    answered = true;

    var q = questions[currentQuestion];
    var correctId = "option" + q.correct;
    var selectedId = "option" + choice;

    var correctBtn = $(correctId);
    correctBtn.classList.remove("brown-btn");
    correctBtn.classList.add("correct-btn");

    if (choice === q.correct) {
      score++;
    } else {
      var selectedBtn = $(selectedId);
      selectedBtn.classList.remove("brown-btn");
      selectedBtn.classList.add("wrong-btn");
    }

    try {
      if (choice === q.correct) {
        raiseConfirmationEvent("answerCorrect");
      } else {
        raiseConfirmationEvent("answerWrong");
      }
    } catch (e) {}

    currentQuestion++;
    if (currentQuestion < questions.length) {
      setTimeout(function () {
        loadQuestion(currentQuestion);
      }, 5000);
    } else {
      try {
        raiseEvent("quizScore", score);
      } catch (e) {}
    }
  }

  // Subscribe to speech answers from dialog system.
  try {
    QiSession(function (session) {
      session.service("ALMemory").then(function (mem) {
        mem.subscriber("speechAnswer").then(function (sub) {
          sub.signal.connect(function (value) {
            if (
              !answered &&
              (value === "1" || value === "2" || value === "3")
            ) {
              selectAnswer(value);
            }
          });
        });
      });
    });
  } catch (e) {
    if (window.console) console.log("Speech subscription not available: " + e);
  }

  function loadQuiz() {
    var xhr = new XMLHttpRequest();
    var url =
      QUIZ_URL +
      (QUIZ_URL.indexOf("?") < 0 ? "?" : "&") +
      "t=" +
      new Date().getTime();
    xhr.open("GET", url, true);
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) return;
      /* Old WebKit on Pepper may return status 0 for successful cross-origin
         requests from file://. Accept status 0 if there is a response body. */
      var body = xhr.responseText || "";
      if (xhr.status === 200 || (xhr.status === 0 && body.length > 0)) {
        try {
          questions = JSON.parse(body);
          if (window.console) console.log("Loaded quiz from " + QUIZ_URL);
          loadQuestion(0);
        } catch (e) {
          if (window.console) console.error("JSON Error: ", e);
        }
      } else {
        if (window.console)
          console.error("Could not load quiz. Status: " + xhr.status);
      }
    };
    xhr.send();
  }

  // Expose handler for inline onclick="selectAnswer('1')" attributes.
  window.selectAnswer = selectAnswer;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadQuiz);
  } else {
    loadQuiz();
  }
})();
