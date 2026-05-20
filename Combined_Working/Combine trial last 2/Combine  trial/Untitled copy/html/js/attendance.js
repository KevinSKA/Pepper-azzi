/* Attendance page logic. Old-WebKit / Pepper safe: ES5 only, no fetch,
   no template literals, no arrow functions. All URLs come from
   window.APP_CONFIG (config.js), regenerated each time combine.py starts. */
(function () {
  var cfg = window.APP_CONFIG || {};
  var STUDENTS_BASE_URL = cfg.STUDENTS_BASE_URL || "http://127.0.0.1:5000";
  var EMAIL_BASE_URL = cfg.EMAIL_BASE_URL || "http://127.0.0.1:5001";

  var currentStudents = [];

  function $(id) {
    return document.getElementById(id);
  }

  function backButtonHtml() {
    return (
      '<div class="action-buttons">' +
      '<a href="index.html" style="text-decoration:none;">' +
      '<button class="back-btn" type="button">Back</button>' +
      "</a></div>"
    );
  }

  function showError(msg) {
    $("mainContainer").innerHTML =
      "<h1>Attendance</h1>" +
      '<div class="error">' +
      msg +
      "</div>" +
      backButtonHtml();
  }

  function renderAttendanceScreen(students, statusText) {
    currentStudents = students || [];

    var studentList = "";
    var i;
    for (i = 0; i < currentStudents.length; i++) {
      studentList +=
        '<div class="student-item">&#8226; ' + currentStudents[i] + "</div>";
    }

    $("mainContainer").innerHTML =
      "<h1>Attendance</h1>" +
      '<div class="message">Students available in class</div>' +
      '<div class="students-box"><div class="student-list">' +
      studentList +
      "</div></div>" +
      '<div class="action-buttons">' +
      '<button class="email-btn" onclick="AttendanceApp.showPasswordScreen()">Send Email</button>' +
      '<a href="index.html" style="text-decoration:none;">' +
      '<button class="back-btn" type="button">Back</button>' +
      "</a></div>" +
      '<div class="status-msg" id="statusMsg">' +
      (statusText || "") +
      "</div>";
  }

  function showPasswordScreen() {
    $("mainContainer").innerHTML =
      "<h1>Attendance</h1>" +
      '<div class="password-screen">' +
      '<div class="password-title">Enter password to send email</div>' +
      '<input type="password" id="emailPassword" placeholder="Password">' +
      '<div class="password-actions">' +
      '<button class="confirm-btn" onclick="AttendanceApp.confirmSendEmail()">Confirm</button>' +
      '<button class="cancel-btn" onclick="AttendanceApp.backToAttendance()">Cancel</button>' +
      "</div>" +
      '<div class="status-msg" id="statusMsg"></div>' +
      "</div>";

    setTimeout(function () {
      var input = $("emailPassword");
      if (input) {
        input.focus();
      }
    }, 100);
  }

  function backToAttendance() {
    renderAttendanceScreen(currentStudents, "");
  }

  function confirmSendEmail() {
    var passwordInput = $("emailPassword");
    var statusMsg = $("statusMsg");
    if (!passwordInput) {
      return;
    }
    var password = passwordInput.value;
    if (!password) {
      if (statusMsg) {
        statusMsg.innerHTML = "Please enter the password";
      }
      return;
    }
    if (statusMsg) {
      statusMsg.innerHTML = "Sending email...";
    }

    var xhr = new XMLHttpRequest();
    xhr.open(
      "GET",
      EMAIL_BASE_URL +
        "/send_attendance_email?password=" +
        encodeURIComponent(password),
      true,
    );
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) {
        return;
      }
      try {
        var data = JSON.parse(xhr.responseText);
        if (data.success) {
          renderAttendanceScreen(
            currentStudents,
            "Attendance email sent successfully",
          );
        } else {
          if (statusMsg) {
            statusMsg.innerHTML = "Error: " + data.error;
          }
        }
      } catch (e) {
        if (statusMsg) {
          statusMsg.innerHTML = "Invalid response from server";
        }
      }
    };
    xhr.send();
  }

  function loadStudents() {
    var container = $("mainContainer");
    container.innerHTML =
      "<h1>Attendance</h1>" +
      '<div class="loading">Loading students from ' +
      STUDENTS_BASE_URL +
      "/students ...</div>";

    var xhr = new XMLHttpRequest();
    var settled = false;

    function settle(fn) {
      if (settled) {
        return;
      }
      settled = true;
      /* Only abort if the request is still in-flight. Aborting a completed
         XHR on old WebKit can zero-out status/responseText or fire onerror. */
      try {
        if (xhr.readyState !== 4) {
          xhr.abort();
        }
      } catch (e) {}
      fn();
    }

    // Manual fallback timeout — Pepper's old WebKit does not always honour
    // xhr.timeout / xhr.ontimeout, so we enforce it ourselves.
    var hardTimeout = setTimeout(function () {
      settle(function () {
        showError(
          "Request timed out reaching " +
            STUDENTS_BASE_URL +
            ". Check that combine.py is running and that the laptop IP in js/config.js is correct.",
        );
      });
    }, 12000);

    xhr.open("GET", STUDENTS_BASE_URL + "/students", true);

    // xhr.timeout is still set, in case the browser does honour it.
    try {
      xhr.timeout = 10000;
    } catch (e) {}

    xhr.onerror = function () {
      settle(function () {
        clearTimeout(hardTimeout);
        showError("Network error reaching " + STUDENTS_BASE_URL);
      });
    };

    xhr.ontimeout = function () {
      settle(function () {
        clearTimeout(hardTimeout);
        showError("Request timed out reaching " + STUDENTS_BASE_URL);
      });
    };

    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) {
        return;
      }
      /* Capture status/body BEFORE settle() calls abort(), which can
         zero-out xhr.status on some browsers (old WebKit, Chrome). */
      var status = xhr.status;
      var body = xhr.responseText || "";
      settle(function () {
        clearTimeout(hardTimeout);
        /* Old WebKit on Pepper may return status 0 for successful cross-origin
           requests from file://. Accept status 0 if there is a response body. */
        if (status !== 200 && !(status === 0 && body.length > 0)) {
          showError(
            "HTTP " + status + ": " + (body || "No response").substring(0, 200),
          );
          return;
        }
        var data;
        try {
          data = JSON.parse(body);
        } catch (e) {
          showError("Invalid JSON. Response: " + body.substring(0, 200));
          return;
        }
        if (!data.success) {
          showError("Server error: " + data.error);
          return;
        }
        var students = data.students;
        if (!students || students.length === 0) {
          container.innerHTML =
            "<h1>Attendance</h1>" +
            '<div class="message">No students found</div>' +
            backButtonHtml();
          return;
        }
        renderAttendanceScreen(students, "");
      });
    };

    try {
      xhr.send();
    } catch (e) {
      settle(function () {
        clearTimeout(hardTimeout);
        showError("Could not send request: " + e);
      });
    }
  }

  // Expose handlers used by inline onclick attributes.
  window.AttendanceApp = {
    showPasswordScreen: showPasswordScreen,
    backToAttendance: backToAttendance,
    confirmSendEmail: confirmSendEmail,
    loadStudents: loadStudents,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadStudents);
  } else {
    loadStudents();
  }
})();
