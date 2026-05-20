# Trial 999 Copy 6 — Pepper Robot Classroom Project

## Architecture

- **face+ROBOTspeech+hand+working+DEPTH.py** — Main script: face recognition, speech, Gemini chat, quiz generation, attendance logging. Runs on Mac.
- **combine.py** — Flask server (port 5000) + HTTPServer (port 5001). Serves API endpoints + HTML pages. Located in `Combined_Working/Combine trial last 2/Combine  trial/Untitled copy/`.
- **serve_quiz.py** — Simple HTTP server on port 8000 (serves questions.json with CORS).
- **Pepper tablet** — Loads HTML pages from `file://` via Choregraphe project. Uses XHR to talk to Flask/HTTPServer.

## Key Paths

- HTML pages: `Combined_Working/.../Untitled copy/html/pages/`
- JS files: `Combined_Working/.../Untitled copy/html/js/`
- Quiz data: `Combined_Working/.../Untitled copy/html/data/questions.json`
- Config (auto-generated): `html/js/config.js` — written by `combine.py` on startup via `write_frontend_config()`

## Servers & Ports

- **Flask (combine.py)**: port 5000 — `/students`, `/get_quiz`, `/edit_json`, `/html/<path>` (static HTML serving), `/set_choice`, `/get_choice`, etc.
- **Attendance HTTPServer (combine.py)**: port 5001 — `/students`, `/send_attendance_email`
- **serve_quiz.py**: port 8000 — static file server for questions.json
- **XAMPP MySQL**: port 3303 (not default 3306!)

## Database

- MySQL via XAMPP, port **3303**, database `classroom_db`
- Tables: `attendance` (id, student_name, log_time), `qna_logs` (question, answer, ask_time)

## Known Issues & Fixes Applied

1. **XHR abort() bug in attendance.js** — `settle()` called `xhr.abort()` after readyState===4, zeroing out xhr.status on old WebKit/Chrome. Fix: only abort if `readyState !== 4`, and capture status/body into locals before calling settle().
2. **Pepper file:// origin** — Old WebKit returns status 0 for successful cross-origin XHR from `file://`. Fix: accept `status === 0` with non-empty responseText in both attendance.js and confirmation.js.
3. **config.js dynamic hostnames** — Hardcoded IPs caused cross-origin failures on localhost. Fix: config.js now uses `window.location.hostname` dynamically, falls back to LAN IP for `file://`.
4. **Gemini quiz JSON parsing** — Gemini sometimes returns malformed JSON. Fix: added markdown fence stripping, 3 retries, and validation in `generate_quiz_from_db()`.

## Config Generation

`combine.py` auto-generates `config.js` on startup using detected LAN IP (`detect_local_ip()`). Uses IIFE with `window.location.hostname` for dynamic URL resolution. Override IP with env var: `LAPTOP_IP=x.x.x.x python combine.py`

## Pepper Integration

- Choregraphe project at `Combined_Working/.../Untitled copy/` with `script.pml`, `behavior.xar`
- Tablet pages loaded via `file://` on Pepper's Android tablet
- QiSession/ALMemory used for Pepper speech events
- Pages are ES5 only (no fetch, no arrow functions, no template literals)

## IP Address

- Laptop IP: detected dynamically, was `10.10.42.26` during this session
