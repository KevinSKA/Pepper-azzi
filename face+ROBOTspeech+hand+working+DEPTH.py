import os
import cv2
import json
import socket
import time
import threading
import mediapipe as mp
import face_recognition
import numpy as np
import mysql.connector
import math
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
MODEL_ID = 'gemini-2.5-flash'

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'port': 3303,
    'password': '', 
    'database': 'classroom_db'
}

recently_logged = {}
app_running = True 

class SocketCamera:
    def __init__(self, port, stream_type="RGB"):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', port)) 
        self.server_socket.listen(1)
        self.conn = None
        self.is_open = False
        self.stream_type = stream_type
        self.latest_frame = None 
        self.lock = threading.Lock()
        
        print(f"📡 Mac listening for {stream_type} stream on {port}...")
        threading.Thread(target=self._accept_connection, daemon=True).start()

    def _accept_connection(self):
        self.conn, addr = self.server_socket.accept()
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"✅ Fast {self.stream_type} Stream connected from {addr}!")
        self.is_open = True
        threading.Thread(target=self._update_loop, daemon=True).start()

    def recvall(self, count):
        buf = b''
        while count:
            newbuf = self.conn.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def _update_loop(self):
        self.conn.settimeout(2.0)
        while self.is_open:
            try:
                if self.stream_type == "RGB":
                    header_bytes = self.recvall(16)
                    if not header_bytes: continue
                    size_str = header_bytes.decode('utf-8').strip()
                    if not size_str.isdigit(): continue
                    size = int(size_str)
                    compressed_data = self.recvall(size)
                    if not compressed_data: continue
                    np_arr = np.frombuffer(compressed_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.latest_frame = frame

                elif self.stream_type == "DEPTH":
                    header_bytes = self.recvall(32)
                    if not header_bytes: continue
                    header_str = header_bytes.decode('utf-8').strip()
                    parts = header_str.split(',')
                    if len(parts) != 3: continue
                    width, height, size = int(parts[0]), int(parts[1]), int(parts[2])
                    raw_bytes = self.recvall(size)
                    if not raw_bytes: continue
                    expected_bytes = width * height * 2
                    if len(raw_bytes) == expected_bytes:
                        frame_array = np.frombuffer(raw_bytes, dtype=np.uint16)
                        frame = frame_array.reshape((height, width))
                        with self.lock:
                            self.latest_frame = frame
            except socket.timeout: continue 
            except Exception: break 

    def read(self):
        if not self.is_open or self.latest_frame is None: 
            return False, None
        with self.lock:
            return True, self.latest_frame.copy()

    def isOpened(self): return True

class SocketSensors:
    def __init__(self, port=12351):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        self.sock.listen(1)
        self.conn = None
        self.buffer = b''
        self.latest_data = (0.0, 0.0, 0, 0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0)
        print(f"📡 Mac listening for SENSORS on {port}...")
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        self.conn, addr = self.sock.accept()
        print(f"✅ Sensor Stream connected from {addr}!")

    def read(self):
        if not self.conn: return self.latest_data
        try:
            self.conn.setblocking(False) 
            while True:
                try:
                    chunk = self.conn.recv(4096) 
                    if not chunk: break 
                    self.buffer += chunk
                except BlockingIOError: break 
            while len(self.buffer) >= 256:
                msg = self.buffer[:256]
                self.buffer = self.buffer[256:] 
                data_str = msg.decode('utf-8').strip()
                parts = data_str.split(',')
                if len(parts) >= 14:
                    try:
                        self.latest_data = (
                            float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3]),     
                            float(parts[4]), float(parts[5]), float(parts[6]),    
                            float(parts[7]), float(parts[8]), float(parts[9]),    
                            float(parts[10]), float(parts[11]), float(parts[12]), int(parts[13]) 
                        )
                    except ValueError: pass 
        except Exception: pass
        return self.latest_data

def generate_top_down_map(depth_map, f_son, b_son, l_ir, r_ir, f_L_R, f_L_C, f_L_L, l_L_1, l_L_2, l_L_3, r_L_1, r_L_2, r_L_3, grid_size=500, max_depth_mm=4000):
    depth_map = cv2.medianBlur(depth_map, 5)
    h, w = depth_map.shape
    cx, cy, fx, fy = w / 2.0, h / 2.0, 525.0, 525.0 
    top_down_map = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    pepper_x, pepper_y = grid_size // 2, grid_size - 40
    cv2.circle(top_down_map, (pepper_x, pepper_y), 8, (0, 0, 255), -1) 
    
    rows, cols = np.indices((h, w))
    valid_mask = (depth_map > 500) & (depth_map < max_depth_mm)
    Z = depth_map[valid_mask]
    X_raw = (cols[valid_mask] - cx) * Z / fx
    Y_raw = (rows[valid_mask] - cy) * Z / fy 
    
    height_mask = (Y_raw > -500) & (Y_raw < 200) 
    final_X, final_Z = X_raw[height_mask], Z[height_mask]
    scale = grid_size / float(max_depth_mm)
    map_x = (final_X * scale + pepper_x).astype(int)
    map_y = (pepper_y - (final_Z * scale)).astype(int)
    
    in_bounds = (map_x >= 0) & (map_x < grid_size) & (map_y >= 0) & (map_y < grid_size)
    top_down_map[map_y[in_bounds], map_x[in_bounds]] = [255, 255, 255]
    top_down_map = cv2.dilate(top_down_map, np.ones((5,5), np.uint8), iterations=1)

    if 0.1 < f_son < 2.0:
        d_px = int(f_son * 1000 * scale)
        cv2.ellipse(top_down_map, (pepper_x, pepper_y), (d_px, d_px), 0, 250, 290, (0, 255, 255), 2)
    if 0.1 < b_son < 2.0:
        d_px = int(b_son * 1000 * scale)
        cv2.ellipse(top_down_map, (pepper_x, pepper_y), (d_px, d_px), 0, 70, 110, (0, 255, 255), 2)

    def draw_laser(dist, angle_deg):
        if dist < 4.0:
            dist_px = dist * 1000 * scale
            rad = math.radians(angle_deg)
            lx = int(pepper_x - math.sin(rad) * dist_px)
            ly = int(pepper_y - math.cos(rad) * dist_px)
            if 0 <= lx < grid_size and 0 <= ly < grid_size:
                cv2.circle(top_down_map, (lx, ly), 6, (255, 255, 0), -1)

    draw_laser(f_L_R, -20); draw_laser(f_L_C, 0); draw_laser(f_L_L, 20)
    draw_laser(l_L_1, 45); draw_laser(l_L_2, 65); draw_laser(l_L_3, 85)
    draw_laser(r_L_1, -45); draw_laser(r_L_2, -65); draw_laser(r_L_3, -85)

    if l_ir > 0.5:
        cv2.rectangle(top_down_map, (pepper_x - 60, pepper_y - 20), (pepper_x - 30, pepper_y), (0, 255, 255), -1)
        cv2.putText(top_down_map, "CLOSE L", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if r_ir > 0.5:
        cv2.rectangle(top_down_map, (pepper_x + 30, pepper_y - 20), (pepper_x + 60, pepper_y), (0, 255, 255), -1)
        cv2.putText(top_down_map, "CLOSE R", (410, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return top_down_map

class SocketTextServer:
    def __init__(self, port=12349):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', port))
        self.server_socket.listen(1)
        self.conn = None
        print(f"📡 Mac listening for PEPPER TEXT on {port}...")
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        self.conn, addr = self.server_socket.accept()
        self.conn.settimeout(5.0)
        print(f"✅ Text Stream connected from {addr}!")

    def read_text(self):
        if not self.conn: return None
        try:
            data = self.conn.recv(1024)
            if data: return data.decode('utf-8').strip()
        except socket.timeout: return None
        except Exception: pass
        return None

def log_attendance(name):
    global recently_logged
    t = time.time()
    if name in recently_logged and (t - recently_logged[name] < 300): return
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        c = conn.cursor()
        c.execute("INSERT INTO attendance (student_name, log_time) VALUES (%s, NOW())", (name,))
        conn.commit()
        conn.close()
        recently_logged[name] = t
        with open("db_logs.txt", "a") as f: f.write(f"[{datetime.now()}] ✅ SUCCESS: Logged {name}\n")
    except Exception as e:
        with open("db_logs.txt", "a") as f: f.write(f"[{datetime.now()}] ❌ ERROR: {e}\n")

def log_qna(name, question, answer):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        c = conn.cursor()
        c.execute("INSERT INTO qna_logs (student_name, question, answer, ask_time) VALUES (%s, %s, %s, NOW())", (name, question, answer))
        conn.commit()
        conn.close()
        with open("db_logs.txt", "a") as f: f.write(f"[{datetime.now()}] ✅ Q&A SAVED: {name} asked a question.\n")
    except Exception as e:
        with open("db_logs.txt", "a") as f: f.write(f"[{datetime.now()}] ❌ Q&A DB ERROR: {e}\n")

QUESTIONS_JSON_PATH = "/Users/User/Downloads/Trial 999 copy 6/Combined_Working/Combine trial last 2/Combine  trial/Untitled copy/html/data/questions.json"

def generate_quiz_from_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        c = conn.cursor()
        c.execute("SELECT question, answer FROM qna_logs ORDER BY ask_time DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        if not rows:
            print("⚠️ No Q&A history in DB — keeping existing questions.json")
            return

        history = "\n".join([f"Q: {q}\nA: {a}" for q, a in rows])
        prompt = f"""Using the student Q&A history below, generate EXACTLY 5 multiple-choice quiz questions that test the same topics.

HISTORY:
{history}

Return ONLY a JSON array of 5 objects with this exact schema:
- "heading": "Question N" where N is 1..5
- "text": the quiz question, MUST be 55 characters or fewer, must end with "?"
- "options": array of exactly 3 strings, prefixed "1) ", "2) ", "3) ". it shouldn't end with a interrogation mark.
- "correct": one of "1", "2", "3"
- "eventPrefix": "answer" for Q1, "answer2" for Q2, "answer3" for Q3, "answer4" for Q4. OMIT this field entirely on Q5.

No markdown, no commentary and make sure to not cut any word in the questions or answers — JSON array only."""

        new_questions = None
        for _attempt in range(3):
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.8, response_mime_type="application/json")
            )
            raw = response.text.strip()
            # Strip markdown fences if Gemini wraps the JSON
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            try:
                new_questions = json.loads(raw)
                if isinstance(new_questions, list) and len(new_questions) > 0:
                    break
                print(f"⚠️ Gemini returned empty or non-list JSON (attempt {_attempt+1})")
                new_questions = None
            except json.JSONDecodeError as je:
                print(f"⚠️ Gemini returned invalid JSON (attempt {_attempt+1}): {je}")
                new_questions = None

        if not new_questions:
            print("❌ Failed to get valid quiz JSON after 3 attempts — keeping existing questions.json")
            return

        for i, q in enumerate(new_questions):
            if len(q.get("text", "")) > 55:
                q["text"] = q["text"][:54].rstrip() + "?"
            q["heading"] = f"Question {i+1}"
            if i < 4:
                q["eventPrefix"] = "answer" if i == 0 else f"answer{i+1}"
            else:
                q.pop("eventPrefix", None)

        with open(QUESTIONS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(new_questions, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno()) # Forces Mac to write to physical disk
            f.close()
        print(f"✅ Quiz regenerated: {len(new_questions)} questions written to questions.json")
        time.sleep(1.5) # Ensure file system updates before Pepper reads it
    except Exception as e:
        print(f"❌ Quiz generation error: {e}")
        with open("db_logs.txt", "a") as f: f.write(f"[{datetime.now()}] ❌ QUIZ GEN ERROR: {e}\n")

def send_to_pepper_socket(command_string: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((PEPPER_IP, PORT)) 
        s.sendall(command_string.encode('utf-8'))
        s.close()
        return f"Sent: {command_string}"
    except Exception as e: return f"Socket Error: {e}"

def text_interaction_loop(user_name, text_server):
    global is_target_present, chat_thread_active, system_state, app_running
    chat_thread_active = True
    log_attendance(user_name)
    print(f"\n🌟 TRIGGER ACTIVATED: Hello {user_name}!")
    send_to_pepper_socket(f"Hello {user_name}, I am ready. What is your question?")
    
    tools = [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="send_to_pepper_socket",
            description="Use this ONLY to trigger physical movements, reset the system, move to the photo phase, or start the pop quiz. Valid commands are: 'high_five', 'move_forward_1', 'move_backward_1', 'move_left_1', 'move_right_1', 'reset_search', 'start_photo', 'start_quiz'.",
            parameters=types.Schema(type="OBJECT", properties={"command_string": types.Schema(type="STRING")}, required=["command_string"])
        )
    ])]

    instructions = f"""
    You are Pepper, a social humanoid robot located at USEK University.
    You are currently talking to a student named {user_name}.
    RULES:
    1. Be friendly, polite, and helpful. Keep responses short.
    2. To speak, JUST REPLY WITH PLAIN TEXT. Do not use the tool for talking. 
    3. KEEP IT SHORT. MAX 3 SENTENCES.
    4. ONLY use 'send_to_pepper_socket' if explicitly asked to move, high-five, OR to end the conversation.
    5. If the student says "thank you", "goodbye", indicates no more questions, or wants to take a picture, say a quick goodbye and IMMEDIATELY use the 'send_to_pepper_socket' tool with the command 'start_quiz'.
    6. You are to answer anything related to artificial intelligence and technology, but if the question is not related to those topics, politely tell the student that you can only answer questions about AI and technology.
    """

    config = types.GenerateContentConfig(system_instruction=instructions, tools=tools, temperature=0.7)
    chat = client.chats.create(model=MODEL_ID, config=config)
    
    while is_target_present and app_running:
        print("🤖 Waiting for text from Pepper...")
        query = text_server.read_text()
        
        if query:
            print(f"💬 You said: {query}")
            send_to_pepper_socket(f"show_t:{user_name}:{query}")
            try:
                response = chat.send_message(query)
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        cmd = part.function_call.args["command_string"]
                        if cmd == "start_photo":
                            print(f"📸 AI triggered photo phase! Shutting down Mac Bridge...")
                            send_to_pepper_socket("start_photo")
                            is_target_present = False
                            app_running = False
                        elif cmd == "start_quiz":
                            print(f"📝 AI triggered pop quiz! Regenerating questions from DB...")
                            generate_quiz_from_db()
                            
                            send_to_pepper_socket("start_quiz")
                            is_target_present = False
                            app_running = False
                        elif cmd == "reset_search":
                            print(f"👋 Moving to next student.")
                            send_to_pepper_socket("reset_search") 
                            is_target_present = False 
                        else:
                            print(f"⚙️ Physical Action Triggered: {cmd}")
                            send_to_pepper_socket(cmd)
                    elif part.text:
                        clean_text = part.text.strip().replace("*", "").replace("#", "").replace("_", "").replace("`", "").replace("\n", ". ")
                        print(f"🗣️ Gemini says: {clean_text}")
                        send_to_pepper_socket(f"show_t:Pepper:{clean_text}")
                        send_to_pepper_socket(clean_text)
                        log_qna(user_name, query, clean_text)
            except Exception as e: print(f"❌ Gemini AI Error: {e}")
        time.sleep(0.1) 

    if app_running:
        print(f"\n👋 Resetting to SEARCHING.")
        time.sleep(3.0) 
        send_to_pepper_socket("Who else has a question?")
        chat_thread_active = False
        system_state = "SEARCHING"

# --- 5. SETUP MODELS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'pose_landmarker_heavy.task')
PEPPER_IP = '10.10.42.80' 
PORT = 12344

detector = mp.tasks.vision.PoseLandmarker.create_from_options(
    mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )
)

try:
    charbel_image = face_recognition.load_image_file(os.path.join(script_dir, "student_images/charbel.jpeg"))
    kevin_image = face_recognition.load_image_file(os.path.join(script_dir, "student_images/kevin.jpeg"))
    joseph_image = face_recognition.load_image_file(os.path.join(script_dir, "student_images/joseph.jpeg"))
    maria_image = face_recognition.load_image_file(os.path.join(script_dir, "student_images/maria.jpeg"))
    tatyana_image = face_recognition.load_image_file(os.path.join(script_dir, "student_images/tatyana.jpeg"))
    known_face_encodings = [face_recognition.face_encodings(charbel_image)[0], face_recognition.face_encodings(kevin_image)[0], face_recognition.face_encodings(joseph_image)[0], face_recognition.face_encodings(maria_image)[0], face_recognition.face_encodings(tatyana_image)[0]]
    known_face_names = ["Charbel", "Kevin", "Joseph", "Maria", "Tatyana"]
except Exception as e: print(f"❌ Face Init Error: {e}"); exit()

# --- 6. MAIN VISION LOOP ---
rgb_capture = SocketCamera(port=12348, stream_type="RGB")
depth_capture = SocketCamera(port=12350, stream_type="DEPTH")
text_receiver = SocketTextServer(port=12349)
sensor_capture = SocketSensors(port=12351)

cv2.namedWindow('Pepper Brain (AI)', cv2.WINDOW_NORMAL) 
cv2.resizeWindow('Pepper Brain (AI)', 500, 500) 
cv2.namedWindow('Pepper Top-Down Map', cv2.WINDOW_NORMAL) 
cv2.resizeWindow('Pepper Top-Down Map', 500, 500) 

system_state = "SEARCHING" 
is_target_present = False
chat_thread_active = False
last_seen_time = 0
frame_count = 0 
move_start_time = 0
expected_travel_time = 0
identify_start_time = 0 
nav_status = "👀 Waiting for raised hand..."
current_target_name = "Student"

while app_running and rgb_capture.isOpened() and depth_capture.isOpened():
    ret_rgb, frame = rgb_capture.read()
    ret_depth, depth_map = depth_capture.read()
    
    (f_son, b_son, l_ir, r_ir, f_L_R, f_L_C, f_L_L, l_L_1, l_L_2, l_L_3, r_L_1, r_L_2, r_L_3, heartbeat) = sensor_capture.read()

    print("\033[H\033[J", end="") 
    print(f"=== PEPPER SENSOR DASHBOARD | TICK: {heartbeat} ===")
    print(f"🔊 SONAR -> Front: {f_son:.2f}m | Back: {b_son:.2f}m")
    print(f"🔴 IR    -> Left:  {l_ir}     | Right: {r_ir}")
    print(f"🤖 AI STATUS: {nav_status}")
    print(f"⚡ F_LAS -> Right: {f_L_R:.2f}m | Center: {f_L_C:.2f}m | Left: {f_L_L:.2f}m")
    print("=============================================")

    if not ret_rgb or frame is None:
        cv2.waitKey(1); continue
        
    if ret_depth and depth_map is not None:
        top_down_img = generate_top_down_map(
            depth_map, f_son, b_son, l_ir, r_ir, f_L_R, f_L_C, f_L_L, l_L_1, l_L_2, l_L_3, r_L_1, r_L_2, r_L_3, grid_size=500, max_depth_mm=4000)
        cv2.imshow('Pepper Top-Down Map', top_down_img)
    
    frame_count += 1
    current_time = time.time()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    if system_state == "SEARCHING":
        result = detector.detect_for_video(mp_image, int(current_time * 1000))
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            if lm[15].y < (lm[11].y - 0.1) or lm[16].y < (lm[12].y - 0.1):
                px = int((lm[11].x + lm[12].x) / 2.0 * frame.shape[1])
                py = int((lm[11].y + lm[12].y) / 2.0 * frame.shape[0]) + 40 
                cv2.circle(frame, (px, py), 40, (0, 255, 255), 2) 
                
                if ret_depth and depth_map is not None:
                    px = max(0, min(px, depth_map.shape[1] - 1))
                    py = max(0, min(py, depth_map.shape[0] - 1))
                    box_size = 40 
                    y1, y2 = max(0, py - box_size), min(depth_map.shape[0], py + box_size)
                    x1, x2 = max(0, px - box_size), min(depth_map.shape[1], px + box_size)
                    depth_patch = depth_map[y1:y2, x1:x2]
                    valid_depths = depth_patch[(depth_patch > 500) & (depth_patch < 4000)]
                    
                    if len(valid_depths) > 20: 
                        dist_mm = np.median(valid_depths)
                        dist_m = dist_mm / 1000.0
                        angle_deg = max(-25.0, min(25.0, (px - 320) * (60.0 / 640.0) * 0.2))  
                        angle_rad = math.radians(angle_deg)
                        target_x = (dist_m * math.cos(angle_rad)) - 0.6
                        target_y = 0.0
                        check_dist = min(target_x, 1.0) 
                        is_path_clear = True
                        
                        if f_L_C < check_dist or f_son < check_dist: is_path_clear = False
                        if target_y > 0.2 and f_L_L < check_dist: is_path_clear = False
                        if target_y < -0.2 and f_L_R < check_dist: is_path_clear = False
                        
                        if target_x <= 0.1: nav_status = "⚠️ You are too close! Step back."
                        elif not is_path_clear: nav_status = f"🛑 Immediate Path Blocked! Need {check_dist:.2f}m clear to start."
                        else:
                            nav_status = f"🎯 TARGET LOCKED! Moving X: {target_x:.2f}m"
                            send_to_pepper_socket(f"navto_{target_x:.2f}_{target_y:.2f}")
                            system_state = "NAVIGATING"
                            move_start_time = current_time
                            expected_travel_time = (dist_m / 0.3) + 1.0

    elif system_state == "NAVIGATING":
        DANGER_ZONE = 0.35 
        
        # ---> FIX 4: 1.5s Grace Period to ignore tilt noise, and filter out 0.0 values <---
        obs_sonar = (0.0 < f_son < DANGER_ZONE)
        obs_laser = (0.0 < f_L_C < DANGER_ZONE) or (0.0 < f_L_L < DANGER_ZONE) or (0.0 < f_L_R < DANGER_ZONE)
        obs_ir = (l_ir > 0 or r_ir > 0)
        
        if (current_time - move_start_time > 1.5) and (obs_sonar or obs_laser or obs_ir):
            print("\n🚨 EMERGENCY BRAKE! Dynamic obstacle detected!")
            send_to_pepper_socket("stop")
            system_state = "SEARCHING" 
            time.sleep(2.0) # ---> FIX 5: Human cooldown so it doesn't instantly restart <---
            
        elif current_time - move_start_time > expected_travel_time:
            print("\n🏁 Arrived! Looking for face...")
            system_state = "IDENTIFYING"
            identify_start_time = current_time 

    elif system_state in ["IDENTIFYING", "INTERACTING"]:
        target_seen_this_frame = False
        if frame_count % 5 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_small)
            face_encs = face_recognition.face_encodings(rgb_small, face_locs)
            
            for face_encoding in face_encs:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
                if True in matches:
                    current_target_name = known_face_names[matches.index(True)]
                    target_seen_this_frame = True
                    last_seen_time = current_time
                    break 

        if system_state == "IDENTIFYING":
            if target_seen_this_frame:
                system_state = "INTERACTING"
                is_target_present = True
                if not chat_thread_active:
                    threading.Thread(target=text_interaction_loop, args=(current_target_name, text_receiver), daemon=True).start()
            # ---> FIX 6: Prevent getting stuck in IDENTIFYING forever <---
            elif current_time - identify_start_time > 10.0:
                print("⚠️ No face detected. Resetting to SEARCHING.")
                system_state = "SEARCHING"
                
        elif system_state == "INTERACTING" and not target_seen_this_frame:
            if current_time - last_seen_time > 10.0:
                is_target_present = False

    color = (0, 255, 255) if system_state == "NAVIGATING" else ((0, 255, 0) if system_state == "INTERACTING" else (0, 0, 255))
    cv2.putText(frame, f"State: {system_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    display_frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Pepper Brain (AI)', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break


print("\n🔌 Shutting down... Sending final stop command to Pepper.")
send_to_pepper_socket("stop")
time.sleep(0.5)
detector.close(); cv2.destroyAllWindows()