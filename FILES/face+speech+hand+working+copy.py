import os
import cv2
import socket
import time
import threading
import mediapipe as mp
import face_recognition
import speech_recognition as sr
import numpy as np
import mysql.connector
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. INITIALIZATION ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
MODEL_ID = 'gemini-2.5-flash-lite'
recognizer = sr.Recognizer()

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '', 
    'database': 'classroom_db'
}

recently_logged = {}

# --- 2. SOCKET CAMERA CLASS ---
class SocketCamera:
    def __init__(self, port=12348):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', port)) 
        self.server_socket.listen(1)
        self.conn = None
        self.is_open = False
        print(f"📡 Mac listening for COMPRESSED stream on {port}...")
        threading.Thread(target=self._accept_connection, daemon=True).start()

    def _accept_connection(self):
        self.conn, addr = self.server_socket.accept()
        # TCP_NODELAY ensures frames aren't buffered by the OS
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"✅ Fast Stream connected from {addr}!")
        self.is_open = True

    def recvall(self, count):
        buf = b''
        while count:
            newbuf = self.conn.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def read(self):
        if not self.is_open or not self.conn: return False, None
        try:
            header = self.recvall(16)
            if not header: return False, None
            size = int(header.decode('utf-8').strip())
            jpeg_data = self.recvall(size)
            if not jpeg_data: return False, None
            nparr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return True, frame
        except Exception as e:
            print(f"❌ Socket Read Error: {e}")
            return False, None

    def isOpened(self):
        return True
    
    
# --- 3. HELPER FUNCTIONS ---
def log_attendance(name):
    global recently_logged
    t = time.time()
    if name in recently_logged and (t - recently_logged[name] < 300): 
        return
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        c = conn.cursor()
        c.execute("INSERT INTO attendance (student_name) VALUES (%s)", (name,))
        conn.commit()
        conn.close()
        recently_logged[name] = t
        print(f"📝 Logged: {name} into database.")
    except Exception as e:
        print(f"❌ Database Error: {e}")

def send_to_pepper_socket(command_string: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((PEPPER_IP, PORT)) 
        s.sendall(command_string.encode('utf-8'))
        s.close()
        return f"Sent: {command_string}"
    except Exception as e:
        return f"Socket Error: {e}"

# --- 4. AUDIO INTERACTION ---
def audio_interaction_loop(user_name):
    global is_target_present, chat_thread_active, system_state
    chat_thread_active = True
    log_attendance(user_name)
    
    print(f"\n🌟 TRIGGER ACTIVATED: Hello {user_name}!")
    send_to_pepper_socket(f"Hello {user_name}, I am ready. What is your question?")
    
    tools = [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="send_to_pepper_socket",
            description="Moves the robot, does a high five, or makes it speak.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"command_string": types.Schema(type="STRING")},
                required=["command_string"]
            )
        )
    ])]

    instructions = f"""
    You are Pepper, a social humanoid robot located at USEK University.
    You are currently talking to a student named {user_name}.
    
    RULES:
    1. Be friendly, polite, and helpful.
    2. Keep your spoken responses very short (maximum 2 sentences).
    3. Use the 'send_to_pepper_socket' tool for ALL communication. 
    4. If the user asks you to move, high-five, or wave, use the tool.
    5. Always maintain the persona of a helpful campus assistant.
    6. If asked about maria , tell that "Maria is always right!!"
    """

    config = types.GenerateContentConfig(
        system_instruction=instructions, # <--- Instructions applied here
        tools=tools, 
        temperature=0.7
    )
    
    chat = client.chats.create(model=MODEL_ID, config=config)
    
    with sr.Microphone() as source:
        # --- AUDIO FIX: SENSITIVITY CALIBRATION ---
        print("🎤 Calibrating microphone for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        recognizer.energy_threshold = 300  # Lower this to 100 if it still doesn't hear you
        
        while is_target_present:
            try:
                print("🎤 Listening... (Wait for prompt)")
                # phrase_time_limit prevents the AI from waiting forever
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=8)
                
                print("🧠 Recognizing...")
                query = recognizer.recognize_google(audio)
                print(f"💬 You said: {query}")

                response = chat.send_message(query)

                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        cmd = part.function_call.args["command_string"]
                        print(f"⚙️ Action: {cmd}")
                        send_to_pepper_socket(cmd)
                    elif part.text:
                        send_to_pepper_socket(part.text.strip())

            except sr.WaitTimeoutError:
                continue 
            except sr.UnknownValueError:
                print("❓ Could not understand audio.")
                continue
            except Exception as e:
                print(f"❌ Session Error: {e}")
                time.sleep(1) 

    print(f"\n👋 Resetting to SEARCHING.")
    send_to_pepper_socket("Goodbye!")
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
    my_image = face_recognition.load_image_file(os.path.join(script_dir, "Charbel.jpeg"))
    known_face_encodings = [face_recognition.face_encodings(my_image)[0]]
    known_face_names = ["Charbel"]
except Exception as e:
    print(f"❌ Face Init Error: {e}"); exit()

# --- 6. MAIN VISION LOOP ---
video_capture = SocketCamera(port=12348)
system_state = "SEARCHING" 
is_target_present = False
chat_thread_active = False
last_seen_time = 0
BUFFER_TIME = 2.0 
frame_count = 0  # To skip face recognition frames
identified_name = "Unknown" # Default name until recognized 
#if you want to redo from here

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        cv2.waitKey(1); continue
    
    frame_count += 1
    current_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # STATE 1: SEARCHING (Hand Raise)
    if system_state == "SEARCHING":
        result = detector.detect_for_video(mp_image, int(current_time * 1000))
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            if lm[15].y < (lm[11].y - 0.1) or lm[16].y < (lm[12].y - 0.1):
                system_state = "IDENTIFYING"

    # STATE 2: IDENTIFYING (Only run Face Rec every 5 frames to keep FPS high)
    elif system_state in ["IDENTIFYING", "INTERACTING"]:
        target_seen_this_frame = False
        
        if frame_count % 5 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_small)
            face_encs = face_recognition.face_encodings(rgb_small, face_locs)
            
            for face_encoding in face_encs:
                if True in face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5):
                    target_seen_this_frame = True
                    last_seen_time = current_time

        if system_state == "IDENTIFYING" and target_seen_this_frame:
            system_state = "INTERACTING"
            is_target_present = True
            if not chat_thread_active:
                threading.Thread(target=audio_interaction_loop, args=(known_face_names[0],), daemon=True).start()
        
        elif system_state == "INTERACTING" and not target_seen_this_frame:
            if current_time - last_seen_time > BUFFER_TIME:
                is_target_present = False

    # Overlay & Display
    color = (0, 255, 0) if system_state == "INTERACTING" else (0, 0, 255)
    cv2.putText(frame, f"State: {system_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow('Pepper Brain', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

detector.close(); cv2.destroyAllWindows()
