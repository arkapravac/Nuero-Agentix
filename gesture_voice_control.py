import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import speech_recognition as sr
import pyttsx3
import threading
import keyboard
from queue import Queue

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize Speech Recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Initialize MediaPipe with optimized settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lowered for better detection
    min_tracking_confidence=0.5,    # Lowered for smoother tracking
    model_complexity=0  # Use simpler model for faster processing
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

# Control variables
frame_reduction = 150  # Increased edge buffer for smoother boundary handling
smoothing_alpha = 0.15  # Reduced alpha for stronger smoothing
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
is_voice_control_active = False
is_gesture_control_active = True

# Frame processing variables
frame_buffer = []
buffer_size = 5  # Increased frame averaging for stability
movement_threshold = 2.0  # Minimum movement threshold to reduce jitter

def speak(text):
    engine.say(text)
    engine.runAndWait()

def count_fingers(landmarks):
    fingers = []
    # Thumb
    if landmarks[4][0] > landmarks[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers
    for tip in range(8, 21, 4):
        if landmarks[tip][1] < landmarks[tip-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def listen_for_commands():
    global is_voice_control_active, is_gesture_control_active
    
    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                print("শুনছি...")
                audio = recognizer.listen(source)
                
                try:
                    command = recognizer.recognize_google(audio, language='bn-BD').lower()
                    print(f"শোনা গেছে: {command}")
                    
                    if "মাউস" in command:
                        if "চালু" in command:
                            is_gesture_control_active = True
                            speak("হ্যান্ড জেসচার কন্ট্রোল চালু করা হয়েছে")
                        elif "বন্ধ" in command:
                            is_gesture_control_active = False
                            speak("হ্যান্ড জেসচার কন্ট্রোল বন্ধ করা হয়েছে")
                    
                    elif "ক্লিক" in command:
                        pyautogui.click()
                        speak("ক্লিক করা হয়েছে")
                    
                    elif "ডাবল ক্লিক" in command:
                        pyautogui.doubleClick()
                        speak("ডাবল ক্লিক করা হয়েছে")
                    
                    elif "রাইট ক্লিক" in command:
                        pyautogui.rightClick()
                        speak("রাইট ক্লিক করা হয়েছে")
                    
                    elif "স্ক্রল" in command:
                        if "উপরে" in command:
                            pyautogui.scroll(100)
                        elif "নিচে" in command:
                            pyautogui.scroll(-100)
                    
                    elif "বন্ধ" in command and "প্রোগ্রাম" in command:
                        speak("প্রোগ্রাম বন্ধ করা হচ্ছে")
                        break
                        
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    speak("দুঃখিত, স্পিচ রিকগনিশন সার্ভিবৃত সমস্যা হচ্ছে")
                    
        except Exception as e:
            print(f"Error: {str(e)}")

# Start voice recognition in a separate thread
voice_thread = threading.Thread(target=listen_for_commands)
voice_thread.daemon = True
voice_thread.start()

speak("প্রোগ্রাম চালু হয়েছে")

def process_frame(frame):
    if frame is None:
        return None
    # Normalize frame size and enhance quality
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.GaussianBlur(frame, (5,5), 0)  # Reduce noise
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Enhance contrast and brightness
    
    # Enhanced noise reduction pipeline
    frame = cv2.GaussianBlur(frame, (7, 7), 0)  # Increased blur kernel
    frame = cv2.medianBlur(frame, 5)  # Additional median filter for noise reduction
    
    # Add to frame buffer for temporal smoothing
    frame_buffer.append(frame)
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)
    
    # Average frames if buffer is full
    if len(frame_buffer) == buffer_size:
        frame = np.mean(frame_buffer, axis=0).astype(np.uint8)
    
    return frame
def smooth_coordinates(x, y, prev_x, prev_y, alpha=0.15):
    # Enhanced exponential moving average with movement threshold
    delta_x = x - prev_x
    delta_y = y - prev_y
    
    # Apply threshold to reduce micro-movements
    if abs(delta_x) < movement_threshold:
        delta_x = 0
    if abs(delta_y) < movement_threshold:
        delta_y = 0
    
    # Apply smoothing with dynamic alpha
    smoothed_x = prev_x + alpha * delta_x
    smoothed_y = prev_y + alpha * delta_y
    
    return smoothed_x, smoothed_y

# Initialize voice queue for non-blocking narration
voice_queue = Queue()
last_position_announcement = 0
position_announcement_delay = 1.0  # Announce position every 1 second

def speak_worker():
    while True:
        text = voice_queue.get()
        if text == "STOP":
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {str(e)}")
        voice_queue.task_done()

def speak_non_blocking(text, priority=False):
    if priority:
        # Clear queue for priority messages
        while not voice_queue.empty():
            voice_queue.get()
    voice_queue.put(text)

# Start speech thread
speech_thread = threading.Thread(target=speak_worker, daemon=True)
speech_thread.start()

def main():
    try:
        while True:
            success, img = cap.read()
            if not success:
                continue
                
            img = cv2.flip(img, 1)
            img = process_frame(img)  # Apply frame processing
            
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            
            if is_gesture_control_active and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarks.append([cx, cy])
                    
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    fingers = count_fingers(landmarks)
                    total_fingers = sum(fingers)
                    
                    index_x = landmarks[8][0]
                    index_y = landmarks[8][1]
                    # Inside main() function where coordinates are processed:
                    # Improved coordinate mapping with better edge detection
                    screen_x = int(np.interp(index_x, (frame_reduction, 1280 - frame_reduction), (0, screen_width)))
                    screen_y = int(np.interp(index_y, (frame_reduction, 720 - frame_reduction), (0, screen_height)))
                    
                    # Apply smoother coordinate transitions with optimized alpha
                    curr_x, curr_y = smooth_coordinates(screen_x, screen_y, prev_x, prev_y, smoothing_alpha)
                    
                    if total_fingers == 1 and fingers[1] == 1:
                        pyautogui.moveTo(curr_x, curr_y, duration=0.08)  # Slightly reduced duration for better responsiveness
                        # Announce position periodically with lower priority
                        current_time = time.time()
                        if current_time - last_position_announcement > position_announcement_delay:
                            speak_non_blocking(f"মাউস পজিশন: {int(curr_x)}, {int(curr_y)}", priority=False)
                            last_position_announcement = current_time
                    elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                        pyautogui.click()
                        speak_non_blocking("ক্লিক করা হচ্ছে", priority=True)
                        time.sleep(0.3)
                    elif total_fingers == 3:
                        pyautogui.rightClick()
                        speak_non_blocking("রাইট ক্লিক করা হচ্ছে", priority=True)
                        time.sleep(0.3)
                    elif total_fingers == 4:
                        pyautogui.doubleClick()
                        speak_non_blocking("ডাবল ক্লিক করা হচ্ছে", priority=True)
                        time.sleep(0.3)
                    elif total_fingers == 5:
                        # Scroll mode
                        scroll_delta = curr_y - prev_y
                        scroll_sensitivity = 0.3
                        scroll_amount = int(scroll_delta * scroll_sensitivity)
                        if abs(scroll_amount) > 2:
                            pyautogui.scroll(scroll_amount)
                            speak_non_blocking("স্ক্রল করা হচ্ছে", priority=False)
                            time.sleep(0.02)
                    
                    prev_x, prev_y = curr_x, curr_y
            
            # Normalize display window
            display_img = cv2.resize(img, (960, 540))  # Consistent display size
            
            # Show status on screen
            status = "জেসচার কন্ট্রোল: চালু" if is_gesture_control_active else "জেসচার কন্ট্রোল: বন্ধ"
            cv2.putText(display_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Gesture Control", display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                speak_non_blocking("প্রোগ্রাম বন্ধ করা হচ্ছে")
                break

    finally:
        voice_queue.put("STOP")
        speech_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()