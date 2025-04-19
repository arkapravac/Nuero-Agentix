import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import speech_recognition as sr
import threading
import keyboard
from googletrans import Translator
from gtts import gTTS
import os
import playsound
import random
import datetime

# Initialize translator
translator = Translator()

# Initialize Speech Recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

# Control variables
frame_reduction = 100
smoothening = 7
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
is_voice_control_active = False
is_gesture_control_active = True
last_speech_time = time.time()

# Bangla responses
greetings = [
    "হ্যালো! আমি আপনার সহকারী",
    "কেমন আছেন?",
    "আমি আপনাকে কিভাবে সাহায্য করতে পারি?",
    "স্বাগতম!"
]

responses = {
    "কেমন আছো": ["ভালো আছি, আপনি কেমন আছেন?", "আমি সব সময়ই ভালো থাকি!"],
    "তুমি কি করতে পারো": ["আমি আপনার কম্পিউটার কন্ট্রোল করতে পারি, ভয়েস কমান্ড বুঝতে পারি, এবং আপনার সাথে কথা বলতে পারি।"],
    "তোমার নাম কি": ["আমার নাম স্মার্ট অ্যাসিস্ট্যান্ট", "আপনি আমাকে যে নামে ডাকবেন সুই নামেই সাড়া দেব"],
    "ধন্যবাদ": ["আপনাকেও ধন্যবাদ!", "সাহায্য করতে পেরে খুশি হলাম"],
}

# Initialize voice feedback queue and thread
from queue import Queue
voice_queue = Queue()

def voice_worker():
    while True:
        text = voice_queue.get()
        if text is None:
            break
        try:
            tts = gTTS(text=text, lang='bn')
            temp_file = f"temp_voice_{int(time.time())}.mp3"
            tts.save(temp_file)
            playsound.playsound(temp_file)
            os.remove(temp_file)
        except Exception as e:
            print(f"Error in speech: {str(e)}")
        voice_queue.task_done()

# Start voice worker thread
voice_thread = threading.Thread(target=voice_worker)
voice_thread.daemon = True
voice_thread.start()

def speak_bengali(text):
    voice_queue.put(text)
    try:
        tts = gTTS(text=text, lang='bn')
        temp_file = "temp_voice.mp3"
        tts.save(temp_file)
        playsound.playsound(temp_file)
        os.remove(temp_file)
    except Exception as e:
        print(f"Error in speech: {str(e)}")

def get_response(text):
    text = text.lower()
    
    # Check for greetings
    if any(word in text for word in ["হাই", "হ্যালো", "হেলো"]):
        return random.choice(greetings)
    
    # Check for time
    if "কয়টা" in text or "সময়" in text:
        now = datetime.datetime.now()
        return f"এখন সময় {now.hour}টা {now.minute} মিনিট"
    
    # Check for predefined responses
    for key in responses:
        if key in text:
            return random.choice(responses[key])
    
    # Default response
    return "দুঃখিত, আমি আপনার কথা বুঝতে পারিনি। আবার বলুন?"

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
    global is_voice_control_active, is_gesture_control_active, last_speech_time
    
    speak_bengali("হ্যালো! আমি আপনার স্মার্ট অ্যাসিস্ট্যান্ট। আমি আপনাকে কিভাবে সাহায্য করতে পারি?")
    
    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                print("শুনছি...")
                audio = recognizer.listen(source)
                
                try:
                    command = recognizer.recognize_google(audio, language='bn-BD').lower()
                    print(f"শোনা গেছে: {command}")
                    current_time = time.time()
                    
                    # Handle system commands
                    if "মাউস" in command:
                        if "চালু" in command:
                            is_gesture_control_active = True
                            speak_bengali("হ্যান্ড জেসচার কন্ট্রোল চালু করা হয়েছে")
                        elif "বন্ধ" in command:
                            is_gesture_control_active = False
                            speak_bengali("হ্যান্ড জেসচার কন্ট্রোল বন্ধ করা হয়েছে")
                    
                    elif "ক্লিক" in command:
                        pyautogui.click()
                        speak_bengali("ক্লিক করা হয়েছে")
                    
                    elif "ডাবল ক্লিক" in command:
                        pyautogui.doubleClick()
                        speak_bengali("ডাবল ক্লিক করা হয়েছে")
                    
                    elif "রাইট ক্লিক" in command:
                        pyautogui.rightClick()
                        speak_bengali("রাইট ক্লিক করা হয়েছে")
                    
                    elif "স্ক্রল" in command:
                        if "উপরে" in command:
                            pyautogui.scroll(100)
                        elif "নিচে" in command:
                            pyautogui.scroll(-100)
                    
                    elif "বন্ধ" in command and "প্রোগ্রাম" in command:
                        speak_bengali("প্রোগ্রাম বন্ধ করা হচ্ছে")
                        break
                    
                    # Handle conversation
                    elif current_time - last_speech_time > 1:  # Prevent too frequent responses
                        response = get_response(command)
                        speak_bengali(response)
                        last_speech_time = current_time
                        
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    speak_bengali("দুঃখিত, স্পিচ রিকগনিশন সার্ভিসে সমস্যা হচ্ছে")
                    
        except Exception as e:
            print(f"Error: {str(e)}")

# Start voice recognition in a separate thread
voice_thread = threading.Thread(target=listen_for_commands)
voice_thread.daemon = True
voice_thread.start()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
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
            
            screen_x = int(np.interp(index_x, (frame_reduction, 640 - frame_reduction), (0, screen_width)))
            screen_y = int(np.interp(index_y, (frame_reduction, 480 - frame_reduction), (0, screen_height)))
            
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            
            if total_fingers == 1 and fingers[1] == 1:
                pyautogui.moveTo(curr_x, curr_y)
                if abs(curr_x - prev_x) > 50 or abs(curr_y - prev_y) > 50:
                    voice_queue.put("মাউস মুভ করছে")
            elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                pyautogui.click()
                voice_queue.put("ক্লিক করা হচ্ছে")
                time.sleep(0.3)
            elif total_fingers == 3:
                pyautogui.rightClick()
                voice_queue.put("রাইট ক্লিক করা হচ্ছে")
                time.sleep(0.3)
            elif total_fingers == 4:
                pyautogui.doubleClick()
                voice_queue.put("ডাবল ক্লিক করা হচ্ছে")
                time.sleep(0.3)
            elif total_fingers == 5:
                scroll_distance = prev_y - curr_y
                scroll_speed = int(np.interp(abs(scroll_distance), (0, 100), (20, 200)))
                if scroll_distance > 0:
                    pyautogui.scroll(scroll_speed)
                    if abs(scroll_distance) > 30:  # Only announce for significant movements
                        voice_queue.put("স্ক্রল উপরে")
                else:
                    pyautogui.scroll(-scroll_speed)
                    if abs(scroll_distance) > 30:  # Only announce for significant movements
                        voice_queue.put("স্ক্রল নিচে")
                time.sleep(0.2)  # Reduced delay for smoother scrolling
            
            prev_x, prev_y = curr_x, curr_y
    
    # Show status on screen
    status = "জেসচার কন্ট্রোল: চালু" if is_gesture_control_active else "জেসচার কন্ট্রোল: বন্ধ"
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Smart Assistant", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak_bengali("প্রোগ্রাম বন্ধ করা হচ্ছে। আবার দেখা হবে!")
        break

cap.release()
cv2.destroyAllWindows()

def speak_action(action_text):
    voice_queue.put(action_text)

def on_scroll(direction):
    if direction > 0:
        speak_action("স্ক্রল উপরে")
    else:
        speak_action("স্ক্রল নিচে")

def on_window_action(action_type, window_name):
    if action_type == "minimize":
        speak_action(f"{window_name} উইন্ডো মিনিমাইজ করা হয়েছে")
    elif action_type == "maximize":
        speak_action(f"{window_name} উইন্ডো ম্যাক্সিমাইজ করা হয়েছে")

def on_file_action(action_type, file_name):
    if action_type == "open":
        speak_action(f"{file_name} খোলা হচ্ছে")
    elif action_type == "close":
        speak_action(f"{file_name} বন্ধ করা হচ্ছে")
# Inside the main loop where scroll is detected
if len(fingers) == 5:  # All fingers up for scrolling
    if landmarks[8][1] < prev_y:  # Moving up
        pyautogui.scroll(50)
        on_scroll(1)
    elif landmarks[8][1] > prev_y:  # Moving down
        pyautogui.scroll(-50)
        on_scroll(-1)