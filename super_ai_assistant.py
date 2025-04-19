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
import wikipedia
import wolframalpha
import requests
import json
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
from transformers import pipeline
import torch
from tensorflow.keras.models import load_model
import webbrowser

# Load environment variables
load_dotenv()

# Initialize OpenAI (needs API key in .env file)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize various AI models
sentiment_analyzer = pipeline("sentiment-analysis")
question_answerer = pipeline("question-answering")
summarizer = pipeline("summarization")

# Initialize translator
translator = Translator()
wikipedia.set_lang('bn')  # Set Wikipedia language to Bengali

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

class AIAssistant:
    def __init__(self):
        self.conversation_history = []
        self.wolfram_client = wolframalpha.Client(os.getenv('WOLFRAM_APP_ID'))
        self.last_response = ""
    
    def get_weather(self, city):
        api_key = os.getenv('WEATHER_API_KEY')
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data["cod"] != "404":
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"{city}তে বর্তমান তাপমাত্রা {temp}°C। আবহাওয়া: {desc}"
        return "দুঃখিত, আবহাওয়ার তথ্য পাওয়া যায়নি।"

    def search_wikipedia(self, query):
        try:
            result = wikipedia.summary(query, sentences=2)
            return translator.translate(result, dest='bn').text
        except:
            return "দুঃখিত, উক্ত বিষয়ে তথ্য পাওয়া যায়নি।"

    def solve_math(self, query):
        try:
            res = self.wolfram_client.query(query)
            answer = next(res.results).text
            return f"উত্তর: {answer}"
        except:
            return "দুঃখিত, গণনা করতে সমস্যা হচ্ছে।"

    def analyze_sentiment(self, text):
        result = sentiment_analyzer(text)[0]
        if result['label'] == 'POSITIVE':
            return "আপনার কথায় ইতিবাচক মনোভাব প্রকাশ পেয়েছে।"
        else:
            return "আপনার কথায় নেতিবাচক মনোভাব প্রকাশ পেয়েছে।"

    def chat_with_gpt(self, text):
        try:
            self.conversation_history.append({"role": "user", "content": text})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history
            )
            ai_response = response.choices[0].message['content']
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            return translator.translate(ai_response, dest='bn').text
        except:
            return "দুঃখিত, GPT মডেলের সাথে যোগাযোগ করতে সমস্যা হচ্ছে।"

    def process_command(self, command):
        command = command.lower()
        
        # Weather related queries
        if "আবহাওয়া" in command:
            city = "ঢাকা"  # Default city
            if "এর" in command:
                city = command.split("এর")[0].strip()
            return self.get_weather(city)
        
        # Wikipedia related queries
        elif "কি" in command or "কে" in command:
            search_term = command.replace("কি", "").replace("কে", "").strip()
            return self.search_wikipedia(search_term)
        
        # Math related queries
        elif any(word in command for word in ["যোগ", "বিয়োগ", "গুণ", "ভাগ", "সমাধান"]):
            return self.solve_math(command)
        
        # Web search
        elif "সার্চ" in command or "খোঁজ" in command:
            search_term = command.replace("সার্চ", "").replace("খোঁজ", "").strip()
            webbrowser.open(f"https://www.google.com/search?q={search_term}")
            return "গুগল সার্চ করা হচ্ছে..."
        
        # Sentiment analysis
        elif "আমার মনোভাব" in command:
            return self.analyze_sentiment(command)
        
        # Default to GPT conversation
        else:
            return self.chat_with_gpt(command)

# Initialize AI Assistant
ai_assistant = AIAssistant()

def speak_bengali(text):
    try:
        tts = gTTS(text=text, lang='bn')
        temp_file = "temp_voice.mp3"
        tts.save(temp_file)
        playsound.playsound(temp_file)
        os.remove(temp_file)
    except Exception as e:
        print(f"Error in speech: {str(e)}")

def count_fingers(landmarks):
    fingers = []
    if landmarks[4][0] > landmarks[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    for tip in range(8, 21, 4):
        if landmarks[tip][1] < landmarks[tip-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def listen_for_commands():
    global is_voice_control_active, is_gesture_control_active, last_speech_time
    
    speak_bengali("হ্যালো! আমি আপনার সুপার স্মার্ট AI অ্যাসিস্ট্যান্ট। আমি আপনাকে যেকোনো বিষয়ে সাহায্য করতে পারি।")
    
    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                print("শুনছি...")
                audio = recognizer.listen(source)
                
                try:
                    command = recognizer.recognize_google(audio, language='bn-BD').lower()
                    print(f"শোনা গেছে: {command}")
                    
                    # System controls
                    if "মাউস" in command:
                        if "চালু" in command:
                            is_gesture_control_active = True
                            speak_bengali("হ্যান্ড জেসচার কন্ট্রোল চালু করা হয়েছে")
                        elif "বন্ধ" in command:
                            is_gesture_control_active = False
                            speak_bengali("হ্যান্ড জেসচার কন্ট্রোল বন্ধ করা হয়েছে")
                    
                    elif "ক্লিক" in command:
                        pyautogui.click()
                    
                    elif "ডাবল ক্লিক" in command:
                        pyautogui.doubleClick()
                    
                    elif "রাইট ক্লিক" in command:
                        pyautogui.rightClick()
                    
                    elif "স্ক্রল" in command:
                        if "উপরে" in command:
                            pyautogui.scroll(100)
                        elif "নিচে" in command:
                            pyautogui.scroll(-100)
                    
                    elif "বন্ধ" in command and "প্রোগ্রাম" in command:
                        speak_bengali("প্রোগ্রাম বন্ধ করা হচ্ছে")
                        break
                    
                    # AI processing
                    else:
                        response = ai_assistant.process_command(command)
                        speak_bengali(response)
                    
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
            elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                pyautogui.click()
                time.sleep(0.3)
            elif total_fingers == 3:
                pyautogui.rightClick()
                time.sleep(0.3)
            
            prev_x, prev_y = curr_x, curr_y
    
    status = "AI অ্যাসিস্ট্যান্ট: সক্রিয়" if is_gesture_control_active else "AI অ্যাসিস্ট্যান্ট: নিষ্ক্রিয়"
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Super AI Assistant", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak_bengali("AI অ্যাসিস্ট্যান্ট বন্ধ হচ্ছে। আবার দেখা হবে!")
        break

cap.release()
cv2.destroyAllWindows() 