import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import pyttsx3
import os
import win32gui
import threading
from queue import Queue

print("AI Assistant শুরু হচ্ছে...")

# Set up PyAutoGUI settings
pyautogui.FAILSAFE = False
pyautogui.MINIMUM_DURATION = 0.1
pyautogui.PAUSE = 0.1

# Configure drawing specs for better visualization
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)
draw_spec_dots = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)

# Initialize hand tracking with optimized configuration
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # Increased for more accurate detection
    min_tracking_confidence=0.7,    # Increased for more stable tracking
    model_complexity=1  # Use more complex model for better accuracy
)
mp_draw = mp.solutions.drawing_utils
# Update drawing specifications for red dots
draw_spec = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
draw_spec_dots = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Speech queue for non-blocking narration
speech_queue = Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text == "STOP":
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"স্পিচ এরর: {str(e)}")
        speech_queue.task_done()

def speak_action(text):
    speech_queue.put(text)

# Start speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Basic operation descriptions
operation_descriptions = {
    "folder_open": "Opening folder",
    "file_select": "Selecting file",
    "menu_open": "Opening menu",
    "drag_start": "Started dragging",
    "drag_end": "Dropped item",
    "text_select": "Selecting text"
}

# Simple position smoothing
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.points = []
    
    def update(self, point):
        self.points.append(point)
        if len(self.points) > self.window_size:
            self.points.pop(0)
        return np.mean(self.points, axis=0)

def count_fingers(hand_landmarks):
    # Finger indices
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_TIP = 8
    INDEX_PIP = 6
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    
    def is_finger_raised(tip_id, pip_id, mcp_id):
        tip_y = hand_landmarks[tip_id].y
        pip_y = hand_landmarks[pip_id].y
        return tip_y < pip_y
    
    def is_thumb_up(tip_id, ip_id, mcp_id):
        tip_y = hand_landmarks[tip_id].y
        ip_y = hand_landmarks[ip_id].y
        mcp_y = hand_landmarks[mcp_id].y
        return tip_y < ip_y and ip_y < mcp_y
    
    # Check thumb and other fingers
    thumb_up = is_thumb_up(THUMB_TIP, THUMB_IP, THUMB_MCP)
    index_raised = is_finger_raised(INDEX_TIP, INDEX_PIP, INDEX_MCP)
    middle_raised = is_finger_raised(MIDDLE_TIP, MIDDLE_PIP, INDEX_MCP)
    ring_raised = is_finger_raised(RING_TIP, RING_PIP, INDEX_MCP)
    pinky_raised = is_finger_raised(PINKY_TIP, PINKY_PIP, INDEX_MCP)
    
    # Return finger count based on combinations
    raised_fingers = [index_raised, middle_raised, ring_raised, pinky_raised]
    finger_count = sum(raised_fingers)
    
    if thumb_up and finger_count == 0:
        return 6  # Thumb up for voice typing
    elif finger_count == 1 and index_raised:
        return 1  # Mouse movement
    elif finger_count == 2 and index_raised and middle_raised:
        return 2  # Left click
    elif finger_count == 3:
        return 3  # Right click
    elif finger_count == 4:
        return 4  # Double click
    elif finger_count >= 4:
        return 5  # Drag and drop
    return 0

def main():
    # Initialize webcam with optimized settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    cap.set(cv2.CAP_PROP_CONTRAST, 128)
    
    # Initialize thumb detection timer
    thumb_up_start_time = 0
    thumb_detection_threshold = 3.0  # 3 seconds threshold
    
    if not cap.isOpened():
        print("Camera not found or cannot be opened")
        return

    # Wait for camera to initialize
    time.sleep(2)
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab initial frame from camera")
        return
    
    print("হ্যান্ড জেসচার কন্ট্রোল প্রস্তুত!")
    print("তর্জনী আঙ্গুল: মাউস মুভ")
    print("তর্জনী + মধ্যমা: লেফট ক্লিক")
    print("তর্জনী + মধ্যমা + অনামিকা: রাইট ক্লিক")
    print("চারটি আঙ্গুল: ডাবল ক্লিক")
    print("পাঁচটি আঙ্গুল: ড্র্যাগ অ্যান্ড ড্রপ")
    print("বন্ধ করতে 'q' চাপুন")
    
    speak_action("AI Assistant is ready")
    
    last_click_time = 0
    click_cooldown = 0.5
    last_fingers = 0
    click_hold_time = 0
    position_filter = MovingAverageFilter(window_size=5)
    is_dragging = False
    last_action_time = 0
    action_cooldown = 1.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Frame buffering and error handling
            try:
                # Enhance frame quality with optimized parameters
                frame = cv2.flip(frame, 1)
                frame = cv2.GaussianBlur(frame, (5,5), 0)  # Reduce noise
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Enhance contrast and brightness
                
                # Enhanced noise reduction pipeline
                frame = cv2.GaussianBlur(frame, (7, 7), 0)  # Increased blur kernel
                frame = cv2.medianBlur(frame, 5)  # Additional median filter for noise reduction
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with timeout protection
                start_time = time.time()
                results = hands.process(rgb_frame)
                
                if time.time() - start_time > 0.5:  # Timeout after 500ms
                    print("Hand detection timeout, skipping frame")
                    continue
                    
                if results is None:
                    print("No results from hand detection")
                    continue
                    
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                time.sleep(0.1)  # Add small delay before retrying
                continue
            
            current_time = time.time()
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw hand landmarks with improved visualization
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    draw_spec_dots,  # Use red dots for landmarks
                    draw_spec  # Use default style for connections
                )
                
                fingers = count_fingers(hand_landmarks.landmark)
                
                if fingers != last_fingers:
                    click_hold_time = current_time
                    if fingers != 5 and is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                
                hold_duration = current_time - click_hold_time
                
                if fingers == 1:
                    # Mouse movement
                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * frame.shape[1])
                    y = int(index_tip.y * frame.shape[0])
                    
                    smoothed_pos = position_filter.update(np.array([x, y]))
                    screen_x = int(np.interp(smoothed_pos[0], [100, frame.shape[1]-100], [0, 1920]))
                    screen_y = int(np.interp(smoothed_pos[1], [100, frame.shape[0]-100], [0, 1080]))
                    
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
                    
                elif fingers == 2 and hold_duration > 0.3 and current_time - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = current_time
                    
                elif fingers == 3 and hold_duration > 0.3 and current_time - last_click_time > click_cooldown:
                        pyautogui.click(button='right')
                        last_click_time = current_time
                    
                elif fingers == 4 and hold_duration > 0.3 and current_time - last_click_time > click_cooldown:
                    pyautogui.doubleClick()
                    last_click_time = current_time
                    
                elif fingers == 5:
                    if not is_dragging and hold_duration > 0.3:
                        pyautogui.mouseDown()
                        is_dragging = True
                    
                    if is_dragging:
                        index_tip = hand_landmarks.landmark[8]
                        x = int(index_tip.x * frame.shape[1])
                        y = int(index_tip.y * frame.shape[0])
                        
                        smoothed_pos = position_filter.update(np.array([x, y]))
                        screen_x = int(np.interp(smoothed_pos[0], [50, frame.shape[1]-50], [0, 1920]))
                        screen_y = int(np.interp(smoothed_pos[1], [50, frame.shape[0]-50], [0, 1080]))
                        
                        pyautogui.moveTo(screen_x, screen_y, duration=0.1)
                
                elif fingers == 6:
                    if thumb_up_start_time == 0:  # Start timing when thumb is first detected
                        thumb_up_start_time = current_time
                    elif current_time - thumb_up_start_time >= thumb_detection_threshold:
                        if current_time - last_action_time > action_cooldown:
                            # Trigger Windows+H for voice typing
                            pyautogui.hotkey('win', 'h')
                            speak_action("Voice typing activated")
                            last_action_time = current_time
                            thumb_up_start_time = 0  # Reset timer
                elif thumb_up_start_time > 0:  # Reset timer if thumb is lowered
                    thumb_up_start_time = 0
                
                last_fingers = fingers
            
            cv2.imshow("AI Assistant", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        speech_queue.put("STOP")
        speech_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()