import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

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

# Variables for smooth control
frame_reduction = 100
smoothening = 7
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

def count_fingers(landmarks):
    fingers = []
    # Check fingers (index to pinky)
    for tip in range(8, 21, 4):
        if landmarks[tip][1] < landmarks[tip-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def main():
    try:
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)  # Mirror image
            
            # Convert to RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarks.append([cx, cy])
                    
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get finger count
                    fingers = count_fingers(landmarks)
                    total_fingers = sum(fingers)
                    
                    # Index finger tip coordinates
                    index_x = landmarks[8][0]
                    index_y = landmarks[8][1]
                    
                    # Convert coordinates
                    screen_x = int(np.interp(index_x, (frame_reduction, 640 - frame_reduction), (0, screen_width)))
                    screen_y = int(np.interp(index_y, (frame_reduction, 480 - frame_reduction), (0, screen_height)))
                    
                    # Smooth values
                    curr_x = prev_x + (screen_x - prev_x) / smoothening
                    curr_y = prev_y + (screen_y - prev_y) / smoothening
                    
                    # Move mouse based on gestures
                    if total_fingers == 1 and fingers[1] == 1:  # Only index finger up
                        pyautogui.moveTo(curr_x, curr_y)
                    elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:  # Index and middle finger up
                        pyautogui.click()
                        time.sleep(0.3)
                    elif total_fingers == 3:  # Three fingers up
                        pyautogui.rightClick()
                        time.sleep(0.3)
                    elif total_fingers == 5:  # All fingers up - scroll mode
                        # Calculate vertical movement since last frame
                        scroll_delta = curr_y - prev_y
                        
                        # Apply non-linear scaling for more intuitive control
                        scroll_sensitivity = 0.3  # Reduced sensitivity for finer control
                        scroll_amount = int(scroll_delta * scroll_sensitivity)
                        
                        # Add threshold to prevent accidental scrolling
                        if abs(scroll_amount) > 2:
                            # Apply smooth scrolling with direction
                            pyautogui.scroll(scroll_amount)
                            time.sleep(0.02)  # Small delay for smoother scrolling
                    
                    prev_x, prev_y = curr_x, curr_y
            
            # Display
            cv2.imshow("Hand Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
