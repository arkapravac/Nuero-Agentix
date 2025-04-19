# AI-Powered Gesture-Based Windows Management Interface

This project implements an advanced computer control system that combines hand gesture recognition and voice commands to provide an intuitive and hands-free way to interact with your computer. The system supports both English and Bengali voice commands, making it accessible to Bengali speakers.

## Features

### Hand Gesture Control
- Mouse cursor control using index finger movements
- Left click with index and middle fingers
- Right click with three fingers
- Double click with four fingers
- Smooth scrolling with five fingers
- Enhanced gesture stability with frame buffering and movement smoothing

### Voice Command Support
- Bengali voice commands for system control
- Voice feedback in Bengali
- Commands for:
  - Enabling/disabling gesture control
  - Mouse clicks
  - Scrolling
  - System control

### Advanced Features
- Real-time hand tracking using MediaPipe
- Smooth cursor movement with position filtering
- Frame stabilization for better tracking
- Non-blocking voice feedback system
- Configurable sensitivity and control parameters

## Requirements

```
opencv-python==4.11.0.86
mediapipe==0.10.21
pyautogui==0.9.54
numpy==1.26.2
SpeechRecognition==3.10.0
pyttsx3==2.90
keyboard==0.13.5
googletrans==3.1.0a0
gtts==2.3.2
playsound==1.2.2
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main program:
   ```
   python gesture_voice_control.py
   ```

2. Hand Gesture Controls:
   - Move cursor: Raise index finger
   - Left click: Raise index and middle fingers
   - Right click: Raise three fingers
   - Double click: Raise four fingers
   - Scroll: Raise all fingers and move hand up/down

3. Voice Commands (Bengali):
   - "মাউস চালু" - Enable gesture control
   - "মাউস বন্ধ" - Disable gesture control
   - "ক্লিক" - Left click
   - "রাইট ক্লিক" - Right click
   - "ডাবল ক্লিক" - Double click
   - "স্ক্রল উপরে" - Scroll up
   - "স্ক্রল নিচে" - Scroll down
   - "প্রোগ্রাম বন্ধ" - Exit program

## Contributing

Contributions are welcome! Please feel free to submit pull requests with improvements to:
- Gesture recognition accuracy
- Voice command processing
- Performance optimizations
- Additional features and commands

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand tracking capabilities
- Google Speech Recognition for voice command processing
- PyAutoGUI for computer control functionality