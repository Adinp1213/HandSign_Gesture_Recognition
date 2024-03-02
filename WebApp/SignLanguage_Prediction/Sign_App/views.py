from django.shortcuts import render,redirect
from .models import*
from django.contrib import messages
from django.contrib.sessions.models import Session
from django.db import connection

import cv2
import mediapipe as mp
import numpy as np
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
# Create your views here.



# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

def Home(request):
	return render(request,"Home.html",{})

def Detect_Alphabet(request):
	return render(request,"Detect_Alphabet.html",{})

def Detect(request):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
     
    model = load_model('mp_hand_gesture')

    cap = cv2.VideoCapture(0)
    sentence = ''
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand landmark prediction
        result = hands.process(framergb)

        className = ''

        # Post-process the result
        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                landmarks = []
                for lm in handslms.landmark:
                    lmx = int(lm.x * frame.shape[1])
                    lmy = int(lm.y * frame.shape[0])
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mp_drawing.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)

                # Predict gesture
                # Assuming model is already loaded and classNames is defined
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

                # Append prediction to sentence if prediction time is less than 5 seconds or prediction changes
                if start_time is None:
                    start_time = time.time()
                else:
                    if time.time() - start_time > 5:
                        sentence += ' ' + className
                        start_time = None

        # Show the prediction on the frame
        cv2.putText(frame, className, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Create a blank image to display the sentence
        sentence_image = np.zeros((400, 800, 3), dtype=np.uint8)
        # Split the sentence into lines if it's too long
        words = sentence.split(' ')
        lines = ['']
        line_width = 0
        line_idx = 0
        for word in words:
            word_width = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
            if line_width + word_width > 800 or len(lines[line_idx].split()) >= 8:
                lines.append('')
                line_width = 0
                line_idx += 1
            lines[line_idx] += word + ' '
            line_width += word_width

        # Draw each line of the sentence
        y = 50
        for line in lines:
            cv2.putText(sentence_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            y += 50

        # Show the final output
        cv2.imshow("Output", frame)

        # Show the sentence in a separate window
        cv2.imshow("Sentence", sentence_image)

        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

    return redirect('/Detect_Alphabet')


