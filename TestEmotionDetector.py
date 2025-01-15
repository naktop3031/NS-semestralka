import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from datetime import timedelta
from collections import Counter

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Initialize video capture
cap = cv2.VideoCapture("videoBean.mp4")

# Collect emotions and their timestamps
emotion_log = []
frame_rate = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current time in seconds
    frame = cv2.resize(frame, (1280, 720))

    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces available in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]

        # Log the emotion and timestamp
        emotion_log.append((emotion, current_time))

        # Display the emotion on the video
        cv2.putText(frame, emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Process the collected data to create a graph
if emotion_log:
    times = [timestamp for _, timestamp in emotion_log]
    emotions = [emotion for emotion, _ in emotion_log]

    # Count the occurrences of each emotion
    emotion_counts = Counter(emotions)
    most_common_emotion, most_common_count = emotion_counts.most_common(1)[0]

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot each unique emotion
    unique_emotions = list(set(emotions))
    for emotion in unique_emotions:
        emotion_times = [times[i] for i in range(len(times)) if emotions[i] == emotion]
        plt.scatter(emotion_times, [emotion] * len(emotion_times), label=emotion)

    # Add annotations
    plt.xlabel("Time (s)")
    plt.ylabel("Emotions")
    plt.title(f"Emotion Detection Over Time\nMost Common Emotion: {most_common_emotion} ({most_common_count} times)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Provide recommendations based on the most common emotion
    recommendations = {
        "Happy": "Great work! Keep up the positive environment.",
        "Sad": "Consider promoting better work-life balance and addressing employee concerns.",
        "Angry": "Look into potential conflicts or stressors in the workplace.",
        "Disgusted": "Review workplace hygiene or interpersonal dynamics.",
        "Fearful": "Ensure employees feel safe and secure at work.",
        "Neutral": "Maintain stability but look for ways to engage employees further.",
        "Surprised": "Encourage an open and adaptive culture to manage surprises effectively."
    }

    suggestion = recommendations.get(most_common_emotion, "No specific recommendation available.")
    print(f"Most Common Emotion: {most_common_emotion}")
    print(f"Recommendation: {suggestion}")
