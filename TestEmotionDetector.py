import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from datetime import datetime
from collections import Counter
import pandas as pd  # For rolling average
import matplotlib.dates as mdates

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Map emotions to numerical values for the graph
emotion_to_value = {
    "Angry": -3,
    "Disgusted": -2,
    "Fearful": -1,
    "Neutral": 0,
    "Happy": 1,
    "Sad": -1,
    "Surprised": 2,
}


# Load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Initialize video capture
cap = cv2.VideoCapture("selfVideo1.mp4")

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
    # Extract times and emotions
    times = [timestamp for _, timestamp in emotion_log]
    emotions = [emotion for emotion, _ in emotion_log]

    # Convert emotions to numerical values for smoothing
    emotion_values = [emotion_to_value[emotion] for emotion in emotions]

    # Create a DataFrame for easier rolling average calculation
    data = pd.DataFrame({"Time": times, "Emotion": emotion_values})

    # Compute rolling averages with different window sizes
    data["Smoothed_Emotion_25"] = data["Emotion"].rolling(window=25, min_periods=1).mean()  # 25-second window
    data["Smoothed_Emotion_5"] = data["Emotion"].rolling(window=5, min_periods=1).mean()  # 5-second window

    # Convert datetime objects to numerical format for plotting
    data["Time_Num"] = mdates.date2num(data["Time"])

    # Create a line plot
    plt.figure(figsize=(12, 6))

    # Plot the 25-second smoothed emotion line (blue) with linewidth 3
    plt.plot(data["Time_Num"], data["Smoothed_Emotion_25"], linestyle='-', color='b', label="Smoothed Emotion (25s)", alpha=0.9, linewidth=3)

    # Plot the 5-second smoothed emotion line (red) with linewidth 1
    plt.plot(data["Time_Num"], data["Smoothed_Emotion_5"], linestyle='-', color='r', label="Smoothed Emotion (5s)", alpha=0.5, linewidth=1)

    # Format the x-axis for readable datetime labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S.%f"))  # Keep full milliseconds
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    # Adjust x-axis labels to avoid overlap
    plt.gcf().autofmt_xdate()

    # Add horizontal lines for each emotion level
    for emotion, value in emotion_to_value.items():
        plt.axhline(y=value, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.text(data["Time_Num"].iloc[-1], value, emotion, va='center', ha='left', color='black', fontsize=10)

    # Add labels, title, and grid
    plt.xlabel("Time (hh:mm:ss.ms)")
    plt.ylabel("Emotion Levels (Smoothed)")
    plt.title("Emotion Detection Over Time (Smoothed Lines)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Provide recommendations based on the most common emotion
    emotion_counts = Counter(emotions)
    most_common_emotion, most_common_count = emotion_counts.most_common(1)[0]

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