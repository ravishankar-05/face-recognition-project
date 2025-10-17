import os
import cv2
import numpy as np
from collections import Counter
import argparse

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def knn(X_train, y_train, x, k=5):
    distances = [np.linalg.norm(x - x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)[0][0]
    return most_common

def prepare_data(data_dir):
    X, y, names = [], [], {}
    class_id = 0

    for person in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person)
        if not os.path.isdir(person_path): continue
        names[class_id] = person

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x_, y_, w, h) in faces:
                face_roi = gray[y_:y_+h, x_:x_+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                X.append(face_roi.flatten())
                y.append(class_id)
                break
        class_id += 1

    X = np.array(X)
    y = np.array(y)
    np.save("face_data.npy", X)
    np.save("face_labels.npy", y)
    np.save("face_names.npy", names)
    print("✅ Dataset created and saved!")
    return X, y, names

def test_image(img_path):
    X_train = np.load("face_data.npy")
    y_train = np.load("face_labels.npy")
    names = np.load("face_names.npy", allow_pickle=True).item()

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x_, y_, w, h) in faces:
        face_roi = gray[y_:y_+h, x_:x_+w]
        face_roi = cv2.resize(face_roi, (100, 100)).flatten()
        pred = knn(X_train, y_train, face_roi)
        name = names[pred]
        cv2.putText(img, name, (x_, y_-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(img, (x_, y_), (x_+w, y_+h), (255,0,0), 2)

    cv2.imshow("Test Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def live_recognition():
    X_train = np.load("face_data.npy")
    y_train = np.load("face_labels.npy")
    names = np.load("face_names.npy", allow_pickle=True).item()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to access webcam.")
        return

    print("🎥 Starting webcam... Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x_, y_, w, h) in faces:
            face_roi = gray[y_:y_+h, x_:x_+w]
            face_roi = cv2.resize(face_roi, (100, 100)).flatten()
            pred = knn(X_train, y_train, face_roi)
            name = names[pred]
            cv2.putText(frame, name, (x_, y_-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x_, y_), (x_+w, y_+h), (255,0,0), 2)

        cv2.imshow("Live Recognition", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Path to faces_data folder")
    parser.add_argument('--test', type=str, help="Path to test image")
    parser.add_argument('--webcam', action='store_true', help="Run live webcam recognition")
    args = parser.parse_args()

    if args.data:
        prepare_data(args.data)

    if args.test:
        test_image(args.test)

    if args.webcam:
        live_recognition()
