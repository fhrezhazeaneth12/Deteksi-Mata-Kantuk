import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import dlib
from PIL import Image

# Inisialisasi GUI
root = tk.Tk()
root.title("Drowsiness Detection")

# Inisialisasi detektor wajah menggunakan Viola-Jones
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi detektor mata menggunakan Haar Cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inisialisasi detektor landmark mata
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

# Fungsi untuk melakukan preprocessing citra mata
def preprocess_eye_image(eye_image):
    if eye_image is None:
        return None

    # Resize citra mata menjadi 128x128 piksel
    eye_image_pil = Image.fromarray(eye_image)
    eye_image_resized = eye_image_pil.resize((128, 128))
    eye_image_resized = np.array(eye_image_resized)

    return eye_image_resized

# Fungsi untuk melakukan ekstraksi fitur dan analisis
def analyze_eye_state(eye_image):
    try:
        # Preprocessing citra mata
        preprocessed_eye_image = preprocess_eye_image(eye_image)

        if preprocessed_eye_image is not None:
            # Melakukan PCA
            eye_feature_vector_pca = pca.transform(preprocessed_eye_image.reshape(1, -1))

            # Melakukan LDA
            eye_feature_vector_lda = lda.transform(eye_feature_vector_pca)

            # Prediksi status kantuk mata dengan model SVM
            prediction = svm.predict(eye_feature_vector_lda)

            # Menggunakan label encoder untuk mendapatkan label hasil prediksi
            predicted_label = label_encoder.inverse_transform(prediction)

            return predicted_label[0]

    except Exception as e:
        print(f"Error: {e}")
        return "Error"

# Fungsi untuk memperbarui label status
def update_status_label():
    global left_eye_status, right_eye_status

    # Tentukan status berdasarkan kondisi mata
    if left_eye_status == "Open" and right_eye_status == "Open":
        status_label.config(text="Status: Awake")
    elif left_eye_status == "Closed" and right_eye_status == "Closed":
        status_label.config(text="Status: Drowsy")
    #elif left_eye_status == "Closed" and right_eye_status == "Open" or left_eye_status == "Open" and right_eye_status == "Closed" :
    #    status_label.config(text="Status: Unknown")
# Buat fungsi untuk keluar dari aplikasi
def exit_app():
    root.quit()

# Fungsi untuk mendeteksi wajah dan mata secara real-time
def detect_face_and_eyes():
    global left_eye_status, right_eye_status

    # Baca frame dari webcam
    ret, frame = cap.read()

    if ret:
        # Konversi frame ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Gambar kotak batas wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop area wajah
            face_roi = gray[y:y + h, x:x + w]

            # Simpan citra hasil deteksi wajah ke dalam file
            cv2.imwrite("detected_face.jpg", frame)
    
            # Deteksi mata dalam wajah
            eyes = eye_cascade.detectMultiScale(face_roi)

            for i, (ex, ey, ew, eh) in enumerate(eyes):
                # Gambar kotak batas mata
                eye_image = face_roi[ey:ey + eh, ex:ex + ew]
                
                # Tentukan posisi mata (kiri atau kanan)
                eye_position = "left" if i == 0 else "right"

                # Simpan citra hasil deteksi mata ke dalam file
                cv2.imwrite(f"detected_eye_{0}.jpg", eye_image)
    
                # Prediksi status mata
                eye_status = analyze_eye_state(eye_image)

                if eye_position == "left":
                    left_eye_status = eye_status
                elif eye_position == "right":
                    right_eye_status = eye_status

                    # Perbarui label status berdasarkan mata terbuka atau tertutup
                if left_eye_status == "Open" and right_eye_status == "Open":
                    status_label.config(text="Status: Awake")
                else:
                    status_label.config(text="Status: Drowsy")

                #elif left_eye_status == "Closed" and right_eye_status == "Closed":
                #    status_label.config(text="Status: Drowsy")
                #elif left_eye_status == "Closed" and right_eye_status == "Open" or left_eye_status == "Open" and right_eye_status == "Closed" :
                #    status_label.config(text="Status: Unknown")
        # Perbarui label status
        update_status_label()

        # Tampilkan frame dengan hasil deteksi mata pada GUI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        label.config(image=photo)
        label.image = photo

    # Loop deteksi setiap 10 ms
    label.after(10, detect_face_and_eyes)
        
# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Buat label untuk menampilkan status mata
status_label = ttk.Label(root, text="Status: ")
status_label.pack()

# Tambahkan variabel untuk melacak status mata kiri dan mata kanan
left_eye_status = "Unknown"
right_eye_status = "Unknown"

# Muat model SVM, PCA, dan LDA dari file
model_obj = joblib.load(r'D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\PROSES\DETEKSI KANTUK\KANTUK\trained_model_obj.pkl')
svm = model_obj['svm']
pca = model_obj['pca']
lda = model_obj['lda']
label_encoder = model_obj['label_encoder']

# Buat label untuk menampilkan gambar
label = ttk.Label(root)
label.pack()

# Mulai deteksi mata secara real-time
detect_face_and_eyes()

# Main loop GUI
root.mainloop()

# Hentikan webcam
cap.release()