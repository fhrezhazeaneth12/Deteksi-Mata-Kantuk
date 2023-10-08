#akurasi realtime
#tidak teridentifikasi jk salah satu trtutp dan akurasi 0

import os #Akses direktori : mengelola file,dll 
import cv2 # pemrosesan citra dan video
import dlib # analisis wajah dan pengenalan objek
import numpy as np #komputasi numerik, terutama konteks array dan operasi matematika.
import tkinter as tk #membuat antarmuka
import matplotlib.pyplot as plt #Membuat visualisasi dan plot data.
import seaborn as sns #membuat visualisasi statistik yang lebih menarik dan informatif.
from PIL import Image, ImageTk #manipulasi gambar dan menampilkan gambar dalam GUI.
from imutils import face_utils #Analisis wajah dan pengenalan objek
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #mengkodekan label kelas dalam bentuk angka.
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report #menganalisis hasil klasifikasi.
from sklearn.metrics import confusion_matrix, roc_curve, auc #untuk evaluasi kinerja model klasifikasi.
from PIL import Image

# Ambang batas EAR (Eye Aspect Ratio) yang menunjukkan mata terpejam (jika dibawah 0.25)
EAR_THRESHOLD = 0.25

# Inisialisasi detektor wajah Viola-Jones
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi detektor landmark wajah
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

# Membaca dataset citra mata terbuka dan mata terpejam
def read_dataset():
    images = []
    labels = []

    # Path dataset citra mata terbuka
    open_eyes_path = r"D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\PROSES\DETEKSI KANTUK\Data\Data Kaggle\train\Open_Eyes\\"
    open_eye_files = [f for f in os.listdir(open_eyes_path) if os.path.isfile(os.path.join(open_eyes_path, f))]

    # Membaca citra mata terbuka
    for file in open_eye_files:
        image = cv2.imread(open_eyes_path + file, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)
            labels.append("Open")

    # Path dataset citra mata terpejam
    closed_eyes_path = r"D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\PROSES\DETEKSI KANTUK\Data\Data Kaggle\train\Closed_Eyes\\"
    closed_eye_files = [f for f in os.listdir(closed_eyes_path) if os.path.isfile(os.path.join(closed_eyes_path, f))]

    # Membaca citra mata terpejam
    for file in closed_eye_files:
        image = cv2.imread(closed_eyes_path + file, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)
            labels.append("Closed")

    return images, labels

# Melakukan preprocessing citra
def preprocess_image(image):
    if image is None:
        return None

    #memeriksa apakah citra yang diberikan memiliki bentuk (shape) yang valid atau tidak.
    if len(image.shape) < 2 or image.shape[0] == 0 or image.shape[1] == 0:
        return None
    #dimensi citra lebih dari atau sama dengan 2. if < 2, citra tidak memiliki salinan atau bentuk yang valid.
    #image.shape[0] = baris, image.shape[1] = kolom
    
    
    # mengonversi citra array NumPy -> objek dalam format citra yang dikenali oleh modul PIL
    image_pil = Image.fromarray(image)

    # Resize the image
    image_resized = image_pil.resize((128, 128))

    # Convert the resized image back to numpy array : mengonversi kembali objek citra PIL yang telah diubah ukurannya menjadi bentuk array NumPy
    image_resized = np.array(image_resized)

    return image_resized

# Menghitung Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    # Menghitung jarak antara titik pada kelopak mata
    A = np.linalg.norm(eye[1] - eye[5]) #titik 1 dan 5 : mewakili ujung horizontal kelopak mata
    B = np.linalg.norm(eye[2] - eye[4]) #titik 2 dan 4 : mewakili ujung diagonal kelopak mata
    #np.linalg.norm  : menghitung panjang vektor atau jarak antara dua titik dalam ruang tiga dimensi (x, y, z)
    
    # Menghitung jarak antara titik pada kelopak mata secara vertikal
    C = np.linalg.norm(eye[0] - eye[3]) #titik 0(atas kelopak) dan 3(bawah) :mewakili ujung vertikal kelopak mata

    # Menghitung rasio aspek mata (EAR)
    ear = (A + B) / (2.0 * C)

    return ear


# Deteksi mata terpejam berdasarkan EAR
def is_eye_closed(eye):
    # Menghitung EAR
    ear = calculate_ear(eye)

    # Cetak nilai EAR ke konsol
    #print("EAR:", ear)
    
    # Mengembalikan True jika EAR di bawah ambang batas
    if ear < EAR_THRESHOLD:
        return True
    else:
        return False

# Melakukan deteksi mata kantuk pada gambar
def detect_drowsiness():
    global total_frames, correct_closed_eye_frames, correct_open_eye_frames
    
    # Mengambil frame dari video
    ret, frame = video_capture.read()
    if ret:
        #mengubah citra berwarna (BGR) dari frame menjadi citra grayscale. 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah menggunakan Viola-Jones
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        #NGUJI model
        #faces = face_cascade.detectMultiScale(gray, scaleFactor=2.2, minNeighbors=10, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Menentukan ROI (Region of Interest) pada wajah
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # bounding box (kotak persegi panjang)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Warna hijau, lebar garis 2

            # Deteksi landmark wajah menggunakan dlib
            shape = predictor(roi_gray, dlib.rectangle(0, 0, w, h))
            shape = face_utils.shape_to_np(shape)

            # Mendapatkan koordinat titik landmark untuk mata kiri dan mata kanan
            left_eye = shape[36:42]
            right_eye = shape[42:48]

            # Menggambar landmark pada gambar
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Deteksi mata terpejam berdasarkan EAR
            left_eye_closed = is_eye_closed(left_eye)
            right_eye_closed = is_eye_closed(right_eye)

            # status mata
            if left_eye_closed and right_eye_closed:
                status = "Drowsy"
            elif not left_eye_closed and not right_eye_closed:
                status = "Awake"
            else:
                status = "Tidak teridentifikasi"
            gui.status_label.config(text="Status: " + status)

            # Mengubah citra mata menjadi vektor fitur menggunakan PCA
            left_eye_features = preprocess_image(left_eye)
            left_eye_features = left_eye_features.flatten().reshape(1, -1)
            left_eye_features = pca.transform(left_eye_features)

            right_eye_features = preprocess_image(right_eye)
            right_eye_features = right_eye_features.flatten().reshape(1, -1)
            right_eye_features = pca.transform(right_eye_features)

            # Mengubah vektor fitur menggunakan LDA
            left_eye_features = lda.transform(left_eye_features)
            right_eye_features = lda.transform(right_eye_features)

            # Memprediksi kelas menggunakan SVM
            left_eye_prediction = svm.predict(left_eye_features)[0]
            right_eye_prediction = svm.predict(right_eye_features)[0]

            # Menghitung akurasi secara real-time
            if left_eye_closed and right_eye_closed and status == "Drowsy":
                correct_closed_eye_frames += 1
            elif not left_eye_closed and not right_eye_closed and status == "Awake":
                correct_open_eye_frames += 1

            total_frames += 1

            # Menghitung akurasi real-time
            if total_frames > 0:
                accuracy_realtime = (correct_closed_eye_frames + correct_open_eye_frames) / total_frames
                gui.accuracy_label.config(text="Real-time Accuracy: {:.2f}%".format(accuracy_realtime * 100))
                #gui.total_frames_label.config(text="Total Frames: {}".format(total_frames))
                #gui.correct_frames_label.config(text="Correct Frames: {}".format(correct_closed_eye_frames + correct_open_eye_frames))

            # Menghitung akurasi
            if left_eye_closed and right_eye_closed and status == "Drowsy":
                correct_closed_eye_frames += 1
            elif not left_eye_closed and not right_eye_closed and status == "Awake":
                correct_open_eye_frames += 1
            elif status == "Tidak teridentifikasi":
                accuracy_realtime = 0
            else:
                accuracy_realtime = (correct_closed_eye_frames + correct_open_eye_frames) / total_frames
            gui.accuracy_label.config(text="Real-time Accuracy: {:.2f}%".format(accuracy_realtime * 100))
            total_frames += 1
            
        # Mengonversi frame menjadi objek gambar Tkinter
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=image)

        # Menampilkan objek gambar Tkinter pada GUI
        gui.video_label.imgtk = imgtk
        gui.video_label.configure(image=imgtk)

    # Memanggil fungsi deteksi mata kantuk secara periodik
    gui.after(10, detect_drowsiness)

# Membaca dataset dan melakukan preprocessing
images, labels = read_dataset()
images_processed = [preprocess_image(image) for image in images]
images_processed = np.array(images_processed)
images_processed = images_processed.reshape(images_processed.shape[0], -1)

# Mengubah label menjadi angka menggunakan LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

#print("Image_processed : \n", images_processed[0])  # Menampilkan contoh citra yang telah diproses
#print("Labels_encoded : \n",labels_encoded[0])    # Menampilkan contoh label yang telah dienkripsi

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(images_processed, labels_encoded, test_size=0.2, random_state=42)

# Melakukan PCA, n_compnents : berapa banyak komponen utama yg akan diambil dari data asli(mencari keseimbangan antara pengurangan dimensi dan mempertahankan informasi)
pca = PCA(n_components=93)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Cetak nilai PCA
#print("X_Train_PCA : \n", X_train_pca)
#print("X_test_PCA : \n", X_test_pca)

# Melakukan LDA, n_comp tidak bisa melebihi jumlah kelas-1. (2-1=1) : dimensi baru yang dihasilkan memiliki hubungan dengan jumlah kelas.
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train_pca, y_train)
X_test_lda = lda.transform(X_test_pca)

# Cetak nilai PCA
#print("X_Train_LDA : \n", X_train_lda)
#print("X_test_LDA : \n", X_test_lda)

# Melatih model SVM, y_train 0 = mata tertutup
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_lda, y_train)

#Cetak SVM
#print("SVM Model : \n")
#print("X_train_lda : \n", X_train_lda)
#print("y_train : \n", y_train)

# Visualisasi data pelatihan setelah PCA
#plt.figure(figsize=(8, 6))
#plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
#plt.title('Scatter Plot Data Pelatihan setelah PCA')
#plt.xlabel('Komponen Utama 1')
#plt.ylabel('Komponen Utama 2')

# Visualisasi data pelatihan setelah LDA
#plt.figure(figsize=(10, 8))
#plt.scatter(X_train_lda[:, 0], y_train, c=y_train, cmap='viridis')
#plt.title('Scatter Plot Data Pelatihan setelah LDA')
#plt.xlabel('Komponen Utama setelah LDA')
#plt.ylabel('Kelas')

# Menghitung akurasi model
y_pred_train = svm.predict(X_train_lda)
y_pred_test = svm.predict(X_test_lda)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Membuat grafik hasil data training dan testing
#plt.figure(figsize=(8, 6))
#plt.plot(["Train", "Test"], [train_accuracy, test_accuracy], marker='o', linestyle='-', color='b')
#plt.title('Akurasi Pelatihan vs Akurasi Pengujian')
#plt.xlabel('Data')
#plt.ylabel('Akurasi')
#plt.ylim([0, 1])
#plt.grid(True)

# Menampilkan hasil akurasi yang detail
print("Train Accuracy:", accuracy_train)
print("Test Accuracy:", accuracy_test)

# Confusion matrix untuk data uji
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix (Test Data):\n", conf_matrix_test)

# Classification report untuk data uji
target_names = label_encoder.classes_
class_report_test = classification_report(y_test, y_pred_test, target_names=target_names)
print("Classification Report (Test Data):\n", class_report_test)

# Visualisasi Confusion Matrix
#plt.figure(figsize=(8, 6))
#sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
#plt.title("Confusion Matrix (Test Data)")
#plt.xlabel("Prediksi")
#plt.ylabel("Aktual")

# Menghitung ROC Curve
#y_score = svm.decision_function(X_test_lda) #skor keputusan (decision score) data uji, seberapa yakin model dalam mengklasifikasikan data sebagai kelas positif.
#fpr, tpr, _ = roc_curve(y_test, y_score)
#roc_auc = auc(fpr, tpr) #sejauh mana model mampu membedakan antara kelas positif dan negatif, semakin besar semakin baik

#menampilkan di console
#print("y_score: \n", y_score)
#print(" False Positive Rate (FPR) : \n", fpr)
#print("True Positive Rate (TPR) : \n", tpr)
#print("ROC AUC", roc_auc)

# Menampilkan ROC Curve
#plt.figure()
#plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic')
#plt.legend(loc='lower right')

# Membuat GUI
class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Drowsiness Detection")
        self.geometry("800x600")

        # Label untuk menampilkan video
        self.video_label = tk.Label(self)
        self.video_label.place(x=50, y=30, width=600, height=450)

        # Label untuk menampilkan status
        self.status_label = tk.Label(self, text="Status: ", font=("Helvetica", 16), bg="white", fg="black")
        self.status_label.place(x=50, y=490)
        
        #self.accuracy_label = tk.Label(self, text="Real-time Accuracy: 0.00%", font=("Helvetica", 16), bg="white", fg="black")
        #self.accuracy_label.place(x=50, y=520)
        
        #self.total_frames_label = tk.Label(self, text="Total Frames: 0", font=("Helvetica", 16), bg="white", fg="black")
        #self.total_frames_label.place(x=50, y=550)
        
        #self.correct_frames_label = tk.Label(self, text="Correct Frames: 0", font=("Helvetica", 16), bg="white", fg="black")
        #self.correct_frames_label.place(x=250, y=550)
        
        # Tombol Keluar
        self.exit_button = tk.Button(self, text="Keluar", command=self.quit, font=("Helvetica", 14), bg="red", fg="white")
        self.exit_button.place(x=700, y=550)
gui = GUI()

# Menampilkan plot
plt.show()

# Membuka kamera
video_capture = cv2.VideoCapture(0)

# Variabel untuk menghitung akurasi secara real-time
total_frames = 0
correct_closed_eye_frames = 0
correct_open_eye_frames = 0

# Memanggil fungsi deteksi mata kantuk secara periodik
gui.after(10, detect_drowsiness)

# Menjalankan GUI
gui.mainloop()