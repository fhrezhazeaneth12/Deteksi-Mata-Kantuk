import os
import cv2
import numpy as np
from PIL import Image, ImageTk #manipulasi gambar dan menampilkan gambar dalam GUI.
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# Membaca dataset dan melakukan preprocessing
images, labels = read_dataset()
images_processed = [preprocess_image(image) for image in images]
images_processed = np.array(images_processed)
images_processed = images_processed.reshape(images_processed.shape[0], -1)

# Mengubah label menjadi angka menggunakan LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(images_processed, labels_encoded, test_size=0.2, random_state=42)

# Melakukan PCA
pca = PCA(n_components=93)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Melakukan LDA
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train_pca, y_train)
X_test_lda = lda.transform(X_test_pca)

# Melatih model SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_lda, y_train)

# Menghitung akurasi pelatihan
training_accuracy = svm.score(X_train_lda, y_train)
print(f"Training Accuracy: {training_accuracy}")

# Menghitung akurasi pengujian
testing_accuracy = svm.score(X_test_lda, y_test)
print(f"Testing Accuracy: {testing_accuracy}")

# Prediksi hasil dari data uji
y_pred = svm.predict(X_test_lda)

# Membuat confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confussion Matrix : \n", conf_matrix)

# Menampilkan confusion matrix dalam bentuk heatmap
#plt.figure(figsize=(8, 6))
#sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Closed', 'Open'], yticklabels=['Closed', 'Open'])
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
#plt.title('Confusion Matrix')
#plt.show()

# Simpan model yang telah dilatih ke dalam file
file_path = r'D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\PROSES\DETEKSI KANTUK\KANTUK\trained_svm_model.pkl'
try:
    joblib.dump(svm, file_path)
    print("Model SVM telah disimpan dengan sukses di", file_path)
except Exception as e:
    print("Terjadi kesalahan saat menyimpan model:", str(e))