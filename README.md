
# Klasifikasi Jenis Batuan Menggunakan Ekstraksi Fitur GLCM dan Metode KNN, SVM, dan Random Forest dengan Berbagai Kombinasi Preprocessing

## Nama Anggota
- Lalu Moh Habib Adrian Maulana : F1D02410066
- Raissa Bunga Astrella : F1D02410087
- Ida Bagus Kevin Adiwiguna : F1D02410115

---

## Project Overview

Project ini merupakan implementasi Pengolahan Citra Digital (PCD) untuk melakukan klasifikasi jenis batuan berdasarkan dataset gambar. Dataset yang digunakan terdiri dari tiga kelas utama, yaitu:

- `Coal`
- `Limestone`
- `Sandstone`

Tujuan utama dari project ini adalah membandingkan beberapa tahapan preprocessing citra untuk melihat pengaruhnya terhadap hasil ekstraksi fitur dan performa model klasifikasi. Proses klasifikasi dilakukan dengan memanfaatkan fitur tekstur dari citra menggunakan metode **Gray Level Co-occurrence Matrix (GLCM)**, kemudian hasil fiturnya digunakan untuk melatih beberapa model machine learning.

Model klasifikasi yang digunakan dalam project ini adalah:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

Project ini tidak hanya berfokus pada nilai akurasi akhir, tetapi juga pada pemilihan preprocessing yang sesuai, proses ekstraksi fitur, serta analisis hasil evaluasi model. Terdapat empat kombinasi preprocessing yang dibandingkan, mulai dari preprocessing sederhana seperti resize dan grayscale, hingga preprocessing yang lebih kompleks seperti median filter, histogram equalization, gaussian blur, sharpening, thresholding, serta operasi morfologi opening dan closing.

---

## Struktur Repository

```text
-PCD-Project_Klasifikasi-Jenis-Batuan-Menggunakan-Ekstraksi-Fitur-GLCM/
│
├── Assets/
│   ├── Coal/
│   ├── Limestone/
│   └── Sandstone/
│
├── hasil_ekstraksi/
│   ├── hasil_ekstraksi_Percobaan1.csv
│   ├── hasil_ekstraksi_Percobaan2.csv
│   ├── hasil_ekstraksi_Percobaan3.csv
│   └── hasil_ekstraksi_Percobaan4.csv
│
├── hasil_klasifikasi/
│   ├── hasil_klasifikasi_Percobaan1.csv
│   ├── hasil_klasifikasi_Percobaan2.csv
│   ├── hasil_klasifikasi_Percobaan3.csv
│   └── hasil_klasifikasi_Percobaan4.csv
│
├── Percobaan1.ipynb
├── Percobaan2.ipynb
├── Percobaan3.ipynb
├── Percobaan4.ipynb
└── README.md
```

Keterangan:

- Folder `Assets/` berisi gambar asli yang dikelompokkan berdasarkan label kelas (`Coal`, `Limestone`, `Sandstone`).
- Folder `hasil_ekstraksi/` berisi file CSV hasil ekstraksi fitur GLCM dari setiap percobaan.
- Folder `hasil_klasifikasi/` berisi file CSV hasil klasifikasi dari setiap percobaan.
- File `Percobaan1.ipynb` — `Percobaan4.ipynb` berisi notebook utama untuk preprocessing, ekstraksi fitur, dan klasifikasi.
- File `README.md` berisi dokumentasi project.

---

## Import Library

Library yang digunakan pada project ini menyesuaikan kebutuhan setiap tahap, mulai dari pembacaan gambar, pengolahan citra, ekstraksi fitur, hingga klasifikasi.

Beberapa library utama yang digunakan:

```python
import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, ConfusionMatrixDisplay
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
```

---

## Load Data

Tahap pertama adalah membaca dataset gambar dari folder `Assets/`. Setiap subfolder pada folder Assets dianggap sebagai label kelas.

Label yang digunakan:

```text
Coal
Limestone
Sandstone
```

Dataset dibaca dengan mengambil seluruh file gambar berekstensi:

```text
.jpg, .jpeg, .png, .bmp, .webp, .jfif
```

Contoh alur load data:

```python
DATASET_DIR = PROJECT_ROOT / "Assets"

data = []
labels = []
file_name = []

valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"]

for sub_folder in os.listdir(DATASET_DIR):
    sub_folder_path = DATASET_DIR / sub_folder

    if not sub_folder_path.is_dir():
        continue

    for filename in os.listdir(sub_folder_path):
        img_path = sub_folder_path / filename

        if img_path.suffix.lower() not in valid_extensions:
            continue

        img = cv.imread(str(img_path))

        if img is None:
            continue

        data.append(img)
        labels.append(sub_folder)
        file_name.append(filename)
```

Pada project ini, gambar kemudian diseragamkan ukurannya sebelum masuk ke tahap preprocessing dan ekstraksi fitur.

---

## Data Understanding

Dataset terdiri dari tiga kelas citra batuan. Setiap kelas disimpan dalam folder yang berbeda sehingga proses labeling dapat dilakukan secara otomatis berdasarkan nama folder.

Distribusi dataset:

| Label | Jumlah Data |
|---|---:|
| `Coal` | 99 gambar |
| `Limestone` | 97 gambar |
| `Sandstone` | 102 gambar |
| **Total** | **298 gambar** |

Karakteristik umum dataset:

- Gambar berasal dari tiga jenis batuan yang berbeda secara visual maupun tekstur.
- Ukuran dan orientasi gambar dapat bervariasi.
- Background gambar tidak selalu seragam.
- Kondisi pencahayaan dapat berbeda pada setiap gambar.
- Beberapa jenis batuan memiliki warna atau tekstur yang cukup mirip satu sama lain, sehingga preprocessing dan ekstraksi fitur tekstur menjadi kunci dalam proses klasifikasi.

Karena dataset memiliki jumlah data yang relatif seimbang untuk setiap kelas, model tidak terlalu terdampak oleh masalah imbalance antar kelas.

---

## Data Preparation

### Data Augmentation

Pada project ini, data augmentation tidak menjadi tahap utama karena jumlah data pada tiap kelas sudah berada di atas rentang minimal (70–100 gambar). Jumlah tersebut sudah mencukupi untuk percobaan klasifikasi berbasis fitur tekstur GLCM.

Namun, augmentation tetap dapat diterapkan apabila ingin menambah variasi data, misalnya dengan:

- Rotasi gambar
- Flip horizontal
- Perubahan brightness
- Zoom atau cropping ringan

Augmentation dapat membantu model mengenali objek dalam berbagai posisi dan kondisi pencahayaan, tetapi pada project ini fokus utama diarahkan pada perbandingan kombinasi preprocessing.

---

## Preprocessing

Preprocessing dilakukan untuk menyeragamkan citra, mengurangi noise, memperbaiki kualitas visual, dan menonjolkan karakteristik tekstur tertentu sebelum dilakukan ekstraksi fitur GLCM.

Terdapat empat skenario kombinasi preprocessing yang diuji coba:

| Preprocessing | Tahapan |
|---|---|
| `Percobaan 1` | Resize → Grayscale |
| `Percobaan 2` | Resize → Grayscale → Median Filter → Histogram Equalization |
| `Percobaan 3` | Resize → Grayscale → Gaussian Blur → Sharpening |
| `Percobaan 4` | Resize → Grayscale → Thresholding → Opening → Closing |

### Penjelasan Preprocessing

#### 1. Resize

Resize digunakan untuk menyamakan ukuran seluruh citra. Hal ini diperlukan agar proses pengolahan citra dan ekstraksi fitur dapat berjalan konsisten di seluruh dataset.

#### 2. Grayscale

Grayscale digunakan untuk mengubah citra RGB/BGR menjadi citra keabuan. Karena metode GLCM bekerja pada hubungan intensitas piksel, citra grayscale lebih sesuai digunakan untuk ekstraksi fitur tekstur batuan.

#### 3. Median Filter

Median filter digunakan untuk mengurangi noise salt-and-pepper tanpa terlalu merusak tepi objek. Filter ini efektif pada citra batuan yang mengandung gangguan kecil akibat perbedaan resolusi atau kualitas kamera.

#### 4. Histogram Equalization

Histogram equalization digunakan untuk meningkatkan kontras citra. Teknik ini membantu memperjelas perbedaan intensitas pada tekstur batuan, terutama pada gambar dengan distribusi intensitas yang sempit atau pencahayaan yang kurang merata.

#### 5. Gaussian Blur

Gaussian blur digunakan untuk menghaluskan citra dengan mengurangi detail frekuensi tinggi. Teknik ini membantu meredam noise halus sebelum dilakukan proses sharpening.

#### 6. Sharpening

Sharpening digunakan untuk mempertajam tepi dan detail tekstur citra batuan. Teknik ini merupakan kebalikan dari blur, yaitu menonjolkan perbedaan intensitas antar piksel yang berdekatan.

#### 7. Thresholding

Thresholding digunakan untuk mengubah citra grayscale menjadi citra biner berdasarkan nilai ambang tertentu. Teknik ini memisahkan area terang dan gelap pada citra, yang dapat membantu dalam analisis pola tekstur batuan.

#### 8. Opening

Opening merupakan operasi morfologi yang terdiri dari erosi diikuti dilasi. Teknik ini digunakan untuk menghilangkan noise kecil (piksel putih yang terisolasi) pada citra biner hasil thresholding.

#### 9. Closing

Closing merupakan operasi morfologi yang terdiri dari dilasi diikuti erosi. Teknik ini digunakan untuk mengisi celah kecil (piksel hitam yang terisolasi) pada objek citra biner, sehingga bentuk objek menjadi lebih utuh.

---

## Skenario Percobaan

Project ini menggunakan empat kombinasi preprocessing untuk melihat pengaruhnya terhadap performa model klasifikasi:

| Percobaan | Preprocessing yang Digunakan |
|---|---|
| Percobaan 1 | Resize → Grayscale |
| Percobaan 2 | Resize → Grayscale → Median Filter → Histogram Equalization |
| Percobaan 3 | Resize → Grayscale → Gaussian Blur → Sharpening |
| Percobaan 4 | Resize → Grayscale → Thresholding → Opening → Closing |

Dengan skenario ini, setiap hasil klasifikasi dapat dibandingkan untuk mengetahui kombinasi preprocessing mana yang paling sesuai untuk dataset batuan yang digunakan.

---

## Feature Extraction

Tahap ekstraksi fitur dilakukan menggunakan metode **Gray Level Co-occurrence Matrix (GLCM)**.

GLCM digunakan untuk mengambil informasi tekstur dari citra berdasarkan hubungan spasial antar piksel. Pada project ini, fitur GLCM dihitung pada empat arah sudut:

- 0°
- 45°
- 90°
- 135°

Implementasi menggunakan `graycomatrix` dan `graycoprops` dari `skimage.feature`.

Fitur yang diekstraksi:

| Fitur | Keterangan |
|---|---|
| Contrast | Mengukur perbedaan intensitas antara piksel bertetangga |
| Homogeneity | Mengukur keseragaman tekstur citra |
| Dissimilarity | Mengukur tingkat perbedaan antar piksel |
| Entropy | Mengukur kompleksitas atau ketidakteraturan tekstur |
| ASM | Mengukur keseragaman distribusi nilai GLCM |
| Energy | Mengukur energi atau kekuatan pola tekstur |
| Correlation | Mengukur hubungan linear antar piksel |

Contoh struktur fungsi ekstraksi fitur:

```python
def extract_glcm_features(image):
    angles = [0, 45, 90, 135]
    features = {}

    for angle in angles:
        matriks = glcm(image, angle)

        features[f"Contrast{angle}"]      = contrast(matriks)
        features[f"Homogeneity{angle}"]   = homogenity(matriks)
        features[f"Dissimilarity{angle}"] = dissimilarity(matriks)
        features[f"Entropy{angle}"]       = entropyGlcm(matriks)
        features[f"ASM{angle}"]           = ASM(matriks)
        features[f"Energy{angle}"]        = energy(matriks)
        features[f"Correlation{angle}"]   = correlation(matriks)

    return features
```

Hasil ekstraksi fitur disimpan ke dalam folder `hasil_ekstraksi/` dengan nama file:

```text
hasil_ekstraksi_Percobaan1.csv
hasil_ekstraksi_Percobaan2.csv
hasil_ekstraksi_Percobaan3.csv
hasil_ekstraksi_Percobaan4.csv
```

---

## Feature Selection

Feature selection dilakukan untuk memilih fitur yang paling relevan dan mengurangi fitur yang terlalu berkorelasi satu sama lain.

Pada project ini, feature selection dilakukan menggunakan pendekatan **correlation**. Tahap ini membantu mengurangi redundansi fitur sehingga proses klasifikasi menjadi lebih efisien.

Contoh pendekatan correlation:

```python
correlation = hasilEkstrak.drop(columns=["Label", "Filename"]).corr()
```

Fitur yang memiliki korelasi terlalu tinggi dapat dipertimbangkan untuk dihapus agar model tidak mempelajari informasi yang berulang.

---

## Splitting Data

Dataset hasil ekstraksi fitur dibagi menjadi data training dan data testing dengan perbandingan:

```text
80% data training
20% data testing
```

Contoh kode:

```python
X = df.drop(columns=["Label", "Filename"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Penggunaan `stratify=y` bertujuan agar distribusi label pada data training dan testing tetap seimbang sesuai proporsi kelas dalam dataset.

---

## Normalization

Normalisasi dilakukan agar seluruh fitur memiliki skala nilai yang lebih seragam. Hal ini penting terutama untuk model KNN dan SVM yang sensitif terhadap jarak atau skala fitur.

Metode normalisasi yang digunakan adalah **StandardScaler**:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

---

## Modeling

Model klasifikasi yang digunakan pada project ini terdiri dari tiga algoritma:

### 1. K-Nearest Neighbors (KNN)

KNN mengklasifikasikan data berdasarkan kedekatan jarak dengan data training. Model ini cukup sederhana namun sangat dipengaruhi oleh skala fitur, sehingga membutuhkan normalisasi sebelum digunakan.

### 2. Support Vector Machine (SVM)

SVM bekerja dengan mencari hyperplane terbaik untuk memisahkan kelas. Model ini cocok untuk data berdimensi tinggi seperti hasil ekstraksi fitur GLCM.

### 3. Random Forest

Random Forest merupakan model ensemble berbasis decision tree. Model ini cukup stabil terhadap variasi fitur dan dapat menangani hubungan non-linear antar fitur tanpa memerlukan normalisasi.

Contoh training model:

```python
knn = KNeighborsClassifier()
svm = SVC()
rf  = RandomForestClassifier(random_state=42)

knn.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
```

---

## Evaluation

Evaluasi model dilakukan untuk mengetahui performa klasifikasi pada setiap kombinasi preprocessing.

Metrik evaluasi yang digunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Contoh kode evaluasi:

```python
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```

### Tabel Hasil Evaluasi

| Preprocessing | Model | Train Accuracy | Test Accuracy | F1-Score |
|:---|:---|:---:|:---:|:---:|
| Resize → Grayscale | Random Forest | 94.96% | 58.33% | 58.24% |
| Resize → Grayscale | SVM | 65.13% | 60.00% | 59.51% |
| Resize → Grayscale | KNN | 69.33% | 41.67% | 40.32% |
| Resize → Grayscale → Median Filter → Histogram Equalization | Random Forest | 93.70% | 50.00% | 48.73% |
| Resize → Grayscale → Median Filter → Histogram Equalization | SVM | 61.34% | 56.67% | 55.96% |
| Resize → Grayscale → Median Filter → Histogram Equalization | KNN | 61.76% | 53.33% | 52.70% |
| Resize → Grayscale → Gaussian Blur → Sharpening | Random Forest | 94.54% | 58.33% | 58.93% |
| Resize → Grayscale → Gaussian Blur → Sharpening | SVM | 69.75% | 60.00% | 59.56% |
| Resize → Grayscale → Gaussian Blur → Sharpening | KNN | 67.65% | 51.67% | 51.56% |
| Resize → Grayscale → Thresholding → Opening → Closing | Random Forest | 92.44% | 40.00% | 40.10% |
| Resize → Grayscale → Thresholding → Opening → Closing | SVM | 57.56% | 45.00% | 45.70% |
| Resize → Grayscale → Thresholding → Opening → Closing | KNN | 68.91% | 45.00% | 42.71% |

### Analisis Evaluasi

- Berdasarkan hasil pengujian, preprocessing pertama (Resize → Grayscale) dan preprocessing ketiga (Resize → Grayscale → Gaussian Blur → Sharpening) menghasilkan performa yang relatif serupa dan paling baik dibandingkan skenario lainnya. Model SVM pada kedua preprocessing tersebut sama-sama memperoleh test accuracy sebesar 60.00%, sedangkan Random Forest memperoleh 58.33%. Hal ini menunjukkan bahwa informasi tekstur batuan pada citra grayscale sudah cukup representatif untuk diekstraksi menggunakan GLCM.

- Pada preprocessing kedua (Resize → Grayscale → Median Filter → Histogram Equalization), performa model mengalami penurunan dibandingkan preprocessing pertama. Random Forest hanya memperoleh test accuracy 50.00%, sedangkan SVM 56.67% dan KNN 53.33%. Meskipun histogram equalization dapat meningkatkan kontras, perubahan distribusi intensitas piksel kemungkinan menyebabkan pola tekstur yang diekstraksi GLCM menjadi kurang konsisten.

- Pada preprocessing keempat (Resize → Grayscale → Thresholding → Opening → Closing), performa seluruh model mengalami penurunan paling signifikan. Random Forest, SVM, dan KNN hanya memperoleh test accuracy masing-masing 40.00%, 45.00%, dan 45.00%. Proses thresholding mengubah citra menjadi bentuk biner sehingga sebagian besar informasi gradasi intensitas tekstur hilang. Operasi morfologi opening dan closing turut mengubah struktur citra, sehingga fitur GLCM yang dihasilkan menjadi kurang representatif untuk membedakan jenis batuan.

- Secara keseluruhan, preprocessing Resize → Grayscale dan Resize → Grayscale → Gaussian Blur → Sharpening menghasilkan performa terbaik dengan SVM sebagai model yang paling konsisten memperoleh test accuracy tertinggi (60.00%) di dua skenario tersebut. Hal ini mengindikasikan bahwa informasi tekstur asli pada citra batuan sudah memadai, dan preprocessing yang terlalu agresif seperti thresholding justru cenderung menurunkan kualitas fitur tekstur yang diekstraksi.

---

## Cara Menjalankan Project

1. Clone repository:

```bash
git clone https://github.com/KevinAdiwiguna/-PCD-Project_Klasifikasi-Jenis-Batuan-Menggunakan-Ekstraksi-Fitur-GLCM.git
cd -PCD-Project_Klasifikasi-Jenis-Batuan-Menggunakan-Ekstraksi-Fitur-GLCM
```

2. Install library yang dibutuhkan:

```bash
pip install numpy pandas matplotlib opencv-python scikit-image scipy scikit-learn seaborn
```

3. Jalankan notebook sesuai percobaan yang diinginkan:

```text
Percobaan1.ipynb   → Preprocessing: Resize + Grayscale
Percobaan2.ipynb   → Preprocessing: Resize + Grayscale + Median Filter + Histogram Equalization
Percobaan3.ipynb   → Preprocessing: Resize + Grayscale + Gaussian Blur + Sharpening
Percobaan4.ipynb   → Preprocessing: Resize + Grayscale + Thresholding + Opening + Closing
```

Setiap notebook sudah mencakup proses preprocessing, ekstraksi fitur GLCM, feature selection, splitting data, normalisasi, training model, dan evaluasi secara lengkap.

---

## Output Project

Output utama dari project ini adalah:

```text
hasil_ekstraksi/
  ├── hasil_ekstraksi_Percobaan1.csv
  ├── hasil_ekstraksi_Percobaan2.csv
  ├── hasil_ekstraksi_Percobaan3.csv
  └── hasil_ekstraksi_Percobaan4.csv

hasil_klasifikasi/
  ├── hasil_klasifikasi_Percobaan1.csv
  ├── hasil_klasifikasi_Percobaan2.csv
  ├── hasil_klasifikasi_Percobaan3.csv
  └── hasil_klasifikasi_Percobaan4.csv
```

- Folder `hasil_ekstraksi/` berisi file CSV fitur GLCM dari masing-masing skenario preprocessing.
- Folder `hasil_klasifikasi/` berisi file CSV dengan nilai Train Accuracy, Test Accuracy, Precision, Recall, dan F1-Score dari setiap model pada setiap percobaan.

---

## Kesimpulan

Berdasarkan hasil penelitian yang telah dilakukan, dapat disimpulkan bahwa tahapan preprocessing berpengaruh terhadap performa klasifikasi citra batuan menggunakan fitur GLCM dan model machine learning.

Dari empat skenario preprocessing yang diuji, preprocessing Resize → Grayscale (Percobaan 1) dan Resize → Grayscale → Gaussian Blur → Sharpening (Percobaan 3) menghasilkan performa terbaik, di mana SVM berhasil mencapai test accuracy tertinggi sebesar 60.00% pada kedua skenario tersebut. Hal ini menunjukkan bahwa informasi tekstur pada citra batuan sudah dapat direpresentasikan dengan baik menggunakan citra grayscale, dan penambahan gaussian blur serta sharpening tidak mengurangi kualitas fitur yang diekstraksi secara signifikan.

Sebaliknya, penambahan preprocessing seperti Histogram Equalization dan Thresholding cenderung menurunkan performa klasifikasi. Proses thresholding khususnya menyebabkan hilangnya informasi gradasi intensitas yang menjadi fondasi utama ekstraksi fitur GLCM, sehingga model tidak mampu membedakan antar kelas batuan dengan baik.

Dari sisi model, SVM menunjukkan konsistensi performa terbaik pada sebagian besar skenario preprocessing, diikuti oleh Random Forest yang memiliki train accuracy tinggi namun cenderung mengalami overfitting. KNN secara konsisten memperoleh test accuracy terendah, kemungkinan akibat sensitivitasnya terhadap distribusi fitur yang kompleks pada data citra batuan.

Secara keseluruhan, kombinasi Resize → Grayscale, ekstraksi fitur GLCM, dan model SVM menjadi pendekatan yang paling direkomendasikan pada project ini karena menghasilkan performa test accuracy tertinggi dengan karakteristik generalisasi yang lebih stabil.
```
