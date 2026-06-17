# Klasifikasi Jenis Batuan Menggunakan Ekstraksi Fitur GLCM dan Metode KNN, SVM, dan Random Forest dengan Berbagai Kombinasi Preprocessing
## Nama Anggota
###  Lalu Moh Habib Adrian Maulana : F1D02410066
###  Raissa Bunga Astrella : F1D02410087
###  Ida Bagus Kevin Adiwiguna: F1D02410115

# Project Overview
Project ini merupakan implementasi Pengolahan Citra Digital (PCD) untuk melakukan klasifikasi jenis batuan berdasarkan dataset gambar. Dataset yang digunakan terdiri dari tiga kelas utama, yaitu:

* Coal
* Limestone
* Sandstone

Tujuan utama dari project ini adalah membandingkan beberapa tahapan preprocessing citra untuk melihat pengaruhnya terhadap hasil ekstraksi fitur dan performa model klasifikasi. Proses klasifikasi dilakukan dengan memanfaatkan fitur tekstur dari citra menggunakan metode Gray Level Co-occurrence Matrix (GLCM), kemudian hasil fiturnya digunakan untuk melatih beberapa model machine learning.
Model klasifikasi yang digunakan dalam project ini adalah:

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Random Forest

Project ini tidak hanya berfokus pada nilai akurasi akhir, tetapi juga pada pemilihan preprocessing yang sesuai, proses ekstraksi fitur, serta analisis hasil evaluasi model. Terdapat empat kombinasi preprocessing yang dibandingkan, mulai dari preprocessing sederhana seperti resize dan grayscale, hingga preprocessing yang lebih kompleks seperti median filter, histogram equalization, gaussian blur, sharpening, thresholding, serta operasi morfologi opening dan closing.

## Struktur Repository

```text
PCD-Project_Klasifikasi-Jenis-Batuan-Menggunakan-Ekstraksi-Fitur-GLCM/
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
│   └── hasil_klasifikasi_Percobaan3.csv
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
- File `Percobaan1.ipynb` — `Percobaan4.ipynb` berisi notebook utama untuk preprocessing, ekstraksi fitur, klasifikasi, dan percobaan morfologi.
- File `README.md` berisi dokumentasi project.

# IMPORT LIBRARY
``` python
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)
import seaborn as sns
```
# Load Data
Setelah import library, dilanjutkan dengan tahapan membaca dataset. Pada praktikum modul 1 - 5 Anda pernah membaca beberapa image ke dalam code. Pada project ini, Anda tidak hanya akan membaca 1 atau 2 image saja, tetapi ratusan bahkan ribuan image pada dataset yang Anda gunakan. Bukan hanya image, tapi Anda juga akan berurusan dengan label setiap image, jadi sesuaikan code pada template dengan dataset label (label adalah nama setiap folder pada dataset Anda yang berisi image) yang Anda miliki. Pertama-tama lakukanlah data loading (baca dataset) beserta labelnya, Anda bisa melakukan penyeragaman ukuran dari dataset dengan resize, jika ukuran setiap image berbeda pada datset Anda. Misalnya ada yang 100x200, 300x100 maka Anda harus mengubahnya ke ukuran yang sama misalnya 100x100 atau 150x150. Sekedar informasi semakin besar ukuran setiap image, maka proses loadingnya pun akan semakin lama, jadi usahakan juga ukuran image rendah, CMIIW~
``` python
  data = []
  labels = []
  file_name = []
```
## Data Understanding
Selanjutnya, Anda diminta untuk melakukan eksplorasi data untuk memahami karakteristik data yang digunakan. Anda dapat menampilkan jumlah data, karakteristik data (kondisi background, noise, pencahyaan, dll), distribusi data, sampel data, dan lainnya. Hal ini bertujuan untuk memahami data yang akan digunakan dalam proses klasifikasi, sehingga dapat memilih teknik preprocessing yang tepat ataupun penanganan jika terdapat data yang tidak seimbang. Berikut ini contohnya:
``` python
  jumlah.data = []
  jumlah.labels = []
  print(Output: file_name)
```
Output: Contoh Visualisasi Distribusi Data: 

![image](https://github.com/user-attachments/assets/bcf4e18c-d6a5-4627-a4d3-c4a2fdb35e8c)

Output: Contoh Sample Data:
![image](https://github.com/user-attachments/assets/0084d31f-386e-49f9-9de5-4863ec4d73de)

# Data Preparation
## Data Augmentation
Pada tahapan ini, Anda diwajibkan untuk menerapkan teknik image augmentation untuk menambah jumlah data, HANYA JIKA data Anda berada di bawah rentang 70-100. Anda bisa melakukan image Augmentation dengan teknik yang ada pada Modul 1.
``` python
  augmented.data = []
  augmented.labels = []
  print(Output: file_name)
```
Output: Contoh Image Augmentation
![image](https://github.com/user-attachments/assets/9ea656a7-a69c-47fa-98fc-2a598b81c3a0)

## Preprocessing
Selanjutnya, ini dia tahapan paling krusial. Anda dapat melakukan teknik preprocessing yang Anda anggap perlu. Jelaskan alasan Anda menggunakan teknik tersebut, Anda wajib menggunakan preprocessing yang ada pada modul-modul yang telah Anda pelajari sebelumnya selama praktikum. Jika Anda merasa preprocessing yang ada pada praktikum tidak sesuai, maka silahkan diskusikan dengan Asisten masing" untuk mendapatkan pencerahan.
``` python
def prepro1():
    pass

def prepro2():
    pass

def prepro3():
    pass
```
## Feature Extraction
Pada tahapan ini, Anda diminta untuk melakukan ekstraksi fitur dengan metode Gray Level Co-occurrence Matrix (GLCM). Dengan GLCM sudut 0, 45, 90, dan 135 derajat, simetris, dan lakukan uji coba dengan distance 1-5. Anda dapat menghitung nilai dari beberapa fitur berikut:

- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation
- Entropy
- ASM
``` python
def glcm(image, derajat):
 ...........
```

## Feature Selection
Pada tahap ini, Anda diminta untuk melakukan seleksi fitur. Anda dapat menggunakan teknik seleksi fitur seperti correlation, PCA, atau teknik seleksi fitur lain yang Anda ketahui. NAH PADA TEMPLATE CODE FEATURE SELECTION MENGGUNAKAN CORRELATION YGY.
``` python
correlation = hasilEkstrak.drop(columns=['Label','Filename']).corr()
......
```

## Splitting Data
Pada tahap ini, Anda diminta untuk membagi data menjadi data training dan data testing. Anda dapat menggunakan perbandingan 80:20 atau 70:30 atau 90:10.
``` python
Dataset, Dataset, Dataset, Dataset = train_test_split(Dataset, y, test_size=0.2, random_state=42)
print(Dataset.shape)
print(Dataset.shape)
```

## Normalization
Pada tahap ini, Anda diminta untuk melakukan normalisasi data. Anda dapat menggunakan teknik normalisasi standarization atau min-max normalization.
``` python
Dataset = (Dataset - Dataset.mean()) / Dataset.std()
Dataset = (Dataset - Dataset.mean()) / Dataset.std()
```

# Modeling
Pada tahap ini, Anda diminta untuk membuat model klasifikasi. Berikut merupakan model yang dapat digunakan:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

Gunakan akurasi sebagai metrik dalam menampilkan hasil klasifikasi.
```python
# Train Random Forest Classifier
rf.fit(dataset, dataset)
# Train SVM Classifier
svm.fit(dataset, dataset)
# Train KNN Classifier
knn.fit(dataset, dataset)

def inidiaClassificationReport(dataset, dataset):
	print(classification_report(dataset, dataset))

```
Output: Contoh Classsification Report
|               | Accuracy | Precision | Recall   | F1-Score |
| ------------- | -------- | --------- | -------- | -------- |
| KNN           | 0.948667 | 0.948664  | 0.948667 | 0.948504 |
| SVM           | 0.976333 | 0.976319  | 0.976333 | 0.976333 |
| Random Forest | 0.959667 | 0.959822  | 0.959667 | 0.959615 |

# Evaluation
Pada bagian ini, Anda perlu mengevaluasi model klasifikasi yang telah Anda buat dengan menampilkan Confusion Matrix, dan Clasification Report: Accuracy, Precision, Recall, F1 Score. **Jelaskan hasil evaluasi yang Anda dapatkan dan berikan analisis mengenai hasil evaluasi tersebut**.

``` python

def plot_confusion_matrix(dataset, dataset, title):
  print(confusion_matrix)
```

Output: Contoh Confusion Matrix
![image](https://github.com/user-attachments/assets/aec4ac9c-e687-4354-b02d-833caf26db6b)
