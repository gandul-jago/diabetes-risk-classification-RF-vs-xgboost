# Perbandingan Kinerja Random Forest dan XGBoost dalam Klasifikasi Risiko Diabetes

Studi komparatif dua algoritma machine learning berbasis pohon keputusan untuk mendeteksi risiko diabetes menggunakan data survei kesehatan masyarakat Amerika Serikat (BRFSS 2015).

**Resume:** [Drive Resume](https://drive.google.com/drive/folders/1EPa6WbEtUVAUumknIie4kwXuA-OND3z7?usp=sharing)

**Peneliti:** Ghaniya Syifa Talitha (243303611239)

---

## Dataset

Dataset yang digunakan adalah **Diabetes Health Indicators Dataset** dari Kaggle, hasil survei Behavioral Risk Factor Surveillance System (BRFSS) 2015 oleh CDC Amerika Serikat.

- **Sumber:** [kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **File:** `diabetes_binary_health_indicators_BRFSS2015.csv`
- **Jumlah data:** 253.680 baris
- **Jumlah fitur:** 21 fitur + 1 label target (`Diabetes_binary`)
- **Missing value:** 0
- **Tipe data:** Seluruh kolom `float64`
- **Distribusi kelas:** Non-Diabetes 86,7% | Diabetes 13,3%

Fitur-fitur mencakup indikator kesehatan seperti tekanan darah tinggi (HighBP), kolesterol tinggi (HighChol), indeks massa tubuh (BMI), kondisi kesehatan umum (GenHlth), usia, tingkat pendidikan, dan pendapatan.

---

## Alur Kerja

```
Load Data → EDA → Preprocessing & Split → Training + Tuning → Evaluasi → Uji Statistik Wilcoxon
```

---

## Exploratory Data Analysis

- Tidak ada missing value maupun data duplikat
- Terdapat class imbalance signifikan: hanya 35.346 data diabetes vs 218.334 non-diabetes
- Fitur paling berkorelasi dengan label: `GenHlth`, `HighBP`, `BMI`, `DiffWalk`, `HighChol`
- BMI memiliki distribusi mendekati normal dengan beberapa outlier ekstrem
- `MentHlth` dan `PhysHlth` memiliki distribusi sangat right-skewed

---

## Preprocessing & Split Data

- Tidak diperlukan encoding (semua fitur sudah numerik)
- Tidak dilakukan normalisasi karena kedua algoritma bersifat tree-based (tidak sensitif terhadap skala)
- Split: **80% train / 20% test** dengan `stratify=y`
- Penanganan class imbalance dilakukan di tiga titik:
  - `stratify=y` saat split
  - `class_weight='balanced'` pada Random Forest
  - `scale_pos_weight=6.18` pada XGBoost

---

## Hyperparameter Tuning

Kedua model di-tuning menggunakan `RandomizedSearchCV` (10 iterasi, 3-fold CV, metrik: ROC-AUC).

**Random Forest — Parameter Terbaik**

| Hyperparameter | Nilai |
|---|---|
| n_estimators | 100 |
| max_depth | 10 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| max_features | log2 |
| CV ROC-AUC (Train) | 0.827 |

**XGBoost — Parameter Terbaik**

| Hyperparameter | Nilai |
|---|---|
| n_estimators | 200 |
| max_depth | 3 |
| learning_rate | 0.05 |
| subsample | 1.0 |
| colsample_bytree | 0.9 |
| scale_pos_weight | 6.18 |
| CV ROC-AUC (Train) | 0.8303 |

---

## Hasil Evaluasi

Evaluasi dilakukan pada 50.736 data uji.

| Metrik | Random Forest | XGBoost |
|---|---|---|
| Accuracy | **74%** | 72% |
| Precision | **32%** | 31% |
| Recall | 79% | **80%** |
| F1-Score | **46%** | 44% |
| ROC-AUC | 0.823 | **0.826** |

XGBoost unggul dalam **Recall** dan **ROC-AUC**, sementara Random Forest lebih baik dalam **Accuracy**, **Precision**, dan **F1-Score**. Recall yang tinggi sangat penting dalam konteks ini karena kesalahan melewatkan penderita diabetes (*false negative*) lebih berbahaya daripada false positive.

---

## Feature Importance

**Random Forest** — top 5: `GenHlth` > `HighBP` > `BMI` > `HighChol` > `Age`

Persepsi seseorang terhadap kondisi kesehatannya sendiri (GenHlth) menjadi prediktor terkuat pada Random Forest.

**XGBoost** — top 5: `HighBP` > `GenHlth` > `HighChol` > `DiffWalk` > `Age`

XGBoost lebih menekankan kondisi medis terukur seperti hipertensi dibanding persepsi kesehatan subjektif.

---

## Uji Statistik Wilcoxon

Uji **Wilcoxon Signed-Rank Test** dilakukan menggunakan 5-fold Stratified Cross-Validation untuk membuktikan signifikansi perbedaan performa.

| Fold | Random Forest AUC | XGBoost AUC |
|---|---|---|
| Fold 1 | 0.8305 | 0.8339 |
| Fold 2 | 0.8264 | 0.8290 |
| Fold 3 | 0.8269 | 0.8299 |
| Fold 4 | 0.8264 | 0.8301 |
| Fold 5 | 0.8221 | 0.8253 |
| Rata-rata | 0.8265 | 0.8296 |

| Parameter | Hasil |
|---|---|
| Metode Uji | Wilcoxon Signed-Rank Test |
| Jumlah Fold (n) | 5 |
| p-value | 0.0625 |
| Threshold Signifikansi | 0.05 |
| Kesimpulan | Tidak signifikan secara statistik (p ≥ 0.05) |

Meskipun XGBoost konsisten lebih tinggi di setiap fold, perbedaan tersebut tidak cukup besar untuk dinyatakan signifikan secara statistik. Kedua model memiliki kemampuan generalisasi yang relatif setara.

---

## Kesimpulan

- Kedua model mencapai AUC > 0.82, menunjukkan kemampuan klasifikasi yang baik pada data tidak seimbang
- **XGBoost direkomendasikan** jika prioritas adalah meminimalkan kasus diabetes yang tidak terdeteksi (Recall lebih tinggi)
- **Random Forest tetap kompetitif** jika efisiensi komputasi dan interpretabilitas menjadi pertimbangan
- Fitur paling berpengaruh di kedua model: `GenHlth`, `HighBP`, `BMI`, `HighChol`
- Penanganan class imbalance dengan `class_weight` dan `scale_pos_weight` terbukti efektif menghasilkan Recall 79–80% pada kelas minoritas

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
```

---

## Struktur File

```
├── xgboost-x-rf.ipynb    # Notebook utama
├── Resume.docx           # Laporan lengkap
└── README.md
```

---

## Cara Menjalankan

1. Download dataset dari link Kaggle di atas
2. Sesuaikan path dataset di dalam notebook
3. Jalankan `xgboost-x-rf.ipynb` dari atas ke bawah
