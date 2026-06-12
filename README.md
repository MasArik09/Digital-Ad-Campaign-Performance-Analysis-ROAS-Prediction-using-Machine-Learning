# 📊 Digital Ad Campaign Performance Analysis & ROAS Prediction
> **Machine Learning & Data Mining Project for Marketing Budget Optimization**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-bluevioloet?style=flat)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)

Proyek ini bertujuan untuk menganalisis performa kampanye iklan digital global dan memprediksi **ROAS (Return on Ad Spend)** menggunakan algoritma Machine Learning (Regresi & K-Means Clustering). Analisis ini dirancang untuk membantu pemasar (*marketers*) mengoptimalkan alokasi anggaran pemasaran di berbagai platform, industri, dan wilayah geografis.

---

## 📂 Struktur Repositori
```text
├── data set/
│   └── global_ads_performance_dataset.csv  # Dataset performa iklan (1.800 baris x 14 kolom)
├── Digital_Ads_ML_Analysis.ipynb          # Jupyter Notebook utama (analisis & pemodelan)
└── README.md                             # Dokumentasi proyek (file ini)
```

---

## 📈 Alur Metodologi & Notebook
Analisis dalam notebook terstruktur menjadi 8 tahap utama:
1. **Setup & Import Library**: Memuat pustaka analisis data dan machine learning.
2. **Load & Eksplorasi Data (EDA)**: Memeriksa tipe data, *missing values*, duplikat, dan statistik deskriptif.
3. **Visualisasi Distribusi & Korelasi**: Menganalisis tren bulanan, korelasi fitur numerik, ROAS per negara, industri, dan platform.
4. **Feature Engineering & Preprocessing**: Membuat fitur rasio baru, ekstraksi waktu, *label encoding* untuk fitur kategorikal, dan normalisasi fitur.
5. **Modeling (Regresi)**: Melatih model regresi untuk memprediksi ROAS.
6. **Evaluasi Model**: Membandingkan hasil RMSE, MAE, dan $R^2$ Score.
7. **Clustering Kampanye**: Segmentasi kampanye menggunakan K-Means dan PCA.
8. **Insight & Rekomendasi Bisnis**: Merumuskan saran alokasi anggaran berbasis data.

---

## 📊 Dataset & Rekayasa Fitur (Feature Engineering)
Dataset yang digunakan adalah **Global Ads Performance Dataset** dengan periode kampanye **Januari - Desember 2024**.

### 1. Fitur Utama Dataset:
* **Dimensi Kampanye**: `date`, `platform` (Google Ads, TikTok Ads, Meta Ads), `campaign_type` (Search, Video, Display, Social, dll), `industry` (Fintech, EdTech, Healthcare, E-commerce, dll), `country` (UAE, UK, USA, dll).
* **Metrik Performa**: `impressions`, `clicks`, `CTR` (Click-Through Rate), `CPC` (Cost Per Click), `ad_spend` (Biaya Iklan), `conversions`, `CPA` (Cost Per Acquisition), `revenue` (Pendapatan).
* **Target**: `ROAS` (Return on Ad Spend = Revenue / Ad Spend).

### 2. Fitur Baru yang Dibuat (Feature Engineering):
* `click_to_conv_rate`: Rasio konversi terhadap klik.
* `revenue_per_click`: Rata-rata pendapatan per klik.
* `spend_efficiency`: Rasio efisiensi pengeluaran iklan.
* `imp_to_click_rate`: Rasio klik terhadap impresi.
* `quarter` & `day_of_week`: Fitur waktu dari tanggal kampanye.
* `is_weekend`: Indikator biner apakah kampanye berjalan di akhir pekan.

---

## 🤖 Pemodelan Regresi & Hasil Evaluasi
Prediksi ROAS dilakukan dengan membandingkan 4 model regresi utama. Evaluasi menggunakan metode Train-Test Split (80:20) dan 5-Fold Cross Validation.

Berikut adalah tabel perbandingan performa model (diurutkan dari yang terbaik):

| Peringkat | Model Regresi | RMSE | MAE | $R^2$ Test | Mean $R^2$ CV | Std $R^2$ CV |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Linear Regression** | **0.0089** | **0.0048** | **1.0000** | **1.0000** | **0.0000** |
| 🥈 | **Ridge Regression** | 0.0135 | 0.0077 | 1.0000 | 1.0000 | 0.0000 |
| 🥉 | **Random Forest** | 0.1676 | 0.0365 | 0.9995 | 0.9982 | 0.0023 |
| 🏅 | **XGBoost** | 0.5472 | 0.2220 | 0.9942 | 0.9924 | 0.0055 |

> **Catatan Teknis**: Model Linear Regression mencapai performa sempurna ($R^2 = 1.0$) karena nilai target ROAS secara matematis diturunkan langsung secara linear dari pembagian `revenue` dan `ad_spend` (fitur yang digunakan dalam model).

---

## 🎯 Segmentasi Kampanye (K-Means Clustering)
Menggunakan **K-Means Clustering** (dengan $K=4$ optimal berdasarkan metode Elbow) dan divisualisasikan menggunakan reduksi dimensi **PCA** (menjelaskan 69.4% variansi).

### Profil Rata-Rata per Cluster:

| Parameter | Cluster 0 (Inefficient) | Cluster 1 (Stable/Average) | Cluster 2 (High-Volume) | Cluster 3 (Super Efficient Stars) |
| :--- | :---: | :---: | :---: | :---: |
| **Jumlah Kampanye** | 322 | 985 | 251 | 242 |
| **CTR** | 4.0% | 3.0% | **6.0%** | 4.0% |
| **CPC** | $2.36 (Tinggi) | $1.40 | $2.02 | **$0.77 (Rendah)** |
| **CPA** | $116.14 (Sangat Tinggi) | $34.52 | $37.91 | **$12.34 (Sangat Rendah)** |
| **Ad Spend** | $8,908.67 | $3,553.30 | $15,416.12 | $3,597.98 |
| **Revenue** | $13,585.17 | $16,053.90 | $78,398.00 | $59,164.94 |
| **ROAS** | **1.55 (Sangat Rendah)** | 5.28 | 5.57 | **18.64 (Sangat Tinggi!)** |
| **Rasio Konversi** | 2.0% | 5.0% | 6.0% | 6.0% |

### Karakteristik & Strategi Cluster:
* **Cluster 0 (Low Performers/Inefficient)**: Biaya per akuisisi (CPA) dan klik (CPC) sangat mahal. Kampanye pada cluster ini membutuhkan evaluasi ulang atau penghentian sementara karena ROAS sangat rendah (1.55).
* **Cluster 1 (Average/Stable)**: Memiliki performa yang stabil dengan ROAS rata-rata 5.28. Sangat baik untuk menjaga performa dasar (*baseline*).
* **Cluster 2 (High-Budget / High-Volume)**: Kampanye dengan anggaran besar ($15.4k+) yang menghasilkan volume konversi dan pendapatan tertinggi. Memiliki CTR terbaik (6%). Cocok untuk meningkatkan kesadaran merek (*brand awareness*) skala besar.
* **Cluster 3 (Stars/Super Efficient)**: Kampanye paling menguntungkan dengan ROAS luar biasa (18.64). Efisiensi biaya sangat tinggi (CPC $0.77, CPA $12.34). **Strategi: Alokasikan anggaran tambahan ke segmen ini untuk scale-up keuntungan.**

---

## 💡 Temuan Utama & Rekomendasi Bisnis

### 📊 Ringkasan Temuan:
* **Platform Terbaik**: **TikTok Ads** (Rata-rata ROAS: **9.54**) jauh mengungguli Google Ads dan Meta Ads.
* **Industri Terbaik**: **EdTech** (Rata-rata ROAS: **6.83**).
* **Tipe Kampanye Terbaik**: **Search Ads** (Rata-rata ROAS: **7.00**).
* **Negara Terbaik**: **UAE** (Rata-rata ROAS: **6.96**).

### 🚀 Rekomendasi Strategis:
1. **Alokasi Budget**: Prioritaskan pengeluaran iklan pada **TikTok Ads** untuk mendongkrak ROAS keseluruhan secara cepat.
2. **Targeting Industri**: Fokuskan sumber daya dan kreativitas kampanye pada segmen **EdTech** yang terbukti sangat efisien dalam berkonversi.
3. **Optimasi Tipe Kampanye**: Tingkatkan alokasi pada **Search Ads** karena menghasilkan return tertinggi dibanding Display atau Video Ads.
4. **Strategi Skala**: Cari kampanye dengan karakteristik profil **Cluster 3** (efisiensi biaya tinggi) untuk digandakan anggarannya (*scale-up*).
5. **Prediksi Proaktif**: Integrasikan model regresi (Linear Regression/Ridge) untuk memperkirakan ROAS kampanye baru sebelum anggaran dihabiskan sepenuhnya.

---

## 🛠️ Cara Menjalankan Notebook Secara Lokal

### 1. Prasyarat (Prerequisites)
Pastikan Anda memiliki Python 3.8+ dan `pip` terinstal di sistem Anda.

### 2. Kloning Repositori
```bash
git clone https://github.com/MasArik09/Digital-Ad-Campaign-Performance-Analysis-ROAS-Prediction-using-Machine-Learning.git
cd Digital-Ad-Campaign-Performance-Analysis-ROAS-Prediction-using-Machine-Learning
```

### 3. Instalasi Library
Instal pustaka yang dibutuhkan menggunakan perintah berikut:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyter
```

### 4. Menjalankan Jupyter Notebook
Jalankan perintah berikut untuk membuka notebook:
```bash
jupyter notebook Digital_Ads_ML_Analysis.ipynb
```
Atau jika Anda menggunakan Visual Studio Code, Anda dapat langsung membuka file `.ipynb` tersebut setelah memasang ekstensi Jupyter.

---

## ⚙️ Teknologi yang Digunakan
* **Bahasa Pemrograman**: Python
* **Analisis & Manipulasi Data**: Pandas, NumPy
* **Visualisasi Data**: Matplotlib, Seaborn
* **Machine Learning**: Scikit-Learn (Linear Regression, Ridge, Random Forest, K-Means Clustering, PCA, Standard Scaler, Label Encoder)
* **Model Boosting**: XGBoost

---

## 📝 Referensi
* Dokumentasi Scikit-Learn: [scikit-learn.org](https://scikit-learn.org)
* Dokumentasi XGBoost: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)
* Dataset: *Global Ads Performance Dataset*
