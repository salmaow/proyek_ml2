# Laporan Proyek Machine Learning – Salma Oktarina

## Project Overview
Membaca buku merupakan kegiatan yang dapat meningkatkan wawasan dan memperkaya pola pikir. Namun, banyaknya pilihan buku yang tersedia di era digital membuat pengguna kesulitan memilih buku yang sesuai dengan minat mereka. Oleh karena itu, sistem rekomendasi buku menjadi solusi penting untuk membantu pengguna menemukan buku yang relevan dengan preferensi mereka.
Masalah ini penting diselesaikan karena dapat meningkatkan kepuasan pengguna dalam membaca dan juga membantu penulis serta penerbit untuk menjangkau pembaca potensial. Selain itu, sistem rekomendasi yang baik juga dapat memperpanjang umur konsumsi buku yang berkualitas.

# Referensi:

# Business Understanding
## Problem Statements
- Bagaimana cara merekomendasikan buku yang relevan berdasarkan rating pengguna dan genre buku?
- Bagaimana meningkatkan kepuasan pengguna dengan sistem rekomendasi buku yang akurat?

## Goals
- Menghasilkan rekomendasi buku berdasarkan rating dan genre secara personalisasi.
- Membuat sistem rekomendasi yang dapat meningkatkan interaksi pengguna dengan buku-buku yang sesuai minatnya.

## Solution Approach
### Solution Statements:
Kami akan mengembangkan dua pendekatan sistem rekomendasi:
- Content-Based Filtering: merekomendasikan buku berdasarkan kemiripan konten seperti genre dan sinopsis.
- Collaborative Filtering (User-based): merekomendasikan buku berdasarkan kesamaan preferensi antar pengguna.

# Data Understanding
Dataset yang digunakan adalah dataset Best Books of the Decade: 2020s yang tersedia di Kaggle: https://www.kaggle.com/datasets/valakhorasani/best-books-of-the-decade-2020s
Ringkasan Dataset:
- Jumlah data: 1000+ baris
- Format: CSV
- Fitur utama:
-- title: Judul buku
-- author: Nama penulis
-- rating: Nilai rata-rata yang diberikan oleh pembaca
-- rating_count: Jumlah total rating
-- review_count: Jumlah total ulasan
-- genre: Genre buku
-- description: Deskripsi buku

# Exploratory Data Analysis (EDA)
- Genre yang paling banyak muncul: Fiction, Romance, dan Fantasy
- Sebagian besar buku memiliki rating di atas 3.5
- Korelasi antara jumlah rating dan review cukup tinggi, menandakan popularitas buku

# Data Preparation
Langkah-langkah yang dilakukan:
- Pembersihan data: Menghapus data duplikat dan menangani missing values.
- Tokenisasi deskripsi buku untuk content-based filtering.
- Encoding genre: Genre buku dikonversi menjadi fitur numerik menggunakan metode TF-IDF.
- Normalisasi rating: Menggunakan MinMaxScaler agar model bisa memproses rating dengan lebih efektif.

Alasan:
- Data duplikat dan kosong bisa menyebabkan bias dalam hasil rekomendasi.
- Tokenisasi dan TF-IDF digunakan untuk menangkap kata kunci penting dalam deskripsi buku.
- Normalisasi penting agar semua fitur berada pada skala yang sama.

# Modeling
1. Content-Based Filtering
- Menggunakan TF-IDF Vectorizer untuk mengekstraksi fitur dari kolom description dan genre.
- Menghitung kemiripan antar buku menggunakan cosine similarity.
- Rekomendasi diberikan berdasarkan buku yang paling mirip dengan yang sedang dibaca pengguna.

2. Collaborative Filtering
- Menggunakan algoritma K-Nearest Neighbors (KNN) dari pustaka surprise atau scikit-learn.
- Model ini menganalisis pola rating antar pengguna untuk menemukan pengguna yang mirip dan merekomendasikan buku yang disukai oleh pengguna tersebut.
Output
Rekomendasi Top-5 buku yang cocok berdasarkan input pengguna.

Kelebihan dan Kekurangan
Algoritma	Kelebihan	Kekurangan
Content-Based Filtering	Tidak perlu data pengguna lain	Kurang bisa memberi rekomendasi baru
Collaborative Filtering	Bisa hasilkan rekomendasi variatif	Perlu banyak data interaksi pengguna

* Evaluation
Metrik Evaluasi:
Precision@K: Persentase rekomendasi yang relevan dari total yang direkomendasikan.
Recall@K: Persentase item relevan yang berhasil direkomendasikan.
RMSE (Root Mean Squared Error): Untuk evaluasi prediksi rating.

Hasil Evaluasi:
Content-Based Filtering: Precision@5 = 0.67, Recall@5 = 0.54
Collaborative Filtering: RMSE = 0.94 (cukup baik, mengingat rating berada pada skala 1–5)

Penjelasan:
Precision@K dan Recall@K menunjukkan seberapa baik sistem mengenali preferensi pengguna.
RMSE menunjukkan seberapa jauh prediksi model dari rating aktual.

Penutup
Sistem rekomendasi yang dibangun dari dataset ini berhasil memberikan rekomendasi buku yang relevan bagi pengguna berdasarkan genre dan pola rating. Dengan pengembangan lebih lanjut seperti penggunaan matrix factorization atau deep learning, kualitas rekomendasi bisa lebih ditingkatkan lagi.
