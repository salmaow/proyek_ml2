# Laporan Proyek Machine Learning – Salma Oktarina

## Project Overview
Membaca buku merupakan kegiatan yang dapat meningkatkan wawasan dan memperkaya pola pikir. Namun, banyaknya pilihan buku yang tersedia di era digital membuat pengguna kesulitan memilih buku yang sesuai dengan minat mereka. Oleh karena itu, sistem rekomendasi buku menjadi solusi penting untuk membantu pengguna menemukan buku yang relevan dengan preferensi mereka.
Masalah ini penting diselesaikan karena dapat meningkatkan kepuasan pengguna dalam membaca dan juga membantu penulis serta penerbit untuk menjangkau pembaca potensial. Selain itu, sistem rekomendasi yang baik juga dapat memperpanjang umur konsumsi buku yang berkualitas.

# Referensi:
Andrew Hans Ritdrix, Panji Wisnu Wirawan. Sistem Rekomendasi Buku MenggunakanMetode Item-Based Collaborative Filtering (hal. 24–31). urnal Masyarakat Informatika.

# Business Understanding
## Problem Statements
- Bagaimana cara merekomendasikan buku yang relevan berdasarkan rating pengguna dan genre buku?
- Bagaimana meningkatkan kepuasan pengguna dengan sistem rekomendasi buku yang akurat?

## Goals
- Menghasilkan rekomendasi buku berdasarkan rating dan genre secara personalisasi.
- Membuat sistem rekomendasi yang dapat meningkatkan interaksi pengguna dengan buku-buku yang sesuai minatnya.

## Solution Approach
### Solution Statements:
- Content-Based Filtering:
Cara Kerja: Metode ini merekomendasikan buku dengan menganalisis kesamaan konten antar buku. Atribut seperti judul, nama penulis, dan penerbit diekstraksi untuk membentuk representasi setiap buku.
Teknik: Algoritma TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengonversi informasi tekstual menjadi bentuk numerik berupa vektor fitur. Kemiripan antar buku dihitung menggunakan metrik seperti cosine similarity atau linear kernel. Buku dengan tingkat kemiripan tertinggi terhadap buku acuan akan dijadikan rekomendasi.

- Collaborative Filtering (Model-Based - SVD):
Cara Kerja: Metode ini menyarankan buku berdasarkan pola interaksi pengguna, terutama dari data rating. Prinsip utamanya adalah: jika dua pengguna memiliki kebiasaan memberikan rating yang serupa, maka buku yang disukai oleh salah satu kemungkinan juga akan disukai oleh yang lain.
Teknik: Digunakan algoritma Singular Value Decomposition (SVD), yaitu teknik untuk melakukan dekomposisi matriks. SVD membantu menemukan faktor-faktor laten atau preferensi tersembunyi di balik data rating. Model ini kemudian mampu memperkirakan rating pengguna terhadap buku yang belum mereka nilai, dan merekomendasikan buku dengan prediksi rating tertinggi.

# Data Understanding
Dataset yang digunakan adalah dataset Best Books of the Decade: 2020s yang tersedia di Kaggle: https://www.kaggle.com/datasets/valakhorasani/best-books-of-the-decade-2020s
Variabel Pada Dataset:
- Dataset Buku: 2.393 entri data
-- Index: kode unik setiap buku.
-- Book Name: judul dari buku.
-- Author: penulis buku.
-- Rating: rata-rata ulasan yang diberikan user (1-5).
-- Number of Votes: total vote untuk buku.
-- Score: jumlah score dari ulasan buku dan total vote buku.

- Dataset Users: 600.000 entri data
-- userId: kode unik setiap user.
-- bookIndex: kode unik setiap buku dari dataset buku.
-- score: score atau ulasan yang diberikan user untuk setiap buku (1-5).

# Exploratory Data Analysis (EDA)
- Distribusi rating yang dibaerikan user pada setiap buku
![Screenshot (1556)](https://github.com/user-attachments/assets/a2017909-0050-41ee-b8b6-23bebc44528f)

# Data Preparation
Langkah-langkah yang dilakukan:
- Pembersihan data: Menghapus data duplikat dan menangani missing values, jika tidak ditangani dapat menyebabkan bias dalam ahsi rekomendasi.
- Mengubah kolom Index pada books_df menjadi bookIndex agar selaras dengan users_df.
- Pembuatan kolom content untuk Content-Based Filtering: Dibuat dengan menggabungkan informasi dari kolom Book Name dan Author menjadi satu string teks tunggal untuk setiap buku. Ini dilakukan pada sampel data books_cb_sample yang digunakan untuk content-based filtering.
- Filter rating eksplisit: Baris data dari ratings_df disimpan dalam dataframe baru ratings_explicit_df.
- Filtering untuk Mengurangi Sparsity: Dilakukan dua tahap pemfilteran pada full_data_explicit untuk mengurangi masalah sparsity data, hanya pengguna yang telah memberikan minimal 5 rating eksplisit yang dipertahankan.
```
# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
print("Menerapkan TF-IDF pada sample")
tfidf_matrix_sample = tfidf.fit_transform(books_cb_sample['content'])
print("Shape of TF-IDF matrix (sample):", tfidf_matrix_sample.shape)
```

# Modeling
1. Content-Based Filtering
- Pendekatan ini merekomendasikan buku berdasarkan kemiripan kontennya (judul dan penulis).
- Pembuatan Fitur Konten: Sebuah kolom content dibuat dengan menggabungkan teks dari Book Name dan Author untuk setiap buku dalam sampel.
- TF-IDF Vectorization: TfidfVectorizer dari sklearn digunakan untuk mengubah teks pada kolom content menjadi matriks representasi numerik (TF-IDF).
- Perhitungan Kemiripan: cosine_similarity dari sklearn.metrics.pairwise digunakan untuk menghitung skor kemiripan antar semua pasangan buku dalam sampel. cosine similarity menghasilkan hasil yang sama dengan linear_kernel ketika inputnya adalah vektor TF-IDF (karena TF-IDF biasanya sudah ternormalisasi atau normalisasi L2 adalah bagian dari prosesnya).
```
cosine_sim_cb_sample = cosine_similarity(tfidf_matrix_sample, tfidf_matrix_sample)
print("Shape of Similarity matrix (sample):", cosine_sim_cb_sample.shape)
```
- Fungsi Rekomendasi: Fungsi get_content_based_recommendations_sample(bookindex, N=10) dibuat untuk menghasilkan top-N rekomendasi. Fungsi ini mengambil index buku sebagai input, mencari buku tersebut dalam sampel, mendapatkan skor kemiripannya dengan semua buku lain, mengurutkannya, dan mengembalikan N buku teratas yang paling mirip beserta skor kesamaannya.
- Hasil (Top-N Recommendation): Contoh output untuk buku 'The Invisible Life of Addie LaRue' dari sampel, N=5:
      bookIndex                                          Book Name  \ Author             similarity_score  
258         259                                            Gallant    Victoria Schwab    0.356272    
1294       1295                Bridge of Souls (Cassidy Blake, #3)    Victoria Schwab    0.237166 
592         593  The Fragile Threads of Power (Threads of Power...    Victoria Schwab    0.181362
2248       2249                                     Invisible Girl    Lisa Jewell        0.108246
1751       1752                                               Obit    Victoria Chang     0.104261 

Kelebihan Pendekatan Content-Based:
- Dapat merekomendasikan item baru yang belum memiliki interaksi pengguna (mengatasi sebagian masalah cold start untuk item).
- Rekomendasi bersifat transparan dan dapat dijelaskan berdasarkan fitur item (misalnya, "buku ini direkomendasikan karena memiliki penulis yang sama").
- Tidak bergantung pada data pengguna lain.

Kekurangan Pendekatan Content-Based:
- Kualitas rekomendasi sangat bergantung pada kualitas fitur yang diekstraksi. Jika fitur tidak representatif, rekomendasi akan buruk.
- Cenderung menghasilkan rekomendasi yang serendipity-nya rendah (kurang mengejutkan) karena hanya merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna.
- Bisa terjadi over-specialization, di mana pengguna terjebak dalam gelembung filter dan tidak terekspos pada item yang beragam.

2. Collaborative Filtering
- Menggunakan algoritma Singular Value Decomposition (SVD) dari pustaka surprise atau scikit-learn.
- Model ini menganalisis pola rating antar pengguna untuk menemukan pengguna yang mirip dan merekomendasikan buku yang disukai oleh pengguna tersebut.
- Persiapan Data untuk Library Surprise: Data rating eksplisit yang telah difilter (filtered_ratings_cf) digunakan. Reader dari library surprise diinisialisasi untuk skala rating 1 hingga 5.
- Pembagian Data: Dataset dibagi menjadi set pelatihan (80%) dan set pengujian (20%) menggunakan fungsi train_test_split dari surprise.model_selection.
- Model SVD: Algoritma SVD (Singular Value Decomposition) dari library surprise dipilih sebagai model. SVD adalah teknik faktorisasi matriks yang populer untuk collaborative filtering. Parameter yang digunakan dalam notebook adalah: n_factors=50 (jumlah faktor laten), n_epochs=20 (jumlah iterasi pelatihan), lr_all=0.005 (learning rate), dan reg_all=0.02 (faktor regularisasi).
- Evaluasi: Setelah pelatihan, model dievaluasi pada set pengujian untuk mengukur seberapa baik ia memprediksi rating yang sebenarnya. Metrik yang digunakan adalah RMSE dan MAE.
- Fungsi Rekomendasi: Fungsi get_collaborative_filtering_recommendations(user_id, N=10) dibuat untuk menghasilkan top-N rekomendasi bagi pengguna tertentu. Fungsi ini:
- Mengidentifikasi semua buku dalam filtered_ratings_cf yang belum pernah dirating oleh user_id target.
- Menggunakan model SVD yang telah dilatih untuk memprediksi rating pengguna target terhadap buku-buku yang belum dirating tersebut.
- Mengurutkan buku-buku tersebut berdasarkan prediksi rating (estimasi) secara menurun.
- Mengembalikan N buku teratas beserta detailnya (ISBN, Judul, Penulis) dan prediksi ratingnya.
- Hasil (Top-N Recommendation): Contoh output untuk userId: 65674, N=5:
   bookIndex                                          Book Name  \ Author                  estimated_rating 
0        502                                             Hester    Laurie Lico Albanese    4.006287
1        401          The Golden Enclaves (The Scholomance, #3)    Naomi Novik             3.929986
2        713                                   The Holiday Swap    Maggie Knox             3.899197
3       2293                      The Ruthless (Queen Crow, #2)    J. Bree                 3.890364
4       1173  The Jakarta Method: Washington's Anticommunist...    Vincent Bevins          3.884843

Kelebihan Pendekatan Collaborative Filtering (SVD):
- Mampu menemukan pola preferensi yang kompleks dan tersembunyi dari data interaksi pengguna-item.
- Tidak memerlukan pengetahuan domain tentang item yang direkomendasikan.

Kekurangan Pendekatan Collaborative Filtering (SVD):
- Mengalami masalah cold start: sulit memberikan rekomendasi untuk pengguna baru (yang belum memiliki riwayat rating) atau item baru (yang belum pernah dirating).
- Kinerja sangat dipengaruhi oleh sparsity data. Jika data rating sangat jarang, model akan kesulitan belajar pola yang signifikan.
- Kurang transparan; sulit menjelaskan mengapa suatu item direkomendasikan selain karena "pengguna serupa menyukainya".

# Evaluation
1. Content-Based Filtering
- Menyajikan evaluasi dari sistem content-based filtering yang sederhana, di mana sistem memberikan rekomendasi buku berdasarkan kesamaan konten dengan buku input "Glassheart".
- Hasilnya menampilkan daftar buku yang direkomendasikan beserta skor kesamaan mereka. 
- Beberapa judul yang direkomendasikan seperti "Majesty (American Royals, #2)" dan "The Invisible Life of Addie LaRue" terlihat terkait skor kesamaan yang relatif rendah (di bawah 0.2) untuk sebagian besar rekomendasi menunjukkan bahwa metode analisis konten atau perhitungan kesamaan yang digunakan mungkin perlu ditingkatkan untuk menghasilkan rekomendasi yang lebih relevan atau serupa secara kuat.

2. Collaborative Filtering
- Metrik evaluasi untuk model Collaborative Filtering menggunakan metode SVD, dengan nilai RMSE sebesar 1.4531 dan MAE sebesar 1.2536.
- Nilai-nilai ini mengindikasikan performa model dalam memprediksi rating pengguna; secara rata-rata, prediksi rating model SVD menyimpang sekitar 1.4531 poin dari rating sebenarnya (dengan penalti lebih besar untuk kesalahan besar) dan memiliki rata-rata selisih absolut 1.2536 poin dari rating sebenarnya.
- Dalam konteks skala rating 1-5, semakin rendah nilai RMSE dan MAE, semakin baik kemampuan model dalam memprediksi rating, sehingga nilai-nilai ini memberikan gambaran kuantitatif tentang akurasi rekomendasi yang dihasilkan oleh model SVD ini.

Sistem rekomendasi yang dibangun dari dataset ini berhasil memberikan rekomendasi buku yang relevan bagi pengguna berdasarkan judul, author, dan rating. Dengan pengembangan lebih lanjut seperti penggunaan matrix factorization atau deep learning, kualitas rekomendasi bisa lebih ditingkatkan lagi.
