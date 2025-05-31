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
- Dataset Buku: 2.329 entri data
-- Index: kode unik setiap buku.
-- Book Name: judul dari buku.
-- Author: penulis buku.
-- Rating: rata-rata ulasan yang diberikan user (1-5).
-- Number of Votes: total vote untuk buku.
-- Score: jumlah score dari ulasan buku dan total vote buku.
-- Tidak ada missing values (null) artinya data siap dianalisis lebih lanjut.
-- Tidak adanya duplikasi data artinya data sudah sesuai.

- Dataset Users: 600.000 entri data
-- userId: kode unik setiap user.
-- bookIndex: kode unik setiap buku dari dataset buku.
-- score: score atau ulasan yang diberikan user untuk setiap buku (1-5).
-- Tidak adanya missing values (null) dan data siap dianalisis lebih lanjut.
-- Terdapat duplikasi data sebanyak 175 entri data.

# Exploratory Data Analysis (EDA)
- Distribusi rating yang dibaerikan user pada setiap buku
![Screenshot (1556)](https://github.com/user-attachments/assets/a2017909-0050-41ee-b8b6-23bebc44528f)

# Data Preparation
Langkah-langkah yang dilakukan:
- Mengubah nama kolom Index pada books_df menjadi bookIndex agar dapat digabungkan dengan users_df.
- Menghapus 175 data duplikat dari users_df untuk memastikan setiap interaksi pengguna unik.
- Menyimpan hanya baris dengan rating eksplisit (bukan implicit feedback) ke dalam ratings_explicit_df.
- Menggabungkan ratings_explicit_df dan books_df berdasarkan bookIndex untuk membuat full_data_explicit, yang digunakan pada model Collaborative Filtering.
- Menyaring pengguna yang memberikan kurang dari 3 ulasan, untuk mengurangi sparsity dan meningkatkan kualitas prediksi.
- Inisialisasi Reader, menentukan skala rating dari 1 sampai 5.
- Train-Test Split, membagi data ke dalam 80% data latih dan 20% data uji menggunakan train_test_split dari surprise.model_selection.
- Membuat kolom content dengan menggabungkan kolom Book Name dan Author untuk setiap baris di books_cb_sample.
- Menggunakan TfidfVectorizer dari Scikit-learn dengan stop_words='english' dan ngram_range=(1,2) untuk mengubah kolom content menjadi vektor numerik.

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
- Evaluasi metode CBF dilakukan dengan melihat hasil rekomendasi berdasarkan input buku. Misalnya, saat diminta rekomendasi berdasarkan buku "Glassheart", sistem menghasilkan rekomendasi seperti:
      bookIndex                                          Book Name  \ Author              similarity_score
2020       2021                      Majesty (American Royals, #2)    Katharine McGee     0.136359
1698       1699  Katharine Parr, the Sixth Wife (Six Tudor Quee...    Alison Weir         0.104207
0             1                  The Invisible Life of Addie LaRue    Victoria Schwab     0.000000
1             2  The House in the Cerulean Sea (Cerulean Chroni...    T.J. Klune          0.000000
2             3                                  Project Hail Mary    Andy Weir           0.000000
        
- Catatan: Nilai skor kemiripan yang relatif rendah (di bawah 0.2) menunjukkan bahwa fitur teks judul dan penulis yang digunakan dalam representasi TF-IDF mungkin kurang mencerminkan kesamaan semantik secara kuat. Hal ini menunjukkan perlunya pengayaan fitur, misalnya dengan menggunakan deskripsi buku atau genre.

Evaluasi untuk Content-Based Filtering (K=5):
- Precision@K      : 0.16436238729068267
-- Artinya: Sistem berhasil menemukan sekitar 35% dari total buku relevan (berdasarkan asumsi pengarang sama) hanya dalam 5 rekomendasi.
-- Ini cukup bagus untuk model CBF karena sistem hanya menggunakan informasi konten (judul + penulis), bukan data pengguna.
- Recall@K         : 0.3509025610141969
-- Artinya: Dari 5 rekomendasi yang diberikan, hanya sekitar 16% yang benar-benar relevan.
-- Ini menunjukkan bahwa sistem masih banyak memberikan rekomendasi yang tidak dianggap relevan (false positives).
- NDCG@K           : 0.37398024903392013
-- Menunjukkan bahwa item relevan kadang muncul di urutan atas, tetapi belum konsisten.
-- Jika mendekati 1, berarti sistem benar-benar menempatkan item relevan di posisi atas, tapi 0.37 menunjukkan masih kurang optimal.

Kelebihan:
- Recall cukup tinggi → sistem bisa menemukan item relevan dalam top-5.
- CBF mudah diterapkan tanpa data pengguna.

Kekurangan:
- Precision rendah → sistem memberikan banyak rekomendasi yang tidak relevan.
- NDCG sedang → urutan rekomendasi masih bisa diperbaiki.

2. Collaborative Filtering
- Model SVD dievaluasi menggunakan dua metrik utama:
-- RMSE (Root Mean Squared Error): 1.4531
-- MAE (Mean Absolute Error): 1.2536
- Dalam konteks skala rating 1–5, nilai ini menunjukkan prediksi model menyimpang ±1.25 poin dari rating sebenarnya. Ini dapat dianggap cukup tinggi, yang menunjukkan potensi perbaikan dengan teknik tambahan seperti tuning hyperparameter atau matrix factorization lanjutan.

Kelebihan:
- Mampu menangkap pola kompleks antar pengguna dan item.
- Tidak memerlukan fitur konten dari buku.

Kekurangan:
- Tidak bisa menangani item atau pengguna baru (cold start).
- Kurang transparan dalam menjelaskan alasan rekomendasi.

Sistem rekomendasi yang dibangun dari dataset ini berhasil memberikan rekomendasi buku yang relevan bagi pengguna berdasarkan judul, author, dan rating. Dengan pengembangan lebih lanjut seperti penggunaan matrix factorization atau deep learning, kualitas rekomendasi bisa lebih ditingkatkan lagi.
