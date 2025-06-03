# Laporan Proyek Machine Learning – Salma Oktarina

## Project Overview
Salah satu permasalahan yang sering dijumpai oleh para pembaca buku adalah menentukan buku-buku yang akan mereka baca selanjutnya. Kesulitan pembaca buku dalam menentukan buku yang akan dibaca disebabkan oleh banyaknya jumlah buku dan beragamnya jumlah buku yang ada. Solusi untuk permasalahan yang dialami pembaca adalah dengan menerapkan sistem rekomendasi buku yang dapat memberikan rekomendasi buku kepada pembaca buku [1].
Sistem rekomendasi dapat digunakan untuk memprediksi barang tertentu yang disukai oleh pengguna atau untuk mengidentifikasi beberapa barang yang mungkin disukai oleh pengguna tertentu [2].
Dalam proyek ini, dibangun dua pendekatan sistem rekomendasi: Content-Based Filtering, yang merekomendasikan buku berdasarkan kemiripan atribut kontennya seperti judul dan penulis; dan Collaborative Filtering, yang memanfaatkan pola interaksi pengguna dengan item untuk menghasilkan rekomendasi. Dataset yang digunakan adalah Best Books of the Decade: 2020s dari Kaggle, yang berisi informasi tentang ribuan buku populer beserta rating dari pengguna.

# Referensi:
[1] Andrew Hans Ritdrix, Panji Wisnu Wirawan. Sistem Rekomendasi Buku MenggunakanMetode Item-Based Collaborative Filtering (hal. 24–31). urnal Masyarakat Informatika.
[2] Deshpande, M., & Karypis, G. (2004). Item Based Top-N Recommendation Algorithms. ACM Transactions on Information Systems (TOIS) Volume 22, 143-177. 

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
- Index: kode unik setiap buku.
- Book Name: judul dari buku.
- Author: penulis buku.
- Rating: rata-rata ulasan yang diberikan user (1-5).
- Number of Votes: total vote untuk buku.
- Score: jumlah score dari ulasan buku dan total vote buku.
- Tidak ada missing values (null) artinya data siap dianalisis lebih lanjut.
- Tidak adanya duplikasi data artinya data sudah sesuai.

- Dataset Users: 600.000 entri data
- userId: kode unik setiap user.
- bookIndex: kode unik setiap buku dari dataset buku.
- score: score atau ulasan yang diberikan user untuk setiap buku (1-5).
- Tidak adanya missing values (null) dan data siap dianalisis lebih lanjut.
- Terdapat duplikasi data sebanyak 175 entri data.

# Exploratory Data Analysis (EDA)
- Distribusi rating yang dibaerikan user pada setiap buku
![Screenshot (1556)](https://github.com/user-attachments/assets/a2017909-0050-41ee-b8b6-23bebc44528f)

# Data Preparation
Langkah-langkah yang dilakukan:
- Mengubah nama kolom Index pada books_df menjadi bookIndex agar dapat digabungkan dengan users_df.
- Menghapus 175 data duplikat dari users_df untuk memastikan setiap interaksi pengguna unik.
- Menyimpan hanya baris dengan rating eksplisit (bukan implicit feedback) ke dalam ratings_explicit_df.
- Menggabungkan ratings_explicit_df dan books_df berdasarkan bookIndex untuk membuat full_data_explicit, yang digunakan pada model Collaborative Filtering.
- Menggunakan TfidfVectorizer dari Scikit-learn dengan stop_words='english' dan ngram_range=(1,2) untuk mengubah kolom content menjadi vektor numerik.

```
# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
print("Menerapkan TF-IDF pada sample")
tfidf_matrix_sample = tfidf.fit_transform(books_cb_sample['content'])
print("Shape of TF-IDF matrix (sample):", tfidf_matrix_sample.shape)
```
- Menyaring pengguna yang memberikan kurang dari 3 ulasan, untuk mengurangi sparsity dan meningkatkan kualitas prediksi.
- Inisialisasi Reader, menentukan skala rating dari 1 sampai 5.
- Train-Test Split, membagi data ke dalam 80% data latih dan 20% data uji menggunakan train_test_split dari surprise.model_selection.
- Membuat kolom content dengan menggabungkan kolom Book Name dan Author untuk setiap baris di books_cb_sample.

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
- Fungsi Rekomendasi: Fungsi get_collaborative_filtering_recommendations(user_id, N=5) dibuat untuk menghasilkan top-N rekomendasi bagi pengguna tertentu. Fungsi ini:
- Mengidentifikasi semua buku dalam filtered_ratings_cf yang belum pernah dirating oleh user_id target.
- Menggunakan model SVD yang telah dilatih untuk memprediksi rating pengguna target terhadap buku-buku yang belum dirating tersebut.
- Mengurutkan buku-buku tersebut berdasarkan prediksi rating (estimasi) secara menurun.
- Mengembalikan N buku teratas beserta detailnya (ISBN, Judul, Penulis) dan prediksi ratingnya.
- Hasil (Top-N Recommendation): Contoh output untuk userId: 65674, N=5:
   bookIndex                                          Book Name  \ Author                    estimated_rating
0       1704                     Your Brain is Always Listening    Daniel G. Amen            3.945382
1       2241  If It Sounds Like a Quack...: A Journey to the...    Matthew Hongoltz-Hetling  3.745854
2       2275  A Queen of Ruin (Deliciously Dark Fairytales, #4)    K.F. Breene               3.703566
               
Kelebihan Pendekatan Collaborative Filtering (SVD):
- Mampu menemukan pola preferensi yang kompleks dan tersembunyi dari data interaksi pengguna-item.
- Tidak memerlukan pengetahuan domain tentang item yang direkomendasikan.

Kekurangan Pendekatan Collaborative Filtering (SVD):
- Mengalami masalah cold start: sulit memberikan rekomendasi untuk pengguna baru (yang belum memiliki riwayat rating) atau item baru (yang belum pernah dirating).
- Kinerja sangat dipengaruhi oleh sparsity data. Jika data rating sangat jarang, model akan kesulitan belajar pola yang signifikan.
- Kurang transparan; sulit menjelaskan mengapa suatu item direkomendasikan selain karena "pengguna serupa menyukainya".

# Evaluation
1. Content-Based Filtering
Evaluasi dilakukan dengan mengukur seberapa relevan rekomendasi yang dihasilkan oleh sistem berdasarkan input buku tertentu. Sistem ini hanya menggunakan informasi dari judul dan penulis buku, tanpa mempertimbangkan data interaksi pengguna.
Contoh hasil rekomendasi untuk buku "Glassheart":
| bookIndex | Book Name                                             | Author          | similarity\_score |
| --------- | ----------------------------------------------------- | --------------- | ----------------- |
| 2021      | Majesty (American Royals, #2)                         | Katharine McGee | 0.1364            |
| 1699      | Katharine Parr, the Sixth Wife (Six Tudor Queens, #6) | Alison Weir     | 0.1042            |
| 1         | The Invisible Life of Addie LaRue                     | Victoria Schwab | 0.0000            |
| 2         | The House in the Cerulean Sea                         | T.J. Klune      | 0.0000            |
| 3         | Project Hail Mary                                     | Andy Weir       | 0.0000            |
        
Metrik Evaluasi (Top-K = 5):
(Seluruh nilai di bawah telah dibulatkan hingga 4 angka desimal untuk konsistensi pelaporan)
- Precision@5: 0.1644
Artinya, sekitar 16,44% dari buku yang direkomendasikan oleh sistem terbukti relevan (misalnya memiliki penulis yang sama atau tema yang serupa).
Rumus:
$$
Precision@K = \frac{|\text{Rekomendasi yang relevan}|}{K}
$$
- Recall@5: 0.3509
Artinya, sistem berhasil menemukan sekitar 35,09% dari total buku relevan hanya dalam 5 rekomendasi. Ini menunjukkan cakupan yang cukup baik.
Rumus:
$$
Recall@K = \frac{|\text{Rekomendasi yang relevan}|}{|\text{Total item relevan}|}
$$
- NDCG@5 (Normalized Discounted Cumulative Gain): 0.3740
Menunjukkan bahwa item yang relevan cenderung muncul di posisi atas rekomendasi, meskipun belum optimal. NDCG berkisar antara 0 (buruk) hingga 1 (sempurna).
Rumus DCG@K:
$$
DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}
$$

Rumus NDCG@K:
$$
NDCG@K = \frac{DCG@K}{IDCG@K}
$$

Keterangan:
- \( rel_i \) adalah relevansi item pada posisi ke-i
- \( IDCG@K \) adalah nilai DCG ideal (jika semua item relevan ada di posisi teratas)

Kelebihan:
- Cukup baik dalam menjangkau item relevan (recall tinggi).
- Cocok untuk kasus di mana data pengguna terbatas atau item baru terus muncul.

Kekurangan:
- Nilai precision relatif rendah, menandakan masih banyak rekomendasi yang tidak relevan.
- Fitur konten yang digunakan (judul dan penulis saja) mungkin belum cukup kaya untuk menangkap makna semantik secara dalam.
- Potensi over-specialization dan kurang serendipitas dalam rekomendasi.

2. Collaborative Filtering
Evaluasi dilakukan dengan mengukur seberapa baik model memprediksi rating pengguna terhadap buku menggunakan data rating eksplisit. Dataset dibagi menjadi 80% data latih dan 20% data uji.

Metrik Evaluasi:
(Angka diambil langsung dari hasil notebook dan dibulatkan ke 4 desimal)
- RMSE (Root Mean Squared Error): 1.4591
Mengukur rata-rata kesalahan kuadrat dari prediksi rating. Semakin kecil nilainya, semakin baik. Dalam skala rating 1–5, nilai ini menunjukkan bahwa prediksi model menyimpang sekitar ±1.46 poin dari rating sebenarnya.
- MAE (Mean Absolute Error): 1.2594
Mengukur rata-rata kesalahan absolut dari prediksi. MAE biasanya lebih mudah diinterpretasikan daripada RMSE karena tidak menghukum error besar secara berlebihan.

Kelebihan:
- Dapat menangkap pola preferensi laten yang kompleks antar pengguna.
- Tidak memerlukan fitur dari buku (judul, genre, dll).

Kekurangan:
- Rentan terhadap masalah cold start (pengguna/item baru).
- Rentan terhadap data sparsity — performa akan menurun jika data rating jarang atau tidak seimbang.
- Kurang transparan karena sulit menjelaskan alasan suatu rekomendasi selain “pengguna serupa menyukainya”.

Sistem rekomendasi yang dibangun dari dataset ini berhasil memberikan rekomendasi buku yang relevan bagi pengguna berdasarkan judul, author, dan rating. Dengan pengembangan lebih lanjut seperti penggunaan matrix factorization atau deep learning, kualitas rekomendasi bisa lebih ditingkatkan lagi.
