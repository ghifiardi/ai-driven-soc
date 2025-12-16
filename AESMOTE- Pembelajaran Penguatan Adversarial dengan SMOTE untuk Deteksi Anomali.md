Rangkuman Pengarahan: AESMOTE - Kerangka Kerja Deteksi Anomali dengan Reinforcement Learning dan SMOTE

Ringkasan Eksekutif

Dokumen ini merangkum kerangka kerja inovatif bernama AESMOTE (Adversarial Reinforcement Learning with SMOTE) yang dirancang untuk mengatasi tantangan kritis dalam Sistem Deteksi Intrusi (IDS), yaitu masalah ketidakseimbangan kelas (class imbalance) pada dataset. Masalah ini secara signifikan melemahkan model machine learning tradisional, terutama dalam kemampuannya mendeteksi serangan siber yang langka namun berbahaya (kelas minoritas).

AESMOTE mengintegrasikan tiga konsep utama: Reinforcement Learning (RL) Adversarial, teknik over-sampling SMOTE, dan arsitektur Double Deep Q-Network (DDQN). Mekanisme intinya menggunakan dua agen—sebuah Agen Pengklasifikasi yang membuat prediksi dan sebuah Agen Lingkungan yang memilih data pelatihan—dengan sistem imbalan (reward) yang saling bertentangan. Strategi adversarial ini secara dinamis memaksa model untuk berlatih pada sampel yang paling sulit dan sering salah diklasifikasikan.

Untuk mengatasi kelangkaan data serangan, kerangka kerja ini menerapkan SMOTE (Synthetic Minority Over-sampling Technique) untuk menghasilkan sampel data sintetis yang cerdas bagi kelas minoritas, sehingga meningkatkan representasi mereka tanpa menyebabkan overfitting. Hasil eksperimen pada dataset tolok ukur NSL-KDD menunjukkan bahwa AESMOTE secara signifikan mengungguli model dasar AE-RL dan teknik sampling lainnya, dengan mencapai F1-Score puncak 0.8243 dan Akurasi lebih dari 0.82. Peningkatan paling dramatis terlihat pada deteksi kelas minoritas R2L, di mana tingkat true-positive melonjak dari 0.29 (pada AE-RL) menjadi 0.69 (pada AESMOTE), membuktikan efektivitasnya dalam memperkuat pertahanan terhadap ancaman yang jarang terjadi.

1. Permasalahan Utama: Ketidakseimbangan Kelas dalam Deteksi Intrusi

Sistem Deteksi Intrusi modern menghadapi dua tantangan fundamental yang menghambat efektivitasnya:

* Ketidakseimbangan Dataset yang Ekstrem: Dataset yang digunakan untuk melatih IDS sering kali sangat tidak seimbang. Lalu lintas jaringan normal (kelas mayoritas) mendominasi sebagian besar data, sementara berbagai jenis serangan (kelas minoritas) seperti U2R (User to Root) dan R2L (Remote to Local) hanya mencakup sebagian kecil. Sebagai contoh, pada dataset NSL-KDD, kelas NORMAL mencakup 53.45% data, sedangkan U2R hanya 0.04%.
* Kelemahan Metrik Evaluasi Tradisional: Akibat ketidakseimbangan ini, model machine learning dapat mencapai akurasi keseluruhan yang tinggi hanya dengan selalu memprediksi kelas mayoritas. Hal ini menciptakan rasa aman yang palsu karena model tersebut secara efektif gagal mendeteksi serangan minoritas yang kritis. Metrik seperti F1-Score, yang menyeimbangkan presisi dan recall, menjadi lebih relevan untuk mengukur kinerja nyata.
* Sifat Statis Model Konvensional: Banyak pendekatan pembelajaran terawasi dan tidak terawasi bersifat statis. Setelah dilatih, model-model ini kesulitan beradaptasi dengan ancaman baru atau perubahan signifikan dalam pola serangan tanpa proses pelatihan ulang yang mahal dan memakan waktu.

2. Kerangka Kerja AESMOTE: Sebuah Pendekatan Tiga Lapis

AESMOTE dirancang untuk mengatasi masalah-masalah di atas melalui kombinasi strategis antara pendekatan tingkat algoritma dan tingkat data.

2.1. Reinforcement Learning (RL) Adversarial

Inti dari kerangka kerja ini adalah mekanisme adversarial yang ditenagai oleh dua agen Reinforcement Learning dengan tujuan yang saling bertentangan.

* Agen Pengklasifikasi (Classifier Agent - CA): Agen ini bertanggung jawab untuk membuat prediksi akhir terhadap jenis serangan (NORMAL, DoS, PROBE, R2L, U2R). Agen ini menerima imbalan positif (+1) untuk setiap klasifikasi yang benar dan imbalan negatif (-1) untuk yang salah. Tujuannya adalah memaksimalkan akurasi prediksi.
* Agen Lingkungan (Environment Agent - EA): Agen ini tidak membuat prediksi serangan, melainkan bertugas memilih sampel data berikutnya yang akan digunakan untuk melatih Agen Pengklasifikasi. Secara krusial, agen ini menerima imbalan yang berlawanan dengan CA. Jika CA berhasil memprediksi sebuah sampel, EA menerima imbalan negatif, dan sebaliknya.
* Dinamika Adversarial: Sistem imbalan yang berlawanan ini menciptakan sebuah "persaingan". EA belajar untuk mengidentifikasi "kelemahan" CA dan secara konsisten menyodorkan sampel-sampel yang paling sulit atau paling sering salah diklasifikasi oleh CA. Hal ini memaksa CA untuk terus belajar dan meningkatkan kinerjanya pada kelas-kelas yang sulit, secara efektif menyeimbangkan fokus pelatihan secara dinamis.

2.2. Teknik Over-Sampling SMOTE

Untuk mengatasi masalah kelangkaan data pada kelas minoritas seperti R2L dan U2R, AESMOTE mengimplementasikan pendekatan tingkat data menggunakan SMOTE.

* Masalah Overfitting pada Duplikasi: Teknik sederhana seperti Random Over-Sampling (ROS) hanya menduplikasi sampel minoritas yang ada. Hal ini dapat dengan mudah menyebabkan overfitting, di mana model menjadi terlalu spesifik pada sampel yang diduplikasi dan gagal menggeneralisasi pada data baru.
* Generasi Sampel Sintetis SMOTE: Alih-alih menduplikasi, SMOTE menghasilkan sampel sintetis yang baru. Proses ini bekerja dengan cara:
  1. Memilih sebuah sampel dari kelas minoritas.
  2. Mengidentifikasi K-tetangga terdekatnya (K-Nearest Neighbors) yang juga berasal dari kelas minoritas yang sama.
  3. Membuat sampel sintetis baru pada titik acak di sepanjang garis yang menghubungkan sampel asli dengan tetangga-tetangganya.
* Manfaat: Pendekatan ini memperluas wilayah keputusan untuk kelas minoritas secara lebih cerdas, mengurangi risiko overfitting, dan membantu model untuk menggeneralisasi pola serangan minoritas dengan lebih baik.

2.3. Model dan Arsitektur Pembelajaran

Untuk memastikan stabilitas dan akurasi, kerangka kerja ini menggunakan komponen-komponen canggih.

* Double Deep Q-Network (DDQN): Model ini menggunakan DDQN, sebuah pengembangan dari Deep Q-Network (DQN) standar. DDQN mengatasi masalah umum berupa overestimation nilai-Q, di mana DQN cenderung terlalu optimis dalam memperkirakan imbalan di masa depan. Dengan menggunakan dua jaringan terpisah untuk pemilihan aksi dan evaluasi aksi, DDQN menghasilkan pembelajaran yang lebih stabil dan andal.
* Fungsi Kerugian Huber (Huber Loss): Digunakan untuk mengurangi sensitivitas model terhadap outlier atau perilaku yang meledak-ledak selama pelatihan, sehingga menciptakan proses regresi yang lebih kuat.

3. Analisis Kinerja dan Hasil Eksperimental

Evaluasi dilakukan pada dataset NSL-KDD, dengan F1-Score sebagai metrik utama karena kemampuannya menangani dataset yang tidak seimbang.

3.1. Kinerja AESMOTE

* Tren Kinerja: Kinerja AESMOTE menunjukkan tren peningkatan seiring dengan bertambahnya jumlah sampel sintetis yang dihasilkan oleh SMOTE. Puncak kinerja dengan F1-Score 0.8243 dicapai ketika jumlah sampel yang dihasilkan berada di antara 5,000 hingga 70,000.
* Peningkatan Kunci pada Kelas Minoritas: Bukti paling signifikan dari keberhasilan AESMOTE adalah peningkatan dramatis dalam deteksi kelas minoritas. Tingkat true-positive untuk kelas R2L meningkat dari 0.29 pada model AE-RL (tanpa SMOTE) menjadi 0.69 pada AESMOTE. Ini menunjukkan bahwa penambahan SMOTE secara efektif memungkinkan model untuk mempelajari dan mengenali pola serangan yang langka.
* Stabilitas Pelatihan: Eksperimen menunjukkan bahwa kinerja model menjadi stabil setelah sekitar 100 episode pelatihan, menunjukkan bahwa model mencapai konvergensi yang optimal pada titik tersebut.

3.2. Perbandingan dengan Metode Lain

AESMOTE diuji dan dibandingkan dengan model dasar (AE-RL) serta varian yang menggunakan teknik sampling lain. Hasilnya diringkas dalam tabel berikut.

Metode	F1-Score	Waktu Proses (detik)	Analisis Singkat
AESMOTE (dengan SMOTE)	0.8243	Lebih Lama	Kinerja tertinggi secara keseluruhan, unggul secara signifikan dalam mendeteksi kelas minoritas (R2L).
AE-RL (tanpa sampling)	0.7800	-	Merupakan garis dasar (baseline) yang menunjukkan kinerja dari mekanisme adversarial saja.
NearMiss2 (Under-sampling)	0.7856	84	Kinerja baik pada kelas mayoritas, tetapi lebih lemah pada kelas minoritas. Waktu proses sangat cepat.
NearMiss1 (Under-sampling)	0.7602	195	Sedikit lebih baik pada kelas minoritas daripada NearMiss2, tetapi dengan mengorbankan akurasi pada kelas mayoritas.
ROS (Random Over-sampling)	0.7083	-	Kinerja paling rendah. Menunjukkan bahwa duplikasi sampel sederhana tidak efektif dan menyebabkan overfitting.

4. Kesimpulan dan Implikasi

Kerangka kerja AESMOTE berhasil menunjukkan solusi yang efektif untuk dua tantangan utama dalam IDS: ketidakseimbangan kelas dan kebutuhan akan pembelajaran yang dinamis dan adaptif.

* Sinergi yang Kuat: Kombinasi strategi RL adversarial pada tingkat algoritma dan teknik over-sampling SMOTE pada tingkat data menciptakan sinergi yang kuat. Mekanisme adversarial menargetkan sampel yang sulit, sementara SMOTE memastikan ada cukup data representatif untuk dipelajari.
* Peningkatan Pertahanan Kritis: Peningkatan substansial dalam mendeteksi serangan langka seperti R2L membuktikan nilai praktis dari model ini. Dalam skenario keamanan siber, kegagalan mendeteksi satu serangan langka bisa jauh lebih merugikan daripada salah mengklasifikasikan banyak lalu lintas normal.
* Arah Masa Depan: Penelitian di masa depan berencana untuk memperluas kerangka kerja ini dengan memperkenalkan "tingkat kesulitan" data dan menggunakan beberapa Agen Lingkungan dalam strategi multi-adversarial. Tujuannya adalah untuk lebih presisi dalam mengidentifikasi dan memperbaiki kelemahan spesifik dari Agen Pengklasifikasi, yang berpotensi menghasilkan sistem deteksi yang lebih kuat dan tangguh.
