Dokumen ini akan menguraikan komponen utama dan praktik terbaik yang diperlukan untuk menyesuaikan agen Anda.

***

# Panduan Implementasi Agent Reinforcement Fine-Tuning (Agent RFT)

## 1. Pendahuluan Mengenai Agent RFT

Agent RFT adalah cara untuk melatih agen Anda secara *end-to-end* pada tugas spesifik Anda untuk mencapai kinerja yang lebih baik. Agent RFT mengubah bobot model sesuai dengan sinyal pembelajaran (*learning signal*) yang Anda tentukan.

**Perbedaan Utama Agent RFT:**
1.  **Interaksi Alat:** Agent RFT memungkinkan agen untuk memanggil alat (*tools*) saat proses *rollout* dan eksplorasi selama pelatihan. Hal ini memungkinkan model belajar dari semua kemungkinan cara penggunaan alat Anda.
2.  **Reward Fleksibel:** Anda dapat menentukan sinyal *reward* arbitrer melalui *endpoint* khusus yang akan dipanggil oleh platform untuk melatih model.

**Manfaat Implementasi Agent RFT:**
*   Meningkatkan kinerja model penalaran (*reasoning models*).
*   Meningkatkan kemampuan agen untuk menggunakan alat dan mencapai jawaban akhir terbaik.
*   **Sangat efisien sampel (*sample efficient*)**, yang penting di domain di mana data pelatihan langka (misalnya, pembuatan *kernel* GPU).
*   Dapat menghasilkan model yang memiliki **latensi lebih rendah**.

## 2. Persiapan Tugas dan Data

Sebelum memulai RFT, pastikan dasar-dasarnya sudah kuat:

| Aspek | Detail dan Saran | Sumber |
| :--- | :--- | :--- |
| **Definisi Tugas** | Tugas harus ditentukan dengan baik (*well specified*) dan dibatasi (*constrained*). Tujuan yang jelas sangat penting. | |
| **Optimalisasi Awal** | Coba optimalkan kinerja agen tanpa *fine-tuning* terlebih dahulu: optimalkan *prompt*, sederhanakan tugas, tambahkan *guardrail* yang lebih baik, atau tambah/kurang alat. | |
| **Kualitas Data Set** | Bangun *data set* berkualitas tinggi di mana set pelatihan (*train*) dan evaluasi (*eval*) **sangat cocok dengan lalu lintas produksi** Anda. Agen seharusnya tidak "terkejut" ketika transisi dari *fine-tuning* ke waktu tayang (*showtime*). | |
| **Kinerja Baseline** | Pastikan model dasar (seperti GPT-5) memiliki kinerja *baseline* non-nol. Jika model tidak pernah benar setelah dieksplorasi berulang kali, model mungkin tidak akan belajar. | |
| **Variansi Data** | Data poin Anda harus memiliki cukup variansi sehingga selama eksplorasi, model tahu perbedaan antara perilaku yang baik dan kurang baik. Variansi ini memungkinkan model untuk "mendaki bukit" (*hill climb*) kinerjanya. | |

## 3. Implementasi Server Alat (Tool Server)

Alat memungkinkan agen Anda berinteraksi dengan dunia luar dan konteks bisnis Anda (misalnya, terminal, sistem penagihan, atau *codebase* internal).

### 3.1. Definisi Alat

Model OpenAI menggunakan alat dalam bentuk JSON yang harus menyertakan:
*   **`name`**: Nama alat (misalnya, `search`, `list`, `cat`).
*   **`URL`**: URL *endpoint* Anda yang akan dipanggil oleh platform.
*   **`headers`**: Termasuk token otentikasi (misalnya, O token) agar hanya Anda yang dapat mengakses *endpoint* tersebut.

### 3.2. Pengembangan Endpoint Alat

Anda perlu mengembangkan *endpoint* yang dapat diakses publik yang mengimplementasikan fungsionalitas alat Anda (misalnya, menggunakan *framework* seperti Fast API).

*   **Contoh Fungsi Alat (Berdasarkan Fin QA Demo):**
    *   **`search`:** Alat pencarian semantik (menggunakan *embedding* dan perhitungan kesamaan kosinus). Ini mirip dengan RAG (Retrieval Augmented Generation).
    *   **`list`:** Menjelajahi direktori dan jalur dokumen.
    *   **`cat`:** Mengembalikan dokumen berdasarkan jalur yang diberikan (seperti membuka dokumen).

### 3.3. Pertimbangan Latensi dan Efisiensi

*   **Batasi Panjang Output Alat:** Batasi panjang output panggilan alat (*tool calls*) karena output yang terlalu panjang akan memperlambat pelatihan dan juga dapat membingungkan model. Membatasi output yang tidak perlu juga menghemat biaya (*tokens*).
*   **ID Unik:** Untuk setiap *rollout* agen, platform memberikan **pengenal unik (*unique identifier*)** untuk semua panggilan alat dan jawaban akhir. Pastikan sistem *backend* Anda dapat mengenali panggilan alat berasal dari *rollout* yang sama, yang penting untuk manajemen status.

## 4. Desain Grader dan Sinyal Reward

Grader adalah inti dari Agent RFT karena menentukan kebijakan yang akan dipelajari model.

### 4.1. Jenis Grader

Platform menawarkan beberapa opsi, tetapi *Endpoint Grader* memberikan fleksibilitas tertinggi:
*   **Model Grader:** Digunakan untuk kesederhanaan, memungkinkan pemberian kredit parsial (misalnya, 0,5 jika jawaban hampir benar).
*   **String Grader:** Sangat rapuh (*brittle*); menghukum model untuk kesalahan format kecil dan tidak disarankan.
*   **Endpoint Grader:** **Platform akan memanggil *endpoint* publik Anda** sehingga Anda dapat menentukan sinyal *reward* khusus yang sesuai dengan domain Anda. Ini memungkinkan Anda untuk memasukkan kriteria dan rubrik Anda sendiri.

### 4.2. Prinsip Desain Reward

*   **Sinyal yang Konsisten:** Sinyal yang Anda berikan kepada model harus konsisten. Jangan katakan jawaban A itu bagus di satu contoh tetapi tidak di contoh berikutnya, karena ini akan membingungkan model.
*   **Reward Parsial (Gradient):** Sangat disarankan untuk memberikan kredit parsial, bukan hanya penilaian biner (benar/salah).
    *   *Mengapa:* Reward parsial memungkinkan model tahu bahwa ia bergerak ke arah yang benar, bahkan jika jawaban akhirnya salah. Ini membantu model untuk *hill climb* secara bertahap.
    *   *Contoh:* Anda dapat memberikan *reward* karena membaca file yang benar, meskipun penalaran selanjutnya salah.
*   **Sulit Direkayasa (*Hard to Game*):** Pastikan *grader* Anda "kedap air" (*watertight*). Model sangat cerdas dan dapat menemukan cara untuk mengeksploitasi celah (*edge case*) dalam *grader* Anda untuk memaksimalkan *reward* (dikenal sebagai *reward hacking*).

### 4.3. Konteks Grading

Karena Anda menerima ID unik untuk setiap panggilan alat, ketika *grader* dipanggil dengan jawaban akhir, Anda dapat melampirkan **semua konteks dari agen (termasuk urutan panggilan alat)** ke jawaban akhir untuk penilaian yang sangat holistik.

## 5. Konfigurasi Pelatihan RFT

Ketika menjalankan pelatihan RFT melalui API, beberapa *hyperparameter* penting yang perlu dipertimbangkan:

| Parameter | Fungsi dan Dampak | Sumber |
| :--- | :--- | :--- |
| **Compute Multiplier** | **Mengontrol jumlah eksplorasi** yang dilakukan model. | |
| | Nilai yang lebih tinggi memberikan model lebih banyak peluang untuk menemukan jalur penalaran yang baik dan mendapatkan *reward* non-nol. | |
| | Namun, meningkatkan ini akan meningkatkan kebutuhan komputasi dan memerlukan **ketahanan *endpoint* Anda yang lebih tinggi** karena platform akan memanggilnya lebih sering. | |
| **Epochs** | Berapa kali model akan melalui setiap sampel dalam *data set* pelatihan. | |
| **Batch Size** | Jumlah sampel yang digunakan dalam setiap langkah pelatihan. | |

### Analisis Kinerja

Setelah pelatihan, Anda dapat menganalisis kurva untuk memahami perubahan kebijakan model:
*   **Kurva Reward:** Amati bagaimana skor validasi (*validation reward*) meningkat dari *baseline*.
*   **Tool Call per Rollout:** Perhatikan penurunan signifikan dalam jumlah panggilan alat (misalnya, dari 6,9 menjadi 4,2) yang mengindikasikan model menjadi lebih efisien dan lebih cepat. Penurunan ini berkorelasi dengan penurunan latensi.
*   **Token Penalaran (*Reasoning Tokens*):** Biasanya, penurunan *tool calls* dan *reasoning tokens* akan terlihat, menghasilkan latensi yang lebih rendah untuk pengguna akhir.

## 6. Pertimbangan Infrastruktur

Karena Agent RFT memanggil alat dan *grader* Anda melalui internet publik, infrastruktur yang kuat adalah kuncinya.

| Aspek Infrastruktur | Detail Teknis | Sumber |
| :--- | :--- | :--- |
| **Mirip Produksi** | Host alat Anda agar **sangat mirip dengan lingkungan produksi** Anda. Perbaikan apa pun selama pelatihan akan langsung diterjemahkan ke produk Anda. | |
| **Lingkungan Rollout Terisolasi** | Gunakan **VM atau *container* yang terisolasi** untuk setiap *rollout*. Ini sangat penting jika alat Anda (seperti *shell tool*) dapat melakukan tindakan destruktif, memastikan satu *rollout* tidak memengaruhi *rollout* lainnya. | |
| **Mengatasi Kebutuhan Burst (Burst Capacity)** | Pelatihan RL cenderung *bursty*. Di awal setiap *rollout*, platform dapat mengirimkan ratusan permintaan *rollout* baru sekaligus (misalnya, 500 permintaan). Infrastruktur Anda harus mampu menangani lonjakan kapasitas ini. | |
| **Pemantauan Kegagalan (Monitoring)** | Memantau kegagalan panggilan alat sangat penting. Jika VM gagal atau ada kesalahan infrastruktur, model menerima *reward* nol. Jika ini sering terjadi, pelatihan dapat runtuh karena model belajar dengan cara yang buruk meskipun tindakannya mungkin benar. | |