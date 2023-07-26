# Laporan Proyek Machine Learning - Alexander Gosal
## Domain Proyek
Stroke merupakan salah satu kondisi medis yang serius dan dapat mengancam jiwa, terjadi ketika suplai darah ke otak terganggu, baik karena adanya sumbatan pada pembuluh darah atau pecahnya pembuluh darah. Menurut *World Health Organization* (WHO), stroke adalah penyebab kematian ke-2 secara global, bertanggung jawab atas sekitar 11% dari total kematian. Menerapkan teknologi *machine learning* dalam bidang medis, seperti mengidentifikasi potensi stroke, dapat memiliki dampak yang signifikan dalam upaya pencegahan dan pengobatan lebih dini, yang pada gilirannya dapat meningkatkan tingkat kesembuhan dan kualitas hidup pasien. *Stroke Prediction Dataset* digunakan dalam proyek ini untuk memberikan informasi penting tentang pasien, termasuk faktor risiko seperti usia, jenis kelamin, tekanan darah, status perokok, status perkawinan, serta informasi medis lainnya. Data-data ini akan diolah dan dianalisis oleh model *machine learning* yang dibangun untuk menemukan pola-pola yang dapat membedakan antara pasien yang berpotensi mengalami stroke dengan yang tidak.

Seiring dengan berkembangnya teknologi, peran kecerdasan buatan atau yang dikenal dengan *Artificial Intelligence* (AI) di bidang kedokteran mulai berkembang sangat luas. Dalam pengaplikasiannya kita bisa memanfaatkan teknologi tersebut untuk mengadopsi proses dan cara berpikir manusia melalui sistem pakar. Sistem pakar ini merupakan program komputer yang dapat meniru proses pemikiran dan pengetahuan pakar untuk menyelesaikan suatu masalah. Dalam dunia kedokteran aplikasi sistem pakar tersebut bisa dimanfaatkan dalam membantu mendiagnosis penyakit-penyakit tertentu, termasuk dalam diagnosis jenis penyakit stroke. Sistem pakar banyak membantu penggunanya dalam memperoleh suatu keputusan akan penyakit serta memberikan solusi baik berupa himbauan, penatalaksanaan, dan juga terapi pengobatan yang sesuai. Masih kurangnya kesadaran serta pemahaman masyarakat terhadap gejala awal dan penanganan lebih dini terhadap penyakit ditambah keterbatasan jumlah tenaga dan peralatan medis mengakibatkan implementasi teknologi ini dipandang sebagai salah satu alternatif yang memudahkan tenaga medis dalam memprediksi prognosis dan melakukan diagnosis secara lebih cepat.

Peranan Artificial Intelligence (AI) dalam bidang kedokteran ini memiliki dampak yang sangat besar. Selain memudahkan dalam verifikasi data dan identifikasi gejala dari pasien secara cepat, penggunaan teknologi ini pun membantu mempermudah tenaga medis dalam melakukan prognosis, diagnosis, penentuan terapi, dan pengobatan yang disesuaikan dengan jenis penyakit pasien. Meskipun demikian, penggunaan teknologi Artificial Intelligence (AI) dalam bidang kedokteran saat ini masih sangat jarang ditemukan. Adanya keterbatasan waktu dari tenaga medis dan kurangnya pemahaman mengenai alur dan proses kerja dari teknologi AI ini mengakibatkan pengembangan aplikasinya masih sangat terbatas. Untuk itu diperlukan adanya kolaborasi dan sinergi yang baik antara tenaga medis dengan tenaga ahli AI dalam mewujudkan penerapan aplikasi tersebut secara nyata dalam bidang kedokteran sehingga pengoperasiannya pun mudah dilakukan bagi para tenaga medis.

Tujuan dari proyek ini adalah untuk menciptakan model yang akurat dan dapat diandalkan dalam mengidentifikasi potensi stroke pada pasien. Dengan memiliki model yang baik, pihak medis dapat menggunakan alat ini sebagai salah satu alat bantu dalam diagnosa awal, memungkinkan intervensi lebih cepat dan penanganan yang lebih tepat. Proyek ini juga bertujuan untuk mengeksplorasi dan menerapkan *deep learning* sebagai alternatif yang lebih kuat dalam mengatasi masalah klasifikasi stroke. Dengan menerapkan metode *deep learning*, diharapkan model dapat mengidentifikasi pola yang lebih kompleks dan mendalam pada data medis stroke.

Namun, perlu diingat bahwa meskipun *machine learning* dapat memberikan kontribusi yang signifikan dalam bidang medis, hasil dari model ini sebaiknya digunakan hanya sebagai alat bantu dan tidak menggantikan pendapat atau evaluasi dari profesional medis yang berkualifikasi. Kesehatan merupakan hal yang sangat sensitif, sehingga keputusan akhir mengenai diagnosis dan pengobatan harus selalu berdasarkan penilaian dari dokter yang berpengalaman.

## Business Understanding
#### Problem Statements  
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut :
1. Bagaimana cara menganalisa dataset *stroke prediction* ?
2. Bagaimana cara mengolah dan memproses data agar dapat digunakan untuk *training* ?
3. Apakah penerapan metode *deep learning* dapat memberikan hasil yang lebih akurat dibandingkan dengan *traditional machine learning* ?
#### Goals 
Tujuan proyek ini dibuat adalah sebagai berikut :
1. Melakukan analisa terhadap data dan mengolah data agar dapat dilakukan proses *training*.
2. Mengembangkan model klasifikasi stroke yang akurat.
3. Mengeksplorasi penerapan deep learning dalam meningkatkan akurasi model klasifikasi stroke.

## Data Understanding
[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) digunakan untuk memprediksi apakah pasien kemungkinan terkena stroke berdasarkan parameter input seperti jenis kelamin, usia, berbagai penyakit, dan status merokok. Setiap baris dalam data memberikan informasi yang relevan tentang pasien.
#### Variabel-variabel pada *Stroke Prediction Dataset* adalah sebagai berikut:
1) id: pengenal unik
2) gender: "Pria", "Wanita" atau "Lainnya"
3) age: usia pasien
4) hypertension: 0 jika pasien tidak hipertensi, 1 jika pasien hipertensi
5) heart_disease: 0 jika pasien tidak memiliki penyakit jantung, 1 jika pasien memiliki penyakit jantung
6) ever_married: "tidak" atau "ya"
7) work_type: "anak-anak", "Pemerintahan", "Tidak pernah_bekerja", "Swasta" atau "Wiraswasta"
8) Residence_type: "Pedesaan" atau "Perkotaan"
9) avg_glucose_level: kadar glukosa rata-rata dalam darah
10) bmi: indeks massa tubuh
11) smoking_status: "sebelumnya merokok", "tidak pernah merokok", "merokok" atau "Tidak diketahui"*
12) stroke: 1 jika pasien mengalami stroke atau 0 jika tidak
*Catatan: "Tidak diketahui" dalam status_merokok berarti informasi tidak tersedia untuk pasien ini
#### Contoh dataset :
| id    | gender | age  | hypertension | heart_disease | ever_married | work_type     | Residence_type | avg_glucose_level | bmi  | smoking_status  | stroke |
|-------|--------|------|--------------|---------------|--------------|---------------|----------------|-------------------|------|-----------------|--------|
| 9046  | Male   | 67.0 |       0      |       1       |      Yes     | Private       |      Urban     |       228.69      | 36.6 | formerly smoked |    1   |
| 51676 | Female | 61.0 |       0      |       0       |      Yes     | Self-employed |      Rural     |       202.21      | NaN  | never smoked    |    1   |
| 31112 | Male   | 80.0 |       0      |       1       |      Yes     | Private       |      Rural     |       105.92      | 32.5 | never smoked    |    1   |
| 60182 | Female | 49.0 |       0      |       0       |      Yes     | Private       |      Urban     |       171.23      | 34.4 | smokes          |    1   |
| 1665  | Female | 79.0 |       1      |       0       |      Yes     | Self-employed |      Rural     |       174.12      | 24.0 | never smoked    |    1   |
| 18234 | Female | 80.0 |       1      |       0       |      Yes     | Private       |      Urban     |       83.75       | NaN  | never smoked    |    0   |
| 44873 | Female | 81.0 |       0      |       0       |      Yes     | Self-employed |      Urban     |       125.20      | 40.0 | never smoked    |    0   |
| 19723 | Female | 35.0 |       0      |       0       |      Yes     | Self-employed |      Rural     |       82.99       | 30.6 | never smoked    |    0   |
| 37544 | Male   | 51.0 |       0      |       0       |      Yes     | Private       |      Rural     |       166.29      | 25.6 | formerly smoked |    0   |
| 44679 | Female | 44.0 |       0      |       0       |      Yes     | Govt_job      |      Urban     |       85.28       | 26.2 | Unknown         |    0   |

## Data Preparation
Pada tahap data cleaning, dilakukan pengecekan terhadap *missing values* dan *outliers*. Kemudian *missing value* dan *outliers* pada dataset ini didrop. Pada tahap data preparation, pertama - tama dilakukan *encoding* dengan mengubah semua nilai categorical value menjadi angka. Hal ini dilakukan karena sebagian besar algoritma machine learning dan statistik memerlukan data yang bersifat numerik untuk dapat bekerja dengan baik. Kemudian dilakukan proses normalisasi menggunakan *min max normalization* agar dapat mempercepat proses komputasi. Setelah itu, dataset dibagi menjadi train dan test set.

## Modeling
Model machine learning dilatih menggunakan framework Tensorflow. Pada prosesnya, dibuat model deep learning sederhana menggunakan 3 layer Dense dengan layer pertama terdiri dari 16 neuron dan *input_shape* pada input layer adalah 20, layer kedua terdiri dari 20 neuron dengan aktivasi relu, dan output layer terdapat 1 neuron dengan fungsi aktivasi *sigmoid* untuk memprediksi apakah stroke atau tidak. Model kemudian dicompile menggunakan *SGD optimizer* dengan *learning_rate* 0.1 dan *binary_crossentropy loss function*. Model kemudian dilatih dengan 100 *epochs* dan *batch_size* 32 dan memperoleh akurasi pada training set sebesar 0.96 . 

## Evaluation
Proses evaluasi dilakukan dengan menggunakan test set dengan metrik *accuracy* dan *loss*. *accuracy* adalah metrik yang paling umum digunakan untuk mengukur performa model pada tugas klasifikasi. Metrik ini mengukur sejauh mana model dapat mengklasifikasikan data dengan benar dari total data yang dievaluasi. Sedangkan *loss* adalah metrik yang digunakan selama proses pelatihan model untuk mengukur seberapa baik model memetakan input ke output yang benar. Tujuan dari pelatihan adalah untuk meminimalkan nilai loss sehingga model dapat belajar dari data dan melakukan prediksi dengan lebih baik. Diperoleh *val_accuracy* sebesar 0.97 dan *val_loss* sebesar 0.13 pada test set. Berdasarkan perolehan tersebut, dapat disimpulkan bahwa model tergeneralisasi dengan baik.

## Referensi
* Agus Byna, Muhammad Basit, "Penerapan Metode Adaboost Untuk Mengoptimasi Prediksi Penyakit Stroke Dengan Algoritma Naïve Bayes", vol 9, No 3 (2020)
* Kompasiana, "Peran Artificial Intelligence Dalam Membantu Diagnosis Penyakit Stroke", https://www.kompasiana.com/fransisca89474/600f9ccbd541df305e3ca582/peran-artifical-intelligence-dalam-membantu-diagnosis-penyakit-stroke (Diakses pada 25 Juli 2023)

