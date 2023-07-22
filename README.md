# Laporan Proyek Machine Learning - Alexander Gosal
## Domain Proyek
Stroke merupakan salah satu kondisi medis yang serius dan dapat mengancam jiwa, terjadi ketika suplai darah ke otak terganggu, baik karena adanya sumbatan pada pembuluh darah atau pecahnya pembuluh darah. Menurut World Health Organization (WHO), troke adalah penyebab kematian ke-2 secara global, bertanggung jawab atas sekitar 11% dari total kematian. Menerapkan teknologi machine learning dalam bidang medis, seperti mengidentifikasi potensi stroke, dapat memiliki dampak yang signifikan dalam upaya pencegahan dan pengobatan lebih dini, yang pada gilirannya dapat meningkatkan tingkat kesembuhan dan kualitas hidup pasien. Dataset stroke prediction digunakan dalam proyek ini untuk memberikan informasi penting tentang pasien, termasuk faktor risiko seperti usia, jenis kelamin, tekanan darah, status perokok, status perkawinan, serta informasi medis lainnya. Data-data ini akan diolah dan dianalisis oleh model machine learning yang dibangun untuk menemukan pola-pola yang dapat membedakan antara pasien yang berpotensi mengalami stroke dengan yang tidak.

Tujuan dari proyek ini adalah untuk menciptakan model yang akurat dan dapat diandalkan dalam mengidentifikasi potensi stroke pada pasien. Dengan memiliki model yang baik, pihak medis dapat menggunakan alat ini sebagai salah satu alat bantu dalam diagnosa awal, memungkinkan intervensi lebih cepat dan penanganan yang lebih tepat. Proyek ini juga bertujuan untuk mengeksplorasi dan menerapkan deep learning sebagai alternatif yang lebih kuat dalam mengatasi masalah klasifikasi stroke. Dengan menerapkan metode deep learning, diharapkan model dapat mengidentifikasi pola yang lebih kompleks dan mendalam pada data medis stroke.

Namun, perlu diingat bahwa meskipun machine learning dapat memberikan kontribusi yang signifikan dalam bidang medis, hasil dari model ini sebaiknya digunakan hanya sebagai alat bantu dan tidak menggantikan pendapat atau evaluasi dari profesional medis yang berkualifikasi. Kesehatan merupakan hal yang sangat sensitif, sehingga keputusan akhir mengenai diagnosis dan pengobatan harus selalu berdasarkan penilaian dari dokter yang berpengalaman.

Referensi :
[Penerapan Metode Adaboost Untuk Mengoptimasi Prediksi Penyakit Stroke Dengan Algoritma Naïve Bayes](http://jurnal.atmaluhur.ac.id/index.php/sisfokom/article/view/1023/685)

## Business Understanding
#### Problem Statements : 
Penerapan machine learning dalam klasifikasi stroke menjanjikan berbagai manfaat yang signifikan. Machine learning dapat membantu dalam deteksi dini potensi stroke dengan mengidentifikasi tanda-tanda awal yang mungkin tidak dapat diidentifikasi dengan metode konvensional. Deteksi dini ini memungkinkan penanganan medis yang lebih cepat dan tepat waktu dan meningkatkan peluang kesembuhan pasien.
#### Goals :
1. Mengembangkan model klasifikasi stroke yang akurat.
2. Mengeksplorasi penerapan deep learning dalam meningkatkan akurasi model klasifikasi stroke.

## Data Understanding
[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) digunakan untuk memprediksi apakah pasien kemungkinan terkena stroke berdasarkan parameter input seperti jenis kelamin, usia, berbagai penyakit, dan status merokok. Setiap baris dalam data memberikan informasi yang relevan tentang pasien.
#### Variabel-variabel pada Stroke Prediction Dataset adalah sebagai berikut:
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

## Data Preparation
Pada tahap data preparation, pertama - tama dilakukan *encoding* dengan mengubah semua nilai categorical value menjadi angka. Hal ini dilakukan karena sebagian besar algoritma machine learning dan statistik memerlukan data yang bersifat numerik untuk dapat bekerja dengan baik. Kemudian dilakukan proses normalisasi menggunakan *min max normalization* agar dapat mempercepat proses komputasi. Setelah itu, dataset dibagi menjadi train dan test set.

## Modeling
Model machine learning dilatih menggunakan framework Tensorflow. Pada prosesnya, dibuat model deep learning sederhana menggunakan layers keras dengan *input_shape* pada input layer adalah 20. Kemudian pada output layer terdapat 1 neuron dengan fungsi aktivasi *sigmoid* untuk memprediksi apakah stroke atau tidak. Model kemudian dicompile menggunakan *SGD optimizer* dan *binary_crossentropy loss function*. Model kemudian dilatih dengan 10 *epochs* dan *batch_size* 32 dan memperoleh akurasi pada training set sebesar 0.96 . 

## Evaluation
Proses evaluasi dilakukan dengan menggunakan test set dengan metrik *accuracy* dan *loss*. *accuracy* adalah metrik yang paling umum digunakan untuk mengukur performa model pada tugas klasifikasi. Metrik ini mengukur sejauh mana model dapat mengklasifikasikan data dengan benar dari total data yang dievaluasi. Sedangkan *loss* adalah metrik yang digunakan selama proses pelatihan model untuk mengukur seberapa baik model memetakan input ke output yang benar. Tujuan dari pelatihan adalah untuk meminimalkan nilai loss sehingga model dapat belajar dari data dan melakukan prediksi dengan lebih baik. Diperoleh *accuracy* sebesar 0.97 dan *loss* sebesar 0.12 pada test set.
