# Strategi Pipeline CUDA FLIP Fluid

Dokumen ini menjelaskan setiap tahap utama dalam `flip_fluid_cuda.cu` dan strategi implementasi yang dipilih untuk masing-masing tahap.

## 1. Integrasi Partikel
- Kernel utama: `integrateKernel`
- Fungsi: melangkahkan posisi partikel berdasarkan kecepatan dan memperhitungkan gravitasi.
- Strategi: metode Euler maju sederhana.
- Alasan: perhitungan ini bersifat embarrassingly parallel, setiap partikel dapat diupdate independen tanpa sinkronisasi.

## 2. (Opsional) Dorong Partikel Agar Tidak Saling Tumpang Tindih
- Fungsi utama: `pushApart`
- Kernel penting: `countKernel`, `scatterKernel`, `separateKernel`
- Strategi: spatial hash + Jacobi-style separation.
- Penjelasan:
  - `countKernel` dan `scatterKernel` membangun struktur data seluler untuk mencari tetangga secara lokal.
  - `separateKernel` membaca snapshot posisi dan warna partikel lama (`posIn` / `colIn`) dan menulis koreksi hanya ke partikel sendiri.
- Alasan implementasi:
  - Cara CPU yang sequential tidak cocok untuk GPU; update in-place antar partikel akan menghasilkan race condition.
  - Jacobi-style membuat setiap partikel melakukan koreksi mandiri berdasarkan tetangga yang konsisten dari snapshot, lalu iterasi berikutnya memperbaiki lagi.
  - Ini mengorbankan sedikit ketepatan urutan demi determinisme dan keamanan paralel.

## 3. Collide Geometry and Boundary
- Kernel utama: `collisionsKernel`
- Fungsi: menghitung tumbukan partikel dengan batas domain dan dengan obstacle bundar.
- Strategi: pelurusan posisi dan nol-kan komponen kecepatan pada dinding; jika berada di dalam obstacle, ganti kecepatan dengan kecepatan obstacle.
- Alasan: batasan domain bersifat lokal per partikel, sehingga cara ini paling sederhana dan cocok untuk GPU.

## 4. Transfer Data Partikel ke Grid (P2G)
- Fungsi utama: `transferToGrid`
- Kernel penting: `savePrevKernel`, `classifyGridKernel`, `classifyParticlesKernel`, `p2gKernel`, `normalizeKernel`, `restoreSolidKernel`
- Strategi:
  - Reset dan snapshot dulu grid menggunakan `savePrevKernel`.
  - Klasifikasi sel padat/udara dan lalu fluid dengan `classifyGridKernel` + `classifyParticlesKernel`.
  - Transfer kecepatan partikel dengan interpolasi bilinear ke 4 node terdekat di MAC grid (`p2gKernel`).
  - Akumulasi bobot pembobotan dalam `fldD`, lalu normalisasi di `normalizeKernel`.
  - `restoreSolidKernel` memulihkan nilai kecepatan pada permukaan solid dari snapshot sebelumnya.
- Alasan implementasi:
  - `p2gKernel` menggunakan `atomicAdd` karena banyak partikel dapat menyumbang ke sel/grid face yang sama.
  - Penggunaan MAC grid berarti setiap komponen kecepatan ditransfer terpisah dan wajah grid membutuhkan penanganan boundary yang teliti.
  - `restoreSolidKernel` menegakkan kondisi batas solid tanpa mempengaruhi formulasi solver karena nilai wajah padat harus tetap konsisten.

## 5. Menghitung Density dan Pressure
- Kernel utama: `updateDensity`, `solvePressure`
- Kernel penting: `densityKernel`, `restDensityKernel`, `pressureInitKernel`, `pressureRBKernel`
- Strategi Density:
  - Hitung densitas partikel ke masing-masing grid cell dengan interpolasi bilinear (`densityKernel`).
  - `restDensityKernel` rata-rata densitas hanya pada sel fluid untuk parameter kompensasi drift.
- Strategi Pressure:
  - Inisialisasi tekanan ke nol dan snapshot lagi kecepatan grid untuk FLIP dengan `pressureInitKernel`.
  - Gunakan red-black Gauss-Seidel iteratif melalui kernel `pressureRBKernel`.
- Penjelasan detail `pressureRBKernel`:
  - Ini bukan solver Jacobi standar; ia mengupdate tekanan secara in-place pada sel-sel dengan paritas selang-seling (`color` merah/hitam).
  - Red-black ordering membuat setiap wajah grid hanya ditulis sekali per pass, sehingga aman paralel tanpa atomics.
  - Setiap iterasi menjalankan dua pass: merah lalu hitam, yang meniru urutan lexicographic CPU tapi tetap paralel.
  - Termasuk opsi kompensasi drift dari selisih densitas terhadap `restDensity`.
- Alasan implementasi:
  - Red-black GS memberikan konvergensi lebih cepat dibanding Jacobi dan lebih mudah diparalelkan daripada sweep lexicographic penuh.
  - In-place update pada GPU penting untuk efisiensi memori dan memperkecil overhead sinkronisasi.

## 6. Transfer Grid Velocity ke Partikel (G2P)
- Kernel utama: `transferToParticles`
- Kernel penting: `g2pKernel`
- Strategi:
  - Interpolasi bilinear dari grid ke partikel.
  - Hitung nilai PIC dari grid saat ini dan nilai FLIP berdasarkan perubahan grid sejak pre-P2G (`prevU`, `prevV`).
  - Blending antara PIC dan FLIP dengan parameter `flipRatio`.
  - Abaikan kontribusi sel udara murni dengan validasi `cellType`.
- Alasan implementasi:
  - Model FLIP mengurangi numerik difusi dibanding PIC, tetapi PIC dibutuhkan untuk stabilitas. Blend mempertahankan keduanya.
  - Validasi tetangga mencegah sampling nilai grid yang tidak valid di area udara.

## 7. Pewarnaan Partikel dan Sel untuk Visualisasi
- Kernel utama: `updateColors`
- Kernel penting: `particleColorsKernel`, `cellColorsKernel`, `setSciColorDev`
- Strategi:
  - Partikel diberi efek fading dan partikel ber-densitas rendah diwarnai biru.
  - Sel grid diberi warna berdasarkan tipe: solid abu-abu, fluid dengan ramp warna ilmiah, udara hitam.
- Alasan implementasi:
  - Ini hanya jalur visualisas, jadi fokus pada penyajian yang mudah dibaca, bukan akurasi fisik.

## 8. Carve Benda ke Grid, Termasuk Partikel
- Kernel utama: `carveKernel`
- Fungsi: menandai sel grid sebagai solid ketika obstacle berada di dalamnya dan menginisialisasi kecepatan wajah grid dengan kecepatan obstacle.
- Strategi: batas area obstacle bundar, set `s[i]=0.0` untuk sel dalam obstacle, lalu atur `u`/`v` ke `vx`/`vy` di dirinya dan tetangga.
- Alasan implementasi:
  - Ini menciptakan solid mask pada grid sehingga pressure solver dan transfer P2G/G2P memperlakukan obstacle sebagai batas tetap.
  - Perubahan kecepatan di wajah grid membantu memperkenalkan gerakan obstacle secara langsung ke field fluida.

## Kesimpulan
Strategi utama dalam implementasi CUDA ini adalah memilih algoritma yang sederhana ketika bisa, tetapi beralih ke metode paralel khusus ketika ketergantungan data muncul. Contoh pentingnya adalah red-black Gauss-Seidel untuk solver pressure dan Jacobi-style particle separation untuk keamanan race-free pada GPU.
