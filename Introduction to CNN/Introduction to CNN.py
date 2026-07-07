import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==========================================
# 1. KONFIGURASI PATH & PARAMETER
# ==========================================
# Pastikan nama folder ini sama persis dengan folder dataset Anda
DATASET_PATH = 'TrashType_Image_Dataset' 
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

print("Menginisialisasi pemrosesan data...")

# ==========================================
# 2. PREPROCESSING & AUGMENTASI DATA
# ==========================================
# Membuat generator dengan split 80% training dan 20% validation
datagen = ImageDataGenerator(
    rescale=1./255,          # Mengubah nilai piksel menjadi rentang 0-1
    rotation_range=20,       # Memutar gambar acak sampai 20 derajat
    zoom_range=0.2,          # Memperbesar/memperkecil acak
    horizontal_flip=True,    # Membalik gambar secara horizontal
    validation_split=0.2     # Alokasi 20% data untuk validasi (tes internal)
)

# Generator untuk Data Training (80%)
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Generator untuk Data Validasi (20%)
val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Menampilkan kelas/label sampah yang terdeteksi
print("\nKelas/Label Sampah yang ditemukan:", list(train_data.class_indices.keys()))
NUM_CLASSES = train_data.num_classes

# ==========================================
# 3. MEMBANGUN ARSITEKTUR CNN MODEL
# ==========================================
print("\nMembangun arsitektur model CNN...")
model = Sequential([
    # Blok Konvolusi 1
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    
    # Blok Konvolusi 2
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Blok Konvolusi 3 (Opsional, ditambahkan agar model lebih pintar)
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Mengubah matriks menjadi vektor satu dimensi
    Flatten(),
    
    # Fully Connected Layer (Jaringan Saraf Tiruan)
    Dense(128, activation='relu'),
    Dropout(0.5), # Mencegah overfitting (menghafal gambar)
    
    # Output Layer (Disesuaikan dengan jumlah kategori sampah Anda)
    Dense(NUM_CLASSES, activation='softmax')
])

# Kompilasi Model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary() # Menampilkan ringkasan model di terminal

# ==========================================
# 4. PROSES PELATIHAN (MODEL TRAINING)
# ==========================================
print(f"\nMemulai pelatihan model selama {EPOCHS} epoch...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ==========================================
# 5. MENYIMPAN MODEL TERLATIH
# ==========================================
# Model disimpan agar nanti bisa langsung dipakai tanpa perlu training ulang
MODEL_NAME = 'model_klasifikasi_sampah.h5'
model.save(MODEL_NAME)
print(f"\nPelatihan Selesai! Model sukses disimpan dengan nama: '{MODEL_NAME}'")