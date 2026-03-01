import os
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20

DATA_DIR = 'data' 

LIFECYCLE_STAGES = ['bud_emergence', 'expansion_maturity', 'senescence', 'abscission']

def collect_tree_data(root_dir=DATA_DIR):
    image_paths = []
    species_labels = []
    lifecycle_labels = []

    if not os.path.exists(root_dir):
        print(f"Warning: Directory '{root_dir}' not found. Please create it and add your dataset.")
        os.makedirs(os.path.join(root_dir, 'oak', 'bud_emergence'), exist_ok=True)
        return pd.DataFrame(columns=['path', 'species', 'lifecycle'])

    for species in os.listdir(root_dir):
        species_dir = os.path.join(root_dir, species)
        if os.path.isdir(species_dir):
            for stage in os.listdir(species_dir):
                stage_dir = os.path.join(species_dir, stage)
                if os.path.isdir(stage_dir):
                    for img_file in os.listdir(stage_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_paths.append(os.path.join(stage_dir, img_file))
                            species_labels.append(species)
                            lifecycle_labels.append(stage)
    
    df = pd.DataFrame({
        'path': image_paths,
        'species': species_labels,
        'lifecycle': lifecycle_labels
    })
    return df

df_combined = collect_tree_data()

if df_combined.empty or len(df_combined) < 10:
    print("Dataset is empty or too small to begin training. Please add images to the 'data/' directory following the structure defined.")
    print("Example: data/oak/bud_emergence/image1.jpg")
    exit(0)

print(f"Total number of images collected: {len(df_combined)}")

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            fmt = img.format
            if fmt not in ('JPEG', 'PNG', 'BMP', 'GIF'):
                return False
            img.verify()
        return True
    except Exception:
        return False

print("Validating images (filtering out corrupt/webp files)...")
valid_mask = df_combined['path'].apply(is_valid_image)
df_combined = df_combined[valid_mask].reset_index(drop=True)
print(f"Valid images after filtering: {len(df_combined)} (removed {(~valid_mask).sum()} bad files)")

df_combined['stratify_label'] = df_combined['species'] + '_' + df_combined['lifecycle']

min_samples = 4
label_counts = df_combined['stratify_label'].value_counts()
valid_labels = label_counts[label_counts >= min_samples].index
df_filtered = df_combined[df_combined['stratify_label'].isin(valid_labels)].copy()

print(f"Dataset size after filtering low-count classes: {len(df_filtered)}")

unique_species = sorted(df_filtered['species'].unique())
unique_lifecycles = sorted(df_filtered['lifecycle'].unique())
num_species_classes = len(unique_species)
num_lifecycle_classes = len(unique_lifecycles)

print(f"Species Classes ({num_species_classes}): {unique_species}")
print(f"Lifecycle Classes ({num_lifecycle_classes}): {unique_lifecycles}")

train_df, temp_df = train_test_split(
    df_filtered, test_size=0.30, stratify=df_filtered['stratify_label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['stratify_label'], random_state=42
)

print(f"Train split: {len(train_df)} | Validation split: {len(val_df)} | Test split: {len(test_df)}")

species_to_idx = {name: idx for idx, name in enumerate(unique_species)}
lifecycle_to_idx = {name: idx for idx, name in enumerate(unique_lifecycles)}

def process_path(file_path, species_label, lifecycle_label):
    img_raw = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    targets = {
        'species': tf.one_hot(species_label, depth=num_species_classes),
        'lifecycle': tf.one_hot(lifecycle_label, depth=num_lifecycle_classes)
    }
    return img, targets

def augment(img, targets):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2) 
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2) 
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, targets

def make_dataset(df, shuffle=False, perform_augmentation=False):
    paths = df['path'].values
    species_labels = df['species'].map(species_to_idx).values
    lifecycle_labels = df['lifecycle'].map(lifecycle_to_idx).values
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, species_labels, lifecycle_labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
        
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    if perform_augmentation:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = make_dataset(train_df, shuffle=True, perform_augmentation=True)
val_ds = make_dataset(val_df, shuffle=False, perform_augmentation=False)
test_ds = make_dataset(test_df, shuffle=False, perform_augmentation=False)

print("\nBuilding Multi-Head MobileNetV2 Architecture...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False 

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

species_head = Dense(num_species_classes, activation='softmax', name='species')(x)
lifecycle_head = Dense(num_lifecycle_classes, activation='softmax', name='lifecycle')(x)

model = Model(inputs=base_model.input, outputs=[species_head, lifecycle_head])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        'species': 'categorical_crossentropy', 
        'lifecycle': 'categorical_crossentropy'
    },
    loss_weights={
        'species': 1.0,
        'lifecycle': 1.0
    }, 
    metrics={
        'species': 'accuracy', 
        'lifecycle': 'accuracy'
    }
)

model.summary()

checkpoint_filepath = 'best_tree_lifecycle_model.weights.h5'

callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss', 
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        verbose=1, 
        min_lr=1e-6
    )
]

print("\nStarting Training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

print("\nEvaluating on Test Set...")
model.load_weights(checkpoint_filepath)
test_results = model.evaluate(test_ds, verbose=1)

print("\n--- Test Results ---")
print(f"Total Loss: {test_results[0]:.4f}")
print(f"Species Loss: {test_results[1]:.4f}")
print(f"Lifecycle Loss: {test_results[2]:.4f}")
print(f"Species Accuracy: {test_results[3]:.4f}")
print(f"Lifecycle Accuracy: {test_results[4]:.4f}")

idx_to_species = {v: k for k, v in species_to_idx.items()}
idx_to_lifecycle = {v: k for k, v in lifecycle_to_idx.items()}

def predict_tree_leaf(img_path):
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)
        species_pred = preds[0][0]
        lifecycle_pred = preds[1][0]
        
        species_label = idx_to_species[np.argmax(species_pred)]
        lifecycle_label = idx_to_lifecycle[np.argmax(lifecycle_pred)]
        
        species_conf = np.max(species_pred) * 100
        lifecycle_conf = np.max(lifecycle_pred) * 100
        
        return {
            'species': species_label,
            'species_confidence': species_conf,
            'lifecycle': lifecycle_label,
            'lifecycle_confidence': lifecycle_conf
        }
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

print("\nRefactoring Complete! You can now use predict_tree_leaf() function to infer images.")
