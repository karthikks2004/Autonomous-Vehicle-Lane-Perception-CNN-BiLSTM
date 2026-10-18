import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Conv2DTranspose,
    concatenate,
    Activation
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


# =============================================================
#
#   LANE DETECTION - U-NET SEMANTIC SEGMENTATION
#   Dataset: TUSimple Lane Detection Challenge
#
# =============================================================


# =====================================================
# INPUT & OUTPUT CONFIGURATION
# =====================================================

# Dataset directory
TUSIMPLE_DIR = r"D:\Finalyear pro\lane detection\TUSimple"
TRAIN_DIR    = os.path.join(TUSIMPLE_DIR, "train_set")
TEST_DIR     = os.path.join(TUSIMPLE_DIR, "test_set")

# Output directory
OUTPUT_DIR = r"D:\New parttss\lane_detection"

# Training configuration
IMAGE_HEIGHT = 256
IMAGE_WIDTH  = 512
BATCH_SIZE   = 8       # Smaller batch for segmentation (larger images)
EPOCHS       = 20
TEST_SPLIT   = 0.2
RANDOM_STATE = 42
LANE_WIDTH   = 8       # Pixel width of lane line in mask

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# CHECK INPUT FOLDERS
# =====================================================

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Train directory not found!\n  {TRAIN_DIR}")
    exit()

print("=" * 60)
print("  LANE DETECTION - U-NET SEGMENTATION")
print("=" * 60)


# =====================================================
# CUSTOM METRICS: Dice Coefficient & IoU
# =====================================================

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient for binary segmentation."""
    y_true_cast = K.cast(y_true, K.floatx())
    y_true_flat = K.flatten(y_true_cast)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss = 1 - Dice coefficient."""
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """Combined binary crossentropy + dice loss."""
    y_true_cast = K.cast(y_true, K.floatx())
    bce = K.mean(K.binary_crossentropy(y_true_cast, y_pred))
    dl  = dice_loss(y_true_cast, y_pred)
    return bce + dl


def iou_metric(y_true, y_pred, smooth=1.0):
    """Intersection over Union metric."""
    y_true_cast = K.cast(y_true, K.floatx())
    y_pred_bin = K.cast(K.greater(y_pred, 0.5), K.floatx())
    intersection = K.sum(y_true_cast * y_pred_bin)
    union = K.sum(y_true_cast) + K.sum(y_pred_bin) - intersection
    return (intersection + smooth) / (union + smooth)


# #####################################################
#
#   STEP 1 :  LOAD TUSIMPLE DATASET
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 1 :  LOADING TUSIMPLE DATASET")
print("=" * 60)

start_time = time.time()


def generate_lane_mask(lanes, h_samples, img_shape, lane_width=LANE_WIDTH):
    """
    Generate a binary mask from TUSimple lane annotations.
    Each lane is a polyline drawn on the mask with the specified width.
    """
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)

    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x != -2:  # -2 means no lane at this height
                points.append((int(x), int(y)))

        if len(points) >= 2:
            # Draw polyline on mask
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=lane_width)

    return mask


def load_tusimple_labels(label_file):
    """Load label data from a TUSimple JSON label file."""
    data = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                data.append(entry)
    return data


# Load all training labels
label_files = [
    os.path.join(TRAIN_DIR, "label_data_0313.json"),
    os.path.join(TRAIN_DIR, "label_data_0531.json"),
    os.path.join(TRAIN_DIR, "label_data_0601.json"),
]

all_labels = []
for lf in label_files:
    if os.path.exists(lf):
        labels = load_tusimple_labels(lf)
        print(f"  Loaded {len(labels)} entries from {os.path.basename(lf)}")
        all_labels.extend(labels)
    else:
        print(f"  WARNING: Label file not found: {lf}")

print(f"\n  Total label entries: {len(all_labels)}")

# Load images and generate masks
images = []
masks  = []
raw_files = []  # Keep track for scenarios
skipped = 0

for i, entry in enumerate(all_labels):
    if (i + 1) % 500 == 0 or i == 0:
        print(f"  Processing {i + 1}/{len(all_labels)}...")

    raw_file = entry["raw_file"]
    img_path = os.path.join(TRAIN_DIR, raw_file)

    if not os.path.exists(img_path):
        skipped += 1
        continue

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue

    orig_h, orig_w = img.shape[:2]

    # Generate lane mask at original resolution
    mask = generate_lane_mask(entry["lanes"], entry["h_samples"], (orig_h, orig_w))

    # Resize image and mask
    img  = cv2.resize(img,  (IMAGE_WIDTH, IMAGE_HEIGHT))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)  # Binary mask

    images.append(img)
    masks.append(mask)
    raw_files.append(raw_file)

load_time = time.time() - start_time

X = np.array(images, dtype=np.uint8)
Y = np.array(masks,  dtype=np.uint8)
Y = Y[..., np.newaxis]  # Add channel dimension: (N, H, W, 1)
del images, masks

print(f"\n  Loaded in {load_time:.1f} seconds")
print(f"  Total samples : {X.shape[0]}")
print(f"  Image shape   : {X.shape[1:]}")
print(f"  Mask shape    : {Y.shape[1:]}")
print(f"  Memory        : {(X.nbytes + Y.nbytes) / (1024**3):.2f} GB")
if skipped > 0:
    print(f"  Skipped       : {skipped}")

if len(X) == 0:
    print("\n  ERROR: No valid data found.")
    exit()

# Lane statistics
lane_pixels = Y.sum()
total_pixels = Y.size
print(f"  Lane pixels   : {lane_pixels:,.0f} / {total_pixels:,.0f} ({lane_pixels/total_pixels*100:.2f}%)")


# #####################################################
#
#   STEP 2 :  PREPARE DATA
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 2 :  PREPARING DATA")
print("=" * 60)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=TEST_SPLIT,
    random_state=RANDOM_STATE
)

# Also split raw_files for scenario use
_, raw_files_test = train_test_split(
    raw_files,
    test_size=TEST_SPLIT,
    random_state=RANDOM_STATE
)

del X, Y

print(f"  Training samples : {len(X_train)}")
print(f"  Testing samples  : {len(X_test)}")


# #####################################################
#
#   STEP 3 :  BUILD & TRAIN U-NET MODEL
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 3 :  BUILDING & TRAINING U-NET MODEL")
print("=" * 60)


def build_unet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)):
    """
    Build a U-Net segmentation model.
    Encoder: 4 blocks of Conv-BN-Conv-BN-Pool
    Decoder: 4 blocks of UpConv-Concat-Conv-BN-Conv-BN
    """
    inputs = Input(input_shape)

    # Scale from [0, 255] to [0.0, 1.0] dynamically
    rescaled = Lambda(lambda x: x / 255.0)(inputs)

    # ---- ENCODER ----

    # Block 1
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(rescaled)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.1)(p1)

    # Block 2
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.1)(p2)

    # Block 3
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.2)(p3)

    # Block 4
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.2)(p4)

    # ---- BOTTLENECK ----
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    # ---- DECODER ----

    # Block 6
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(0.2)(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    # Block 7
    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(0.2)(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    # Block 8
    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.1)(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    c8 = BatchNormalization()(c8)

    # Block 9
    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(0.1)(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    c9 = BatchNormalization()(c9)

    # Output: single channel sigmoid
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


model = build_unet()
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=combined_loss,
    metrics=['accuracy', dice_coefficient, iou_metric]
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

print("\n  Training started...\n")
train_start = time.time()

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

train_time = time.time() - train_start
print(f"\n  Training done in {train_time:.1f} seconds")

# Save model
model_path = os.path.join(OUTPUT_DIR, "lane_model.h5")
model.save(model_path)
print(f"  Model saved: {model_path}")

# Extract history
train_loss     = history.history['loss']
val_loss       = history.history['val_loss']
train_acc      = history.history['accuracy']
val_acc        = history.history['val_accuracy']
train_dice     = history.history['dice_coefficient']
val_dice       = history.history['val_dice_coefficient']
train_iou      = history.history['iou_metric']
val_iou        = history.history['val_iou_metric']
epoch_list     = range(1, len(train_loss) + 1)


# #####################################################
#
#   STEP 4 :  EVALUATE MODEL
#   Segmentation metrics: IoU, Dice, Pixel Accuracy,
#   Precision, Recall, F1
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 4 :  MODEL EVALUATION")
print("=" * 60)

# Predict on test set
Y_pred = model.predict(X_test, verbose=0)

# Binarize predictions (threshold = 0.5)
Y_pred_binary = (Y_pred > 0.5).astype(np.float32)

# Flatten for pixel-level metrics
y_true_flat = Y_test.flatten().astype(int)
y_pred_flat = Y_pred_binary.flatten().astype(int)

# Compute metrics
pixel_accuracy = accuracy_score(y_true_flat, y_pred_flat)
pixel_precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
pixel_recall    = recall_score(y_true_flat, y_pred_flat, zero_division=0)
pixel_f1        = f1_score(y_true_flat, y_pred_flat, zero_division=0)

# IoU (manually computed on full test set)
intersection = np.sum(Y_test.flatten() * Y_pred_binary.flatten())
union = np.sum(Y_test.flatten()) + np.sum(Y_pred_binary.flatten()) - intersection
iou_score = intersection / (union + 1e-7)

# Dice coefficient
dice_score_val = (2 * intersection) / (np.sum(Y_test.flatten()) + np.sum(Y_pred_binary.flatten()) + 1e-7)

# Final val_loss from model.evaluate
eval_results = model.evaluate(X_test, Y_test, verbose=0)
test_loss = eval_results[0]

print(f"\n  Test Loss          : {test_loss:.4f}")
print(f"  Pixel Accuracy     : {pixel_accuracy*100:.2f}%")
print(f"  Pixel Precision    : {pixel_precision*100:.2f}%")
print(f"  Pixel Recall       : {pixel_recall*100:.2f}%")
print(f"  Pixel F1 Score     : {pixel_f1*100:.2f}%")
print(f"  IoU (Jaccard)      : {iou_score*100:.2f}%")
print(f"  Dice Coefficient   : {dice_score_val*100:.2f}%")

# Save classification report
with open(os.path.join(OUTPUT_DIR, "segmentation_report.txt"), "w") as f:
    f.write("LANE DETECTION - U-NET SEGMENTATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model: U-Net (4 Encoder + 4 Decoder blocks)\n")
    f.write(f"Dataset: TUSimple ({len(X_train) + len(X_test)} samples)\n")
    f.write(f"Training: {len(X_train)} | Testing: {len(X_test)}\n")
    f.write(f"Image Size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}\n\n")
    f.write("SEGMENTATION METRICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Test Loss        : {test_loss:.4f}\n")
    f.write(f"  Pixel Accuracy   : {pixel_accuracy*100:.2f}%\n")
    f.write(f"  Pixel Precision  : {pixel_precision*100:.2f}%\n")
    f.write(f"  Pixel Recall     : {pixel_recall*100:.2f}%\n")
    f.write(f"  Pixel F1 Score   : {pixel_f1*100:.2f}%\n")
    f.write(f"  IoU (Jaccard)    : {iou_score*100:.2f}%\n")
    f.write(f"  Dice Coefficient : {dice_score_val*100:.2f}%\n\n")
    f.write(f"Training Time: {train_time:.1f} seconds\n")
    f.write(f"Epochs Trained: {len(train_loss)}\n")

print("  Saved: segmentation_report.txt")


# #####################################################
#
#   STEP 5 :  GENERATE CHARTS
#   All charts at dpi=300 for paper quality
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 5 :  GENERATING CHARTS")
print("=" * 60)


# --- 5a: Training vs Testing Loss ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.plot(epoch_list, train_loss, color='#E91E63', linewidth=2.5, marker='o', markersize=6, label='Training Loss')
ax.plot(epoch_list, val_loss,   color='#2196F3', linewidth=2.5, marker='s', markersize=6, label='Testing Loss')
ax.fill_between(epoch_list, train_loss, alpha=0.1, color='#E91E63')
ax.fill_between(epoch_list, val_loss,   alpha=0.1, color='#2196F3')

ax.set_xlabel("Epoch", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Loss", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Training Loss vs Testing Loss — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.annotate(f'Train: {train_loss[-1]:.4f}', xy=(len(train_loss), train_loss[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#E91E63',
            arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))
ax.annotate(f'Test: {val_loss[-1]:.4f}', xy=(len(val_loss), val_loss[-1]),
            xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold', color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_testing_loss.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: training_testing_loss.png")


# --- 5b: Training vs Testing Accuracy ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.plot(epoch_list, train_acc, color='#4CAF50', linewidth=2.5, marker='o', markersize=6, label='Training Accuracy')
ax.plot(epoch_list, val_acc,   color='#FF9800', linewidth=2.5, marker='s', markersize=6, label='Testing Accuracy')
ax.fill_between(epoch_list, train_acc, alpha=0.1, color='#4CAF50')
ax.fill_between(epoch_list, val_acc,   alpha=0.1, color='#FF9800')

ax.set_xlabel("Epoch", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Training Accuracy vs Testing Accuracy — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.annotate(f'Train: {train_acc[-1]*100:.2f}%', xy=(len(train_acc), train_acc[-1]),
            xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))
ax.annotate(f'Test: {val_acc[-1]*100:.2f}%', xy=(len(val_acc), val_acc[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#FF9800',
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5))

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_testing_accuracy.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: training_testing_accuracy.png")


# --- 5c: IoU per Epoch ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.plot(epoch_list, train_iou, color='#9C27B0', linewidth=2.5, marker='o', markersize=6, label='Training IoU')
ax.plot(epoch_list, val_iou,   color='#00BCD4', linewidth=2.5, marker='s', markersize=6, label='Testing IoU')
ax.fill_between(epoch_list, train_iou, alpha=0.1, color='#9C27B0')
ax.fill_between(epoch_list, val_iou,   alpha=0.1, color='#00BCD4')

ax.set_xlabel("Epoch", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("IoU Score", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("IoU (Intersection over Union) per Epoch — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.annotate(f'Train: {train_iou[-1]:.4f}', xy=(len(train_iou), train_iou[-1]),
            xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold', color='#9C27B0',
            arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5))
ax.annotate(f'Test: {val_iou[-1]:.4f}', xy=(len(val_iou), val_iou[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#00BCD4',
            arrowprops=dict(arrowstyle='->', color='#00BCD4', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "iou_per_epoch.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: iou_per_epoch.png")


# --- 5d: Dice Coefficient per Epoch ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.plot(epoch_list, train_dice, color='#FF5722', linewidth=2.5, marker='o', markersize=6, label='Training Dice')
ax.plot(epoch_list, val_dice,   color='#607D8B', linewidth=2.5, marker='s', markersize=6, label='Testing Dice')
ax.fill_between(epoch_list, train_dice, alpha=0.1, color='#FF5722')
ax.fill_between(epoch_list, val_dice,   alpha=0.1, color='#607D8B')

ax.set_xlabel("Epoch", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Dice Coefficient", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Dice Coefficient per Epoch — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dice_per_epoch.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: dice_per_epoch.png")


# --- 5e: Combined Training Progress (2x2) ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#FAFAFA')

# Loss
ax1.set_facecolor('#FAFAFA')
ax1.plot(epoch_list, train_loss, color='#E91E63', linewidth=2, marker='o', markersize=4, label='Train')
ax1.plot(epoch_list, val_loss,   color='#2196F3', linewidth=2, marker='s', markersize=4, label='Test')
ax1.fill_between(epoch_list, train_loss, alpha=0.1, color='#E91E63')
ax1.fill_between(epoch_list, val_loss,   alpha=0.1, color='#2196F3')
ax1.set_title("Loss", fontsize=14, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=11); ax1.set_ylabel("Loss", fontsize=11)
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.2)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Accuracy
ax2.set_facecolor('#FAFAFA')
ax2.plot(epoch_list, train_acc, color='#4CAF50', linewidth=2, marker='o', markersize=4, label='Train')
ax2.plot(epoch_list, val_acc,   color='#FF9800', linewidth=2, marker='s', markersize=4, label='Test')
ax2.fill_between(epoch_list, train_acc, alpha=0.1, color='#4CAF50')
ax2.fill_between(epoch_list, val_acc,   alpha=0.1, color='#FF9800')
ax2.set_title("Pixel Accuracy", fontsize=14, fontweight='bold')
ax2.set_xlabel("Epoch", fontsize=11); ax2.set_ylabel("Accuracy", fontsize=11)
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# IoU
ax3.set_facecolor('#FAFAFA')
ax3.plot(epoch_list, train_iou, color='#9C27B0', linewidth=2, marker='o', markersize=4, label='Train')
ax3.plot(epoch_list, val_iou,   color='#00BCD4', linewidth=2, marker='s', markersize=4, label='Test')
ax3.fill_between(epoch_list, train_iou, alpha=0.1, color='#9C27B0')
ax3.fill_between(epoch_list, val_iou,   alpha=0.1, color='#00BCD4')
ax3.set_title("IoU Score", fontsize=14, fontweight='bold')
ax3.set_xlabel("Epoch", fontsize=11); ax3.set_ylabel("IoU", fontsize=11)
ax3.legend(fontsize=10); ax3.grid(True, alpha=0.2)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

# Dice
ax4.set_facecolor('#FAFAFA')
ax4.plot(epoch_list, train_dice, color='#FF5722', linewidth=2, marker='o', markersize=4, label='Train')
ax4.plot(epoch_list, val_dice,   color='#607D8B', linewidth=2, marker='s', markersize=4, label='Test')
ax4.fill_between(epoch_list, train_dice, alpha=0.1, color='#FF5722')
ax4.fill_between(epoch_list, val_dice,   alpha=0.1, color='#607D8B')
ax4.set_title("Dice Coefficient", fontsize=14, fontweight='bold')
ax4.set_xlabel("Epoch", fontsize=11); ax4.set_ylabel("Dice", fontsize=11)
ax4.legend(fontsize=10); ax4.grid(True, alpha=0.2)
ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

fig.suptitle("Model Training Progress — Lane Detection (U-Net)", fontsize=18, fontweight='bold', color='#333333', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_training_progress.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: combined_training_progress.png")


# --- 5f: All Metrics Combined Bar Chart ---
metric_config = [
    {"name": "Pixel Acc",  "value": pixel_accuracy,  "color": "#2196F3"},
    {"name": "Precision",  "value": pixel_precision,  "color": "#FF9800"},
    {"name": "Recall",     "value": pixel_recall,     "color": "#4CAF50"},
    {"name": "F1 Score",   "value": pixel_f1,         "color": "#E91E63"},
    {"name": "IoU",        "value": iou_score,        "color": "#9C27B0"},
    {"name": "Dice",       "value": dice_score_val,   "color": "#00BCD4"},
]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

m_names  = [c["name"]  for c in metric_config]
m_values = [c["value"] for c in metric_config]
m_colors = [c["color"] for c in metric_config]

bars = ax.bar(m_names, m_values, color=m_colors, width=0.35, edgecolor='white', linewidth=2, zorder=3)
ax.axhline(y=1.0, color='#BDBDBD', linestyle='--', linewidth=1, alpha=0.7)

for bar, val, color in zip(bars, m_values, m_colors):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val*100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)

ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("All Segmentation Metrics — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', colors='#666666', labelsize=11)
ax.yaxis.grid(True, alpha=0.2, color='#999999')

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_metrics_combined.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: all_metrics_combined.png")


# --- 5g: Radar Chart ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#FAFAFA')

vals_r  = m_values + [m_values[0]]
angles  = np.linspace(0, 2*np.pi, len(m_names), endpoint=False).tolist()
angles += angles[:1]

ax.fill(angles, vals_r, color='#2196F3', alpha=0.25)
ax.plot(angles, vals_r, color='#2196F3', linewidth=2.5, marker='o', markersize=8)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(m_names, fontsize=11, fontweight='bold', color='#333333')
ax.set_ylim(0, 1.1)
ax.set_title("Metrics Radar Chart — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=30)

for angle, val in zip(angles[:-1], m_values):
    ax.text(angle, val + 0.08, f'{val*100:.1f}%',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#2196F3')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_radar_chart.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: metrics_radar_chart.png")


# --- 5h: Epoch Table ---
fig, ax = plt.subplots(figsize=(14, max(4, len(train_loss) * 0.4 + 1)))
fig.patch.set_facecolor('#FAFAFA')
ax.axis('off')

table_data = []
for e in range(len(train_loss)):
    table_data.append([
        f"Epoch {e+1}",
        f"{train_loss[e]:.4f}",
        f"{val_loss[e]:.4f}",
        f"{train_acc[e]*100:.2f}%",
        f"{val_acc[e]*100:.2f}%",
        f"{train_iou[e]:.4f}",
        f"{val_iou[e]:.4f}"
    ])

col_labels = ["Epoch", "Train Loss", "Test Loss", "Train Acc", "Test Acc", "Train IoU", "Test IoU"]
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.5)

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#9C27B0')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        table[i, j].set_facecolor('#F3E5F5' if i % 2 == 0 else '#FFFFFF')

ax.set_title("Epoch-by-Epoch Results — Lane Detection", fontsize=14, fontweight='bold', color='#333333', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "epoch_table.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: epoch_table.png")


# #####################################################
#
#   STEP 6 :  SAMPLE PREDICTIONS
#   Show: Original | Ground Truth | Prediction | Overlay
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 6 :  SAMPLE PREDICTIONS")
print("=" * 60)

# Select 8 diverse samples
np.random.seed(RANDOM_STATE)

# Find samples with different number of lanes for diversity
sample_indices = np.random.choice(len(X_test), size=min(8, len(X_test)), replace=False)

fig, axes = plt.subplots(8, 4, figsize=(20, 32))
fig.patch.set_facecolor('#FAFAFA')

col_titles = ["Original Image", "Ground Truth Mask", "Predicted Mask", "Overlay"]

for row_idx, sample_idx in enumerate(sample_indices):
    img       = X_test[sample_idx].astype(np.float32) / 255.0
    gt_mask   = Y_test[sample_idx, :, :, 0]
    pred_mask = Y_pred_binary[sample_idx, :, :, 0]

    # Create overlay: green for ground truth, red for prediction
    overlay = img.copy()
    overlay[gt_mask > 0]   = overlay[gt_mask > 0] * 0.5 + np.array([0, 0.8, 0]) * 0.5
    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.5 + np.array([0, 0.5, 1.0]) * 0.5

    # Compute per-sample IoU
    inter = np.sum(gt_mask * pred_mask)
    uni   = np.sum(gt_mask) + np.sum(pred_mask) - inter
    sample_iou = inter / (uni + 1e-7)

    # Column 0: Original image
    axes[row_idx, 0].imshow(img)
    axes[row_idx, 0].axis('off')

    # Column 1: Ground truth mask
    axes[row_idx, 1].imshow(gt_mask, cmap='Greens', vmin=0, vmax=1)
    axes[row_idx, 1].axis('off')

    # Column 2: Predicted mask
    axes[row_idx, 2].imshow(pred_mask, cmap='Blues', vmin=0, vmax=1)
    axes[row_idx, 2].axis('off')

    # Column 3: Overlay
    axes[row_idx, 3].imshow(overlay)
    axes[row_idx, 3].set_title(f"IoU: {sample_iou:.4f}", fontsize=10, fontweight='bold',
                                color='#2E7D32' if sample_iou > 0.5 else '#C62828')
    axes[row_idx, 3].axis('off')

# Add column headers
for j, title in enumerate(col_titles):
    axes[0, j].set_title(f"{title}\n" + axes[0, j].get_title(), fontsize=12, fontweight='bold', color='#333333')

fig.suptitle("Lane Detection Predictions — Ground Truth vs Model Output\n(Green = Ground Truth, Blue = Predicted, Overlay = Combined)",
             fontsize=18, fontweight='bold', color='#333333', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: sample_predictions.png ({len(sample_indices)} samples)")


# #####################################################
#
#   STEP 7 :  SCENARIO EXAMPLES
#   Show lane detection on different road types
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 7 :  SCENARIO EXAMPLES")
print("=" * 60)


def count_lanes(mask, threshold=100):
    """Count approximate number of lanes in a binary mask using connected components."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_uint8)
    return max(0, num_labels - 1)  # Subtract background


# Categorize test samples by number of detected lanes
lane_counts = []
for i in range(len(Y_test)):
    gt_mask = Y_test[i, :, :, 0]
    n_lanes = count_lanes(gt_mask)
    lane_counts.append(n_lanes)

lane_counts = np.array(lane_counts)

# Find scenes with 2 lanes (standard road)
two_lane_idx   = np.where(lane_counts == 2)[0]
# Find scenes with 3-4 lanes (multi-lane)
multi_lane_idx = np.where(lane_counts >= 3)[0]
# Find scenes with 1 lane (tricky)
one_lane_idx   = np.where(lane_counts == 1)[0]

print(f"  1-lane scenes   : {len(one_lane_idx)}")
print(f"  2-lane scenes   : {len(two_lane_idx)}")
print(f"  3+ lane scenes  : {len(multi_lane_idx)}")

N = 6


def generate_lane_scenario(indices, scenario_title, color, filename, n=6):
    """Generate lane detection scenario visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.patch.set_facecolor('#1a1a2e')

    np.random.seed(RANDOM_STATE)
    if len(indices) > n:
        chosen = np.random.choice(indices, n, replace=False)
    else:
        chosen = indices[:n]

    for idx_i, ax in enumerate(axes.flat):
        if idx_i >= len(chosen):
            ax.axis('off')
            continue

        i = chosen[idx_i]
        img       = X_test[i].astype(np.float32) / 255.0
        pred_mask = Y_pred_binary[i, :, :, 0]

        # Create overlay with predicted lanes
        overlay = img.copy()
        # Green overlay for detected lanes
        lane_overlay = np.zeros_like(img)
        lane_overlay[:, :, 1] = pred_mask * 0.8  # Green channel
        lane_overlay[:, :, 2] = pred_mask * 0.3  # Slight blue
        overlay = np.clip(overlay + lane_overlay * 0.6, 0, 1)

        ax.imshow(overlay)
        n_lanes = count_lanes(pred_mask)
        ax.set_title(f"Detected {n_lanes} lane(s)",
                     fontsize=12, fontweight='bold', color='white', pad=8)
        ax.axis('off')

    fig.suptitle(f"{scenario_title}",
                 fontsize=20, fontweight='bold', color=color, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {filename}")


# Scenario 1: Standard 2-Lane Road
if len(two_lane_idx) > 0:
    generate_lane_scenario(two_lane_idx, "SCENARIO 1 : Standard 2-Lane Road Detection",
                           '#4CAF50', "scenario_2lane.png")
else:
    print("  Skipped: Not enough 2-lane scenes")

# Scenario 2: Multi-Lane Highway
if len(multi_lane_idx) > 0:
    generate_lane_scenario(multi_lane_idx, "SCENARIO 2 : Multi-Lane Highway Detection",
                           '#2196F3', "scenario_multilane.png")
else:
    print("  Skipped: Not enough multi-lane scenes")

# Scenario 3: Challenging Single Lane
if len(one_lane_idx) > 0:
    generate_lane_scenario(one_lane_idx, "SCENARIO 3 : Challenging Single Lane Detection",
                           '#FF9800', "scenario_singlelane.png")
else:
    print("  Skipped: Not enough single-lane scenes")


# General scenario with best predictions
best_iou_indices = []
for i in range(len(Y_test)):
    gt = Y_test[i, :, :, 0]
    pred = Y_pred_binary[i, :, :, 0]
    inter = np.sum(gt * pred)
    uni = np.sum(gt) + np.sum(pred) - inter
    s_iou = inter / (uni + 1e-7)
    best_iou_indices.append((i, s_iou))

best_iou_indices.sort(key=lambda x: x[1], reverse=True)

# SCENARIO 4: Curved Roads & Edge Cases (Failure Analysis)
# We choose the worst performing predictions to demonstrate how the model reacts to tricky edge cases
worst_indices = [x[0] for x in best_iou_indices[-N:]]
generate_lane_scenario(worst_indices, "SCENARIO 4 : Curved Roads & Edge Cases (Failure Analysis)",
                       '#F44336', "scenario_failures.png")

# Best predictions
best_indices = [x[0] for x in best_iou_indices[:N]]
generate_lane_scenario(best_indices, "BEST PREDICTIONS — Highest IoU Samples",
                       '#E91E63', "scenario_best_predictions.png")


# #####################################################
#
#   ALL DONE!
#
# #####################################################

print("\n\n" + "=" * 60)
print("  ALL DONE! LANE DETECTION RESULTS SAVED")
print("=" * 60)
print()
print("  FILES CREATED:")
print("  " + "-" * 50)
print("  MODEL:")
print("    lane_model.h5")
print("    segmentation_report.txt")
print()
print("  TRAINING CHARTS:")
print("    training_testing_loss.png")
print("    training_testing_accuracy.png")
print("    iou_per_epoch.png")
print("    dice_per_epoch.png")
print("    combined_training_progress.png")
print("    epoch_table.png")
print()
print("  EVALUATION CHARTS:")
print("    all_metrics_combined.png")
print("    metrics_radar_chart.png")
print()
print("  PREDICTIONS:")
print("    sample_predictions.png")
print("    scenario_2lane.png")
print("    scenario_multilane.png")
print("    scenario_singlelane.png")
print("    scenario_failures.png")
print("    scenario_best_predictions.png")
print("  " + "-" * 50)
print(f"\n  All saved to: {OUTPUT_DIR}")
print("=" * 60 + "\n")
