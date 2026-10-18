import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    Lambda,
    Activation,
    concatenate
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


# =============================================================
#
#   COMPLETE AUTONOMOUS DRIVING SYSTEM
#   ===================================
#
#   PART A :  OBSTACLE DETECTION     (CNN Classification - KITTI)
#   PART B :  STEERING PREDICTION    (CNN Regression - Simulator)
#   PART C :  LANE DETECTION         (U-Net Segmentation - TUSimple)
#
#   Each part saves results to a separate folder under:
#       D:\New parttss all comined results\
#           ├── obstacle_detection\
#           ├── steering_prediction\
#           └── lane_detection\
#
# =============================================================


# =====================================================
# GLOBAL CONFIGURATION
# =====================================================

RANDOM_STATE = 42
DPI          = 300      # High resolution for all figures

# --- Part A: Obstacle Detection ---
OBS_IMAGE_DIR  = r"D:\Finalyear pro\Images\data_object_image_2\training\image_2"
OBS_LABEL_DIR  = r"D:\Finalyear pro\Labels\data_object_label_2\training\label_2"
OBS_OUTPUT_DIR = r"D:\New parttss all comined results\obstacle_detection"
OBS_IMAGE_SIZE = 64
OBS_BATCH      = 64
OBS_EPOCHS     = 20
OBS_SPLIT      = 0.2
OBS_CLASSES    = ["Car", "Cyclist", "Pedestrian", "Truck", "Van", "Tram"]

# --- Part B: Steering Prediction ---
STEER_JUNGLE_DIR = r"D:\Finalyear pro\steering prediction\archive\self_driving_car_dataset_jungle"
STEER_MAKE_DIR   = r"D:\Finalyear pro\steering prediction\archive\self_driving_car_dataset_make"
STEER_OUTPUT_DIR = r"D:\New parttss all comined results\steering_prediction"
STEER_IMAGE_H    = 66
STEER_IMAGE_W    = 200
STEER_BATCH      = 64
STEER_EPOCHS     = 20
STEER_SPLIT      = 0.2

# --- Part C: Lane Detection ---
TUSIMPLE_DIR    = r"D:\Finalyear pro\lane detection\TUSimple"
LANE_TRAIN_DIR  = os.path.join(TUSIMPLE_DIR, "train_set")
LANE_OUTPUT_DIR = r"D:\New parttss all comined results\lane_detection"
LANE_IMAGE_H    = 256
LANE_IMAGE_W    = 512
LANE_BATCH      = 8
LANE_EPOCHS     = 20
LANE_SPLIT      = 0.2
LANE_LINE_WIDTH = 8

# Create all output directories
os.makedirs(OBS_OUTPUT_DIR, exist_ok=True)
os.makedirs(STEER_OUTPUT_DIR, exist_ok=True)
os.makedirs(LANE_OUTPUT_DIR, exist_ok=True)


# =====================================================
# VERIFY ALL INPUT FOLDERS
# =====================================================

print("=" * 70)
print("  COMPLETE AUTONOMOUS DRIVING SYSTEM")
print("  Obstacle Detection + Steering Prediction + Lane Detection")
print("=" * 70)
print("\n  Verifying input folders...")

all_ok = True
for d, name in [
    (OBS_IMAGE_DIR,    "Obstacle - KITTI Images"),
    (OBS_LABEL_DIR,    "Obstacle - KITTI Labels"),
    (STEER_JUNGLE_DIR, "Steering - Jungle dataset"),
    (STEER_MAKE_DIR,   "Steering - Make dataset"),
    (LANE_TRAIN_DIR,   "Lane - TUSimple train"),
]:
    if os.path.exists(d):
        print(f"    ✓  {name}")
    else:
        print(f"    ✗  {name}  NOT FOUND: {d}")
        all_ok = False

if not all_ok:
    print("\n  ERROR: Some folders are missing.")
    exit()

print("\n  All folders verified! Starting...\n")
global_start = time.time()


# *************************************************************
# *************************************************************
#
#   PART A :  OBSTACLE DETECTION
#             CNN Classification on KITTI Dataset
#
# *************************************************************
# *************************************************************

print("\n" + "#" * 70)
print("#" * 70)
print("##")
print("##   PART A :  OBSTACLE DETECTION (CNN Classification)")
print("##   Dataset : KITTI Object Detection")
print("##   Classes : Car, Cyclist, Pedestrian, Truck, Van, Tram")
print(f"##   Output  : {OBS_OUTPUT_DIR}")
print("##")
print("#" * 70)
print("#" * 70)


# #####################################################
#   A-STEP 1 :  LOAD KITTI DATASET
# #####################################################

print("\n" + "=" * 70)
print("  A-STEP 1 :  LOADING KITTI DATASET")
print("=" * 70)

obs_start = time.time()

obs_images = []
obs_labels = []

image_files = sorted([f for f in os.listdir(OBS_IMAGE_DIR) if f.endswith((".png", ".jpg"))])
print(f"  Found {len(image_files)} images")

for i, file in enumerate(image_files):
    if (i + 1) % 500 == 0 or i == 0:
        print(f"  Processing image {i + 1}/{len(image_files)}...")

    img_path   = os.path.join(OBS_IMAGE_DIR, file)
    label_path = os.path.join(OBS_LABEL_DIR, os.path.splitext(file)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    height, width = img.shape[:2]

    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue
        obj = parts[0]
        if obj not in OBS_CLASSES:
            continue

        x1 = max(0,      int(float(parts[4])))
        y1 = max(0,      int(float(parts[5])))
        x2 = min(width,  int(float(parts[6])))
        y2 = min(height, int(float(parts[7])))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (OBS_IMAGE_SIZE, OBS_IMAGE_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32) / 255.0

        obs_images.append(crop)
        obs_labels.append(obj)

obs_load_time = time.time() - obs_start

X_obs = np.array(obs_images, dtype=np.float32)
y_obs = np.array(obs_labels)
del obs_images

print(f"\n  Loaded in {obs_load_time:.1f} seconds")
print(f"  Total objects : {X_obs.shape[0]}")
print(f"  Image shape   : {X_obs.shape[1:]}")
print(f"  Memory        : {X_obs.nbytes / (1024**3):.2f} GB")

if len(X_obs) == 0:
    print("\n  ERROR: No valid obstacle data found.")
    exit()

unique, counts = np.unique(y_obs, return_counts=True)
print("\n  CLASS DISTRIBUTION:")
for cls, cnt in zip(unique, counts):
    print(f"    {cls:<15s} : {cnt:>6d}  ({cnt/len(y_obs)*100:5.1f}%)")


# #####################################################
#   A-STEP 2 :  PREPARE OBSTACLE DATA
# #####################################################

print("\n" + "=" * 70)
print("  A-STEP 2 :  PREPARING OBSTACLE DATA")
print("=" * 70)

encoder = LabelEncoder()
y_obs_encoded     = encoder.fit_transform(y_obs)
y_obs_categorical = to_categorical(y_obs_encoded)
num_classes        = len(encoder.classes_)

print(f"  Classes ({num_classes}): {list(encoder.classes_)}")

X_obs_train, X_obs_test, y_obs_train, y_obs_test = train_test_split(
    X_obs, y_obs_categorical,
    test_size=OBS_SPLIT,
    random_state=RANDOM_STATE,
    stratify=y_obs_encoded
)
del X_obs

print(f"  Training : {len(X_obs_train)}")
print(f"  Testing  : {len(X_obs_test)}")

datagen = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, zoom_range=0.1
)
datagen.fit(X_obs_train)


# #####################################################
#   A-STEP 3 :  BUILD & TRAIN OBSTACLE CNN
# #####################################################

print("\n" + "=" * 70)
print("  A-STEP 3 :  BUILDING & TRAINING OBSTACLE CNN")
print("=" * 70)

obs_model = Sequential()
obs_model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(OBS_IMAGE_SIZE,OBS_IMAGE_SIZE,3)))
obs_model.add(BatchNormalization())
obs_model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
obs_model.add(BatchNormalization())
obs_model.add(MaxPooling2D((2,2)))
obs_model.add(Dropout(0.25))

obs_model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
obs_model.add(BatchNormalization())
obs_model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
obs_model.add(BatchNormalization())
obs_model.add(MaxPooling2D((2,2)))
obs_model.add(Dropout(0.25))

obs_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
obs_model.add(BatchNormalization())
obs_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
obs_model.add(BatchNormalization())
obs_model.add(MaxPooling2D((2,2)))
obs_model.add(Dropout(0.25))

obs_model.add(Flatten())
obs_model.add(Dense(256, activation='relu'))
obs_model.add(BatchNormalization())
obs_model.add(Dropout(0.5))
obs_model.add(Dense(128, activation='relu'))
obs_model.add(Dropout(0.5))
obs_model.add(Dense(num_classes, activation='softmax'))

obs_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
obs_model.summary()

obs_early = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
obs_lr    = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n  Training started...\n")
obs_train_start = time.time()

obs_history = obs_model.fit(
    datagen.flow(X_obs_train, y_obs_train, batch_size=OBS_BATCH),
    validation_data=(X_obs_test, y_obs_test),
    epochs=OBS_EPOCHS,
    callbacks=[obs_early, obs_lr],
    verbose=1
)

obs_train_time = time.time() - obs_train_start
print(f"\n  Training done in {obs_train_time:.1f} seconds")

obs_model.save(os.path.join(OBS_OUTPUT_DIR, "kitti_obstacle_model.h5"))
print(f"  Model saved: kitti_obstacle_model.h5")

o_train_acc  = obs_history.history['accuracy']
o_val_acc    = obs_history.history['val_accuracy']
o_train_loss = obs_history.history['loss']
o_val_loss   = obs_history.history['val_loss']
o_epoch_list = range(1, len(o_train_acc) + 1)


# #####################################################
#   A-STEP 4 :  EVALUATE OBSTACLE MODEL
# #####################################################

print("\n" + "=" * 70)
print("  A-STEP 4 :  OBSTACLE MODEL EVALUATION")
print("=" * 70)

o_loss, o_acc_eval = obs_model.evaluate(X_obs_test, y_obs_test, verbose=0)
o_y_pred         = obs_model.predict(X_obs_test, verbose=0)
o_y_pred_classes = np.argmax(o_y_pred, axis=1)
o_y_true_classes = np.argmax(y_obs_test, axis=1)

o_accuracy  = accuracy_score(o_y_true_classes, o_y_pred_classes)
o_precision = precision_score(o_y_true_classes, o_y_pred_classes, average='macro', zero_division=0)
o_recall    = recall_score(o_y_true_classes, o_y_pred_classes, average='macro', zero_division=0)
o_f1        = f1_score(o_y_true_classes, o_y_pred_classes, average='macro', zero_division=0)

per_class_precision = precision_score(o_y_true_classes, o_y_pred_classes, average=None, zero_division=0)
per_class_recall    = recall_score(o_y_true_classes, o_y_pred_classes, average=None, zero_division=0)
per_class_f1        = f1_score(o_y_true_classes, o_y_pred_classes, average=None, zero_division=0)

print(f"\n  Test Loss       : {o_loss:.4f}")
print(f"  Test Accuracy   : {o_accuracy*100:.2f}%")
print(f"  Macro Precision : {o_precision*100:.2f}%")
print(f"  Macro Recall    : {o_recall*100:.2f}%")
print(f"  Macro F1 Score  : {o_f1*100:.2f}%")

print("\n  PER-CLASS BREAKDOWN:")
print(f"  {'Class':<15} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}")
print("  " + "-" * 55)
for i, cls in enumerate(encoder.classes_):
    print(f"  {cls:<15} {per_class_precision[i]*100:>11.2f}% {per_class_recall[i]*100:>11.2f}% {per_class_f1[i]*100:>11.2f}%")

report = classification_report(o_y_true_classes, o_y_pred_classes, target_names=encoder.classes_)
print(f"\n{report}")

with open(os.path.join(OBS_OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write("KITTI OBSTACLE CLASSIFICATION - CNN RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("AVERAGING METHOD: MACRO\n\n")
    f.write(f"Test Accuracy   : {o_accuracy*100:.2f}%\n")
    f.write(f"Macro Precision : {o_precision*100:.2f}%\n")
    f.write(f"Macro Recall    : {o_recall*100:.2f}%\n")
    f.write(f"Macro F1 Score  : {o_f1*100:.2f}%\n\n")
    f.write("PER-CLASS BREAKDOWN:\n")
    f.write(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}\n")
    f.write("-" * 55 + "\n")
    for i, cls in enumerate(encoder.classes_):
        f.write(f"{cls:<15} {per_class_precision[i]*100:>11.2f}% {per_class_recall[i]*100:>11.2f}% {per_class_f1[i]*100:>11.2f}%\n")
    f.write("\nFULL CLASSIFICATION REPORT:\n")
    f.write(report)
print("  Saved: classification_report.txt")


# #####################################################
#   A-STEP 5 :  OBSTACLE CHARTS
# #####################################################

print("\n" + "=" * 70)
print("  A-STEP 5 :  GENERATING OBSTACLE CHARTS")
print("=" * 70)

# --- Confusion Matrix ---
cm = confusion_matrix(o_y_true_classes, o_y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_,
            linewidths=0.5, linecolor='white')
plt.xlabel("Predicted", fontsize=12); plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - KITTI Object Classification", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OBS_OUTPUT_DIR, "confusion_matrix.png"), dpi=DPI)
plt.close()
print("  Saved: confusion_matrix.png")

# --- Individual + Combined Metric Charts ---
obs_chart_cfg = [
    {"name": "Accuracy",  "value": o_accuracy,  "color": "#2196F3"},
    {"name": "Precision", "value": o_precision, "color": "#FF9800"},
    {"name": "Recall",    "value": o_recall,    "color": "#4CAF50"},
    {"name": "F1 Score",  "value": o_f1,        "color": "#E91E63"},
]

for cfg in obs_chart_cfg:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
    ax.bar([cfg["name"]], [cfg["value"]], color=cfg["color"], width=0.25, edgecolor='white', linewidth=2, zorder=3)
    ax.axhline(y=1.0, color='#BDBDBD', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0, cfg["value"] + 0.03, f'{cfg["value"]*100:.2f}%', ha='center', va='bottom',
            fontsize=22, fontweight='bold', color=cfg["color"])
    ax.set_ylim(0, 1.2); ax.set_ylabel("Score", fontsize=14, fontweight='bold', color='#333333')
    ax.set_title(cfg["name"], fontsize=20, fontweight='bold', color=cfg["color"], pad=20)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0]); ax.set_yticklabels(['0%','20%','40%','60%','80%','100%'])
    ax.yaxis.grid(True, alpha=0.2)
    plt.tight_layout()
    fname = f"{cfg['name'].lower().replace(' ', '_')}_chart.png"
    plt.savefig(os.path.join(OBS_OUTPUT_DIR, fname), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}  =  {cfg['value']*100:.2f}%")

# Combined metrics
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
om_names  = [c["name"] for c in obs_chart_cfg]
om_values = [c["value"] for c in obs_chart_cfg]
om_colors = [c["color"] for c in obs_chart_cfg]
bars = ax.bar(om_names, om_values, color=om_colors, width=0.30, edgecolor='white', linewidth=2, zorder=3)
ax.axhline(y=1.0, color='#BDBDBD', linestyle='--', linewidth=1, alpha=0.7)
for bar, val, color in zip(bars, om_values, om_colors):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.025, f'{val*100:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold', color=color)
ax.set_ylim(0, 1.2); ax.set_ylabel("Score (%)", fontsize=14, fontweight='bold')
ax.set_title("All Performance Metrics - KITTI Classification\n(Macro Average)", fontsize=15, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0]); ax.set_yticklabels(['0%','20%','40%','60%','80%','100%'])
ax.yaxis.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OBS_OUTPUT_DIR, "all_metrics_combined.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: all_metrics_combined.png")

# Per-class bar chart
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
x = np.arange(len(encoder.classes_))
bw = 0.22
b1 = ax.bar(x-bw, per_class_precision*100, bw, label='Precision', color='#FF9800', edgecolor='white', zorder=3)
b2 = ax.bar(x,    per_class_recall*100,    bw, label='Recall',    color='#4CAF50', edgecolor='white', zorder=3)
b3 = ax.bar(x+bw, per_class_f1*100,       bw, label='F1 Score',  color='#E91E63', edgecolor='white', zorder=3)
ax.set_xticks(x); ax.set_xticklabels(encoder.classes_, fontsize=12, fontweight='bold')
ax.set_ylabel("Score (%)", fontsize=13, fontweight='bold')
ax.set_title("Per-Class Precision, Recall & F1 - KITTI CNN", fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12); ax.set_ylim(0, 120); ax.yaxis.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for bar in list(b1)+list(b2)+list(b3):
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+1.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OBS_OUTPUT_DIR, "per_class_metrics.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: per_class_metrics.png")

# Radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#FAFAFA')
vr = om_values + [om_values[0]]
angs = np.linspace(0, 2*np.pi, len(om_names), endpoint=False).tolist() + [0]
ax.fill(angs, vr, color='#2196F3', alpha=0.25)
ax.plot(angs, vr, color='#2196F3', linewidth=2.5, marker='o', markersize=8)
ax.set_xticks(angs[:-1]); ax.set_xticklabels(om_names, fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_title("Metrics Radar Chart (Macro Average)", fontsize=16, fontweight='bold', pad=30)
for a, v in zip(angs[:-1], om_values):
    ax.text(a, v+0.08, f'{v*100:.1f}%', ha='center', fontsize=11, fontweight='bold', color='#2196F3')
plt.tight_layout()
plt.savefig(os.path.join(OBS_OUTPUT_DIR, "metrics_radar_chart.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: metrics_radar_chart.png")

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#FAFAFA')
ax1.set_facecolor('#FAFAFA')
ax1.plot(o_epoch_list, o_train_loss, color='#E91E63', linewidth=2.5, marker='o', markersize=5, label='Train Loss')
ax1.plot(o_epoch_list, o_val_loss,   color='#2196F3', linewidth=2.5, marker='s', markersize=5, label='Test Loss')
ax1.fill_between(o_epoch_list, o_train_loss, alpha=0.1, color='#E91E63')
ax1.fill_between(o_epoch_list, o_val_loss,   alpha=0.1, color='#2196F3')
ax1.set_title("Training & Testing Loss", fontsize=14, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=12, fontweight='bold'); ax1.set_ylabel("Loss", fontsize=12, fontweight='bold')
ax1.legend(fontsize=11); ax1.grid(True, alpha=0.2)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.set_facecolor('#FAFAFA')
ax2.plot(o_epoch_list, o_train_acc, color='#4CAF50', linewidth=2.5, marker='o', markersize=5, label='Train Acc')
ax2.plot(o_epoch_list, o_val_acc,   color='#FF9800', linewidth=2.5, marker='s', markersize=5, label='Test Acc')
ax2.fill_between(o_epoch_list, o_train_acc, alpha=0.1, color='#4CAF50')
ax2.fill_between(o_epoch_list, o_val_acc,   alpha=0.1, color='#FF9800')
ax2.set_title("Training & Testing Accuracy", fontsize=14, fontweight='bold')
ax2.set_xlabel("Epoch", fontsize=12, fontweight='bold'); ax2.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.2)
ax2.set_yticks([0,0.2,0.4,0.6,0.8,1.0]); ax2.set_yticklabels(['0%','20%','40%','60%','80%','100%'])
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.suptitle("Model Training Progress - KITTI CNN", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OBS_OUTPUT_DIR, "combined_loss_accuracy.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: combined_loss_accuracy.png")

# Epoch table
fig, ax = plt.subplots(figsize=(12, max(4, len(o_train_acc)*0.4+1)))
fig.patch.set_facecolor('#FAFAFA'); ax.axis('off')
tbl = []
for e in range(len(o_train_acc)):
    tbl.append([f"Epoch {e+1}", f"{o_train_loss[e]:.4f}", f"{o_val_loss[e]:.4f}",
                f"{o_train_acc[e]*100:.2f}%", f"{o_val_acc[e]*100:.2f}%"])
cols = ["Epoch", "Train Loss", "Test Loss", "Train Acc", "Test Acc"]
t = ax.table(cellText=tbl, colLabels=cols, cellLoc='center', loc='center')
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.0, 1.5)
for j in range(len(cols)):
    t[0,j].set_facecolor('#2196F3'); t[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(tbl)+1):
    for j in range(len(cols)):
        t[i,j].set_facecolor('#E3F2FD' if i%2==0 else '#FFFFFF')
ax.set_title("Epoch-by-Epoch Results — Obstacle Detection", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OBS_OUTPUT_DIR, "epoch_table.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: epoch_table.png")


# #####################################################
#   A-STEP 6 :  OBSTACLE SAMPLE PREDICTIONS
# #####################################################

print("\n" + "=" * 70)
print("  A-STEP 6 :  OBSTACLE SAMPLE PREDICTIONS")
print("=" * 70)

correct_idx = [i for i in range(len(o_y_pred_classes)) if o_y_pred_classes[i] == o_y_true_classes[i]]
wrong_idx   = [i for i in range(len(o_y_pred_classes)) if o_y_pred_classes[i] != o_y_true_classes[i]]

sel_correct = []
for ci in range(num_classes):
    cc = [i for i in correct_idx if o_y_true_classes[i] == ci]
    if cc:
        np.random.seed(RANDOM_STATE)
        sel_correct.extend(np.random.choice(cc, min(2, len(cc)), replace=False).tolist())

sel_wrong = []
for ci in range(num_classes):
    cw = [i for i in wrong_idx if o_y_true_classes[i] == ci]
    if cw:
        np.random.seed(RANDOM_STATE + ci)
        sel_wrong.extend(np.random.choice(cw, min(1, len(cw)), replace=False).tolist())

np.random.seed(RANDOM_STATE)
if len(sel_correct) > 12: sel_correct = list(np.random.choice(sel_correct, 12, replace=False))
if len(sel_wrong) > 4:    sel_wrong   = list(np.random.choice(sel_wrong, 4, replace=False))

sample_sel = sel_correct + sel_wrong
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.patch.set_facecolor('#FAFAFA')
for i, ax in enumerate(axes.flat):
    if i >= len(sample_sel): ax.axis('off'); continue
    idx = sample_sel[i]
    ax.imshow(X_obs_test[idx])
    pl = encoder.inverse_transform([o_y_pred_classes[idx]])[0]
    tl = encoder.inverse_transform([o_y_true_classes[idx]])[0]
    ok = pl == tl
    bc = '#2E7D32' if ok else '#C62828'
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor(bc); sp.set_linewidth(4)
    st = "✓ CORRECT" if ok else "✗ WRONG"
    ax.set_title(f"Predicted: {pl}\nActual: {tl}\n{st}", fontsize=11, color=bc, fontweight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Obstacle Predictions — Diverse Classes + Failures\n(Green=Correct, Red=Wrong)",
             fontsize=16, fontweight='bold', color='#333333', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OBS_OUTPUT_DIR, "sample_predictions.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  Saved: sample_predictions.png ({len(sel_correct)} correct + {len(sel_wrong)} failures)")

print("\n" + "=" * 70)
print("  ✓  PART A COMPLETE — Obstacle Detection Done!")
print("=" * 70)

# Free obstacle memory
del X_obs_train, X_obs_test, y_obs_train, y_obs_test, o_y_pred
del obs_model


# *************************************************************
# *************************************************************
#
#   PART B :  STEERING WHEEL PREDICTION
#             CNN Regression using NVIDIA Architecture
#
# *************************************************************
# *************************************************************

print("\n\n" + "#" * 70)
print("#" * 70)
print("##")
print("##   PART B :  STEERING WHEEL PREDICTION (CNN Regression)")
print("##   Dataset : Self-Driving Car Simulator (Jungle + Make)")
print(f"##   Output  : {STEER_OUTPUT_DIR}")
print("##")
print("#" * 70)
print("#" * 70)


# #####################################################
#   B-STEP 1 :  LOAD STEERING DATASET
# #####################################################

print("\n" + "=" * 70)
print("  B-STEP 1 :  LOADING STEERING DATASET")
print("=" * 70)

steer_start = time.time()

def load_driving_data(dataset_dir, dataset_name):
    csv_path = os.path.join(dataset_dir, "driving_log.csv")
    img_dir  = os.path.join(dataset_dir, "IMG")
    print(f"\n  Loading {dataset_name}...")
    df = pd.read_csv(csv_path, header=None,
                     names=["center","left","right","steering","throttle","reverse","speed"])
    images, angles, skipped = [], [], 0
    for idx, row in df.iterrows():
        if (idx+1) % 2000 == 0 or idx == 0: print(f"    Processing {idx+1}/{len(df)}...")
        fn = os.path.basename(row["center"].strip())
        ip = os.path.join(img_dir, fn)
        if not os.path.exists(ip): skipped += 1; continue
        img = cv2.imread(ip)
        if img is None: skipped += 1; continue
        img = img[60:135, :]
        img = cv2.resize(img, (STEER_IMAGE_W, STEER_IMAGE_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images.append(img); angles.append(float(row["steering"]))
    print(f"    Loaded: {len(images)} | Skipped: {skipped}")
    return images, angles

imgs_j, angs_j = load_driving_data(STEER_JUNGLE_DIR, "Jungle")
imgs_m, angs_m = load_driving_data(STEER_MAKE_DIR,   "Make")

X_steer = np.array(imgs_j + imgs_m, dtype=np.float32)
y_steer = np.array(angs_j + angs_m, dtype=np.float32)
del imgs_j, imgs_m, angs_j, angs_m

print(f"\n  Combined: {len(X_steer)} samples | Range: [{y_steer.min():.4f}, {y_steer.max():.4f}]")


# #####################################################
#   B-STEP 2 :  PREPARE STEERING DATA
# #####################################################

print("\n" + "=" * 70)
print("  B-STEP 2 :  PREPARING STEERING DATA")
print("=" * 70)

X_st_train, X_st_test, y_st_train, y_st_test = train_test_split(
    X_steer, y_steer, test_size=STEER_SPLIT, random_state=RANDOM_STATE)
del X_steer
print(f"  Training: {len(X_st_train)} | Testing: {len(X_st_test)}")


# #####################################################
#   B-STEP 3 :  BUILD & TRAIN NVIDIA CNN
# #####################################################

print("\n" + "=" * 70)
print("  B-STEP 3 :  BUILDING & TRAINING NVIDIA CNN")
print("=" * 70)

steer_model = Sequential()
steer_model.add(Lambda(lambda x: x - 0.5, input_shape=(STEER_IMAGE_H, STEER_IMAGE_W, 3)))
steer_model.add(Conv2D(24, (5,5), strides=(2,2), activation='elu')); steer_model.add(BatchNormalization())
steer_model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu')); steer_model.add(BatchNormalization())
steer_model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu')); steer_model.add(BatchNormalization())
steer_model.add(Conv2D(64, (3,3), activation='elu')); steer_model.add(BatchNormalization())
steer_model.add(Conv2D(64, (3,3), activation='elu')); steer_model.add(BatchNormalization())
steer_model.add(Flatten())
steer_model.add(Dense(100, activation='elu')); steer_model.add(Dropout(0.3))
steer_model.add(Dense(50, activation='elu'));  steer_model.add(Dropout(0.3))
steer_model.add(Dense(10, activation='elu'))
steer_model.add(Dense(1))

steer_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
steer_model.summary()

print("\n  Training started...\n")
st_train_start = time.time()
st_history = steer_model.fit(
    X_st_train, y_st_train, validation_data=(X_st_test, y_st_test),
    epochs=STEER_EPOCHS, batch_size=STEER_BATCH,
    callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)],
    verbose=1)

st_train_time = time.time() - st_train_start
print(f"\n  Training done in {st_train_time:.1f} seconds")
steer_model.save(os.path.join(STEER_OUTPUT_DIR, "steering_model.h5"))
print("  Model saved: steering_model.h5")

s_tl = st_history.history['loss'];     s_vl = st_history.history['val_loss']
s_tm = st_history.history['mae'];      s_vm = st_history.history['val_mae']
s_ep = range(1, len(s_tl) + 1)


# #####################################################
#   B-STEP 4 :  EVALUATE STEERING MODEL
# #####################################################

print("\n" + "=" * 70)
print("  B-STEP 4 :  STEERING MODEL EVALUATION")
print("=" * 70)

y_st_pred = steer_model.predict(X_st_test, verbose=0).flatten()
s_mae  = mean_absolute_error(y_st_test, y_st_pred)
s_mse  = mean_squared_error(y_st_test, y_st_pred)
s_rmse = np.sqrt(s_mse)
s_r2   = r2_score(y_st_test, y_st_pred)

print(f"\n  MAE: {s_mae:.6f} | MSE: {s_mse:.6f} | RMSE: {s_rmse:.6f} | R²: {s_r2:.6f}")

with open(os.path.join(STEER_OUTPUT_DIR, "regression_report.txt"), "w") as f:
    f.write("STEERING WHEEL PREDICTION - CNN REGRESSION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total: {len(X_st_train)+len(X_st_test)} | Train: {len(X_st_train)} | Test: {len(X_st_test)}\n\n")
    f.write(f"MAE: {s_mae:.6f}\nMSE: {s_mse:.6f}\nRMSE: {s_rmse:.6f}\nR²: {s_r2:.6f}\n")
    f.write(f"\nTraining Time: {st_train_time:.1f}s | Epochs: {len(s_tl)}\n")
print("  Saved: regression_report.txt")


# #####################################################
#   B-STEP 5 :  STEERING CHARTS
# #####################################################

print("\n" + "=" * 70)
print("  B-STEP 5 :  GENERATING STEERING CHARTS")
print("=" * 70)

# Loss + MAE combined
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#FAFAFA')
ax1.set_facecolor('#FAFAFA')
ax1.plot(s_ep, s_tl, color='#E91E63', linewidth=2.5, marker='o', markersize=5, label='Train Loss')
ax1.plot(s_ep, s_vl, color='#2196F3', linewidth=2.5, marker='s', markersize=5, label='Test Loss')
ax1.fill_between(s_ep, s_tl, alpha=0.1, color='#E91E63'); ax1.fill_between(s_ep, s_vl, alpha=0.1, color='#2196F3')
ax1.set_title("Loss (MSE)", fontsize=14, fontweight='bold'); ax1.legend(fontsize=11); ax1.grid(True, alpha=0.2)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax2.set_facecolor('#FAFAFA')
ax2.plot(s_ep, s_tm, color='#4CAF50', linewidth=2.5, marker='o', markersize=5, label='Train MAE')
ax2.plot(s_ep, s_vm, color='#FF9800', linewidth=2.5, marker='s', markersize=5, label='Test MAE')
ax2.fill_between(s_ep, s_tm, alpha=0.1, color='#4CAF50'); ax2.fill_between(s_ep, s_vm, alpha=0.1, color='#FF9800')
ax2.set_title("MAE", fontsize=14, fontweight='bold'); ax2.legend(fontsize=11); ax2.grid(True, alpha=0.2)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
fig.suptitle("Training Progress — Steering Prediction", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(STEER_OUTPUT_DIR, "combined_loss_mae.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: combined_loss_mae.png")

# Actual vs Predicted
fig, ax = plt.subplots(figsize=(10, 8)); fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.scatter(y_st_test, y_st_pred, alpha=0.3, s=10, color='#2196F3', edgecolors='none')
mn, mx = min(y_st_test.min(), y_st_pred.min()), max(y_st_test.max(), y_st_pred.max())
ax.plot([mn, mx], [mn, mx], '--', linewidth=2, color='#E91E63', label='Perfect')
ax.set_xlabel("Actual", fontsize=14, fontweight='bold'); ax.set_ylabel("Predicted", fontsize=14, fontweight='bold')
ax.set_title(f"Actual vs Predicted Steering (R²={s_r2:.4f})", fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(STEER_OUTPUT_DIR, "actual_vs_predicted.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: actual_vs_predicted.png")

# Error + Distribution
s_err = y_st_pred - y_st_test
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)); fig.patch.set_facecolor('#FAFAFA')
ax1.set_facecolor('#FAFAFA')
ax1.hist(y_st_test, bins=50, alpha=0.6, color='#2196F3', edgecolor='white', label='Actual', density=True)
ax1.hist(y_st_pred, bins=50, alpha=0.6, color='#FF9800', edgecolor='white', label='Predicted', density=True)
ax1.set_title("Steering Distribution", fontsize=14, fontweight='bold'); ax1.legend(fontsize=11)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax2.set_facecolor('#FAFAFA')
ax2.hist(s_err, bins=50, alpha=0.7, color='#4CAF50', edgecolor='white')
ax2.axvline(0, color='#E91E63', linestyle='--', linewidth=2, label='Zero Error')
ax2.set_title("Error Distribution", fontsize=14, fontweight='bold'); ax2.legend(fontsize=11)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
fig.suptitle("Steering Analysis", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(STEER_OUTPUT_DIR, "steering_error_analysis.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: steering_error_analysis.png")

# All metrics bar
scfg = [{"name":"MAE","value":s_mae,"color":"#2196F3"},{"name":"MSE","value":s_mse,"color":"#FF9800"},
        {"name":"RMSE","value":s_rmse,"color":"#4CAF50"},{"name":"R²","value":s_r2,"color":"#E91E63"}]
fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
sn = [c["name"] for c in scfg]; sv = [c["value"] for c in scfg]; sc = [c["color"] for c in scfg]
bars = ax.bar(sn, sv, color=sc, width=0.35, edgecolor='white', linewidth=2, zorder=3)
for b, v, c in zip(bars, sv, sc):
    ax.text(b.get_x()+b.get_width()/2, v+max(sv)*0.03, f'{v:.6f}', ha='center', fontsize=13, fontweight='bold', color=c)
ax.set_title("Regression Metrics — Steering", fontsize=16, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(STEER_OUTPUT_DIR, "all_metrics_combined.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: all_metrics_combined.png")

# Epoch table
fig, ax = plt.subplots(figsize=(12, max(4, len(s_tl)*0.4+1))); fig.patch.set_facecolor('#FAFAFA'); ax.axis('off')
td = [[f"Epoch {e+1}", f"{s_tl[e]:.6f}", f"{s_vl[e]:.6f}", f"{s_tm[e]:.6f}", f"{s_vm[e]:.6f}"] for e in range(len(s_tl))]
cl = ["Epoch","Train Loss","Test Loss","Train MAE","Test MAE"]
tb = ax.table(cellText=td, colLabels=cl, cellLoc='center', loc='center')
tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1.0, 1.5)
for j in range(len(cl)): tb[0,j].set_facecolor('#2196F3'); tb[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(td)+1):
    for j in range(len(cl)): tb[i,j].set_facecolor('#E3F2FD' if i%2==0 else '#FFFFFF')
ax.set_title("Epoch Results — Steering", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(STEER_OUTPUT_DIR, "epoch_table.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: epoch_table.png")


# #####################################################
#   B-STEP 6 :  STEERING SAMPLE PREDICTIONS
# #####################################################

print("\n" + "=" * 70)
print("  B-STEP 6 :  STEERING SAMPLE PREDICTIONS")
print("=" * 70)

abs_err = np.abs(y_st_pred - y_st_test)
np.random.seed(RANDOM_STATE)
s_sel = []
for idx_list, n in [([i for i in range(len(y_st_test)) if abs(y_st_test[i])<0.05], 4),
                     ([i for i in range(len(y_st_test)) if y_st_test[i]<-0.1], 3),
                     ([i for i in range(len(y_st_test)) if y_st_test[i]>0.1], 3)]:
    if idx_list: s_sel += list(np.random.choice(idx_list, min(n, len(idx_list)), replace=False))
for i in np.argsort(abs_err)[:30]:
    if i not in s_sel and len(s_sel) < 13: s_sel.append(i)
for i in np.argsort(abs_err)[-30:][::-1]:
    if i not in s_sel and len(s_sel) < 16: s_sel.append(i)

fig, axes = plt.subplots(4, 4, figsize=(20, 16)); fig.patch.set_facecolor('#FAFAFA')
for i, ax in enumerate(axes.flat):
    if i >= len(s_sel): ax.axis('off'); continue
    idx = s_sel[i]; ax.imshow(X_st_test[idx])
    a, p = y_st_test[idx], y_st_pred[idx]; err = abs(p-a)
    bc = '#2E7D32' if err < 0.05 else ('#FF9800' if err < 0.15 else '#C62828')
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor(bc); sp.set_linewidth(4)
    d = "→RIGHT" if a > 0.05 else ("←LEFT" if a < -0.05 else "↑STRAIGHT")
    ax.set_title(f"A:{a:.3f} P:{p:.3f}\n{d} err:{err:.3f}", fontsize=9, fontweight='bold', color=bc, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Steering Predictions (Green=Good, Orange=Fair, Red=Poor)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(STEER_OUTPUT_DIR, "sample_predictions.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  Saved: sample_predictions.png")

print("\n" + "=" * 70)
print("  ✓  PART B COMPLETE — Steering Prediction Done!")
print("=" * 70)

del X_st_train, X_st_test, y_st_train, y_st_test, y_st_pred, steer_model


# *************************************************************
# *************************************************************
#
#   PART C :  LANE DETECTION
#             U-Net Semantic Segmentation
#
# *************************************************************
# *************************************************************

print("\n\n" + "#" * 70)
print("#" * 70)
print("##")
print("##   PART C :  LANE DETECTION (U-Net Segmentation)")
print("##   Dataset : TUSimple Lane Detection Challenge")
print(f"##   Output  : {LANE_OUTPUT_DIR}")
print("##")
print("#" * 70)
print("#" * 70)

# Custom metrics
def dice_coefficient(y_true, y_pred, smooth=1.0):
    yt_c = K.cast(y_true, K.floatx())
    return (2.*K.sum(K.flatten(yt_c)*K.flatten(y_pred))+smooth) / (K.sum(K.flatten(yt_c))+K.sum(K.flatten(y_pred))+smooth)
def dice_loss(y_true, y_pred): return 1 - dice_coefficient(y_true, y_pred)
def combined_loss(y_true, y_pred):
    yt_c = K.cast(y_true, K.floatx())
    return K.mean(K.binary_crossentropy(yt_c, y_pred)) + dice_loss(yt_c, y_pred)
def iou_metric(y_true, y_pred, smooth=1.0):
    yt_c = K.cast(y_true, K.floatx())
    yb = K.cast(K.greater(y_pred, 0.5), K.floatx())
    inter = K.sum(yt_c * yb); union = K.sum(yt_c)+K.sum(yb)-inter
    return (inter+smooth)/(union+smooth)


# #####################################################
#   C-STEP 1 :  LOAD TUSIMPLE DATASET
# #####################################################

print("\n" + "=" * 70)
print("  C-STEP 1 :  LOADING TUSIMPLE DATASET")
print("=" * 70)

lane_start = time.time()

def gen_lane_mask(lanes, h_samples, shape, w=LANE_LINE_WIDTH):
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    for lane in lanes:
        pts = [(int(x), int(y)) for x, y in zip(lane, h_samples) if x != -2]
        if len(pts) >= 2: cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, w)
    return mask

label_files = [os.path.join(LANE_TRAIN_DIR, f) for f in
               ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]]

all_labels = []
for lf in label_files:
    if os.path.exists(lf):
        with open(lf, 'r') as f:
            entries = [json.loads(line.strip()) for line in f if line.strip()]
        print(f"  Loaded {len(entries)} from {os.path.basename(lf)}")
        all_labels.extend(entries)

print(f"  Total: {len(all_labels)} entries")

l_imgs, l_masks, skipped = [], [], 0
for i, entry in enumerate(all_labels):
    if (i+1) % 500 == 0 or i == 0: print(f"  Processing {i+1}/{len(all_labels)}...")
    ip = os.path.join(LANE_TRAIN_DIR, entry["raw_file"])
    if not os.path.exists(ip): skipped += 1; continue
    img = cv2.imread(ip)
    if img is None: skipped += 1; continue
    oh, ow = img.shape[:2]
    mask = gen_lane_mask(entry["lanes"], entry["h_samples"], (oh, ow))
    img = cv2.cvtColor(cv2.resize(img, (LANE_IMAGE_W, LANE_IMAGE_H)), cv2.COLOR_BGR2RGB)
    mask = (cv2.resize(mask, (LANE_IMAGE_W, LANE_IMAGE_H), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    l_imgs.append(img); l_masks.append(mask)

X_lane = np.array(l_imgs, np.uint8)
Y_lane = np.array(l_masks, np.uint8)[..., np.newaxis]
del l_imgs, l_masks
print(f"\n  Loaded: {X_lane.shape[0]} | Memory: {(X_lane.nbytes+Y_lane.nbytes)/(1024**3):.2f} GB")


# #####################################################
#   C-STEP 2 :  PREPARE LANE DATA
# #####################################################

print("\n" + "=" * 70)
print("  C-STEP 2 :  PREPARING LANE DATA")
print("=" * 70)

X_ln_train, X_ln_test, Y_ln_train, Y_ln_test = train_test_split(
    X_lane, Y_lane, test_size=LANE_SPLIT, random_state=RANDOM_STATE)
del X_lane, Y_lane
print(f"  Training: {len(X_ln_train)} | Testing: {len(X_ln_test)}")


# #####################################################
#   C-STEP 3 :  BUILD & TRAIN U-NET
# #####################################################

print("\n" + "=" * 70)
print("  C-STEP 3 :  BUILDING & TRAINING U-NET")
print("=" * 70)

def build_unet(shape=(LANE_IMAGE_H, LANE_IMAGE_W, 3)):
    inp = Input(shape)
    # Scale from [0, 255] to [0.0, 1.0] dynamically
    rescaled = Lambda(lambda x: x / 255.0)(inp)
    # Encoder
    c1=Conv2D(32,(3,3),activation='relu',padding='same')(rescaled); c1=BatchNormalization()(c1)
    c1=Conv2D(32,(3,3),activation='relu',padding='same')(c1); c1=BatchNormalization()(c1)
    p1=Dropout(0.1)(MaxPooling2D((2,2))(c1))
    c2=Conv2D(64,(3,3),activation='relu',padding='same')(p1); c2=BatchNormalization()(c2)
    c2=Conv2D(64,(3,3),activation='relu',padding='same')(c2); c2=BatchNormalization()(c2)
    p2=Dropout(0.1)(MaxPooling2D((2,2))(c2))
    c3=Conv2D(128,(3,3),activation='relu',padding='same')(p2); c3=BatchNormalization()(c3)
    c3=Conv2D(128,(3,3),activation='relu',padding='same')(c3); c3=BatchNormalization()(c3)
    p3=Dropout(0.2)(MaxPooling2D((2,2))(c3))
    c4=Conv2D(256,(3,3),activation='relu',padding='same')(p3); c4=BatchNormalization()(c4)
    c4=Conv2D(256,(3,3),activation='relu',padding='same')(c4); c4=BatchNormalization()(c4)
    p4=Dropout(0.2)(MaxPooling2D((2,2))(c4))
    # Bottleneck
    c5=Conv2D(512,(3,3),activation='relu',padding='same')(p4); c5=BatchNormalization()(c5)
    c5=Conv2D(512,(3,3),activation='relu',padding='same')(c5); c5=BatchNormalization()(c5)
    # Decoder
    u6=Dropout(0.2)(concatenate([Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(c5),c4]))
    c6=Conv2D(256,(3,3),activation='relu',padding='same')(u6); c6=BatchNormalization()(c6)
    c6=Conv2D(256,(3,3),activation='relu',padding='same')(c6); c6=BatchNormalization()(c6)
    u7=Dropout(0.2)(concatenate([Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c6),c3]))
    c7=Conv2D(128,(3,3),activation='relu',padding='same')(u7); c7=BatchNormalization()(c7)
    c7=Conv2D(128,(3,3),activation='relu',padding='same')(c7); c7=BatchNormalization()(c7)
    u8=Dropout(0.1)(concatenate([Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c7),c2]))
    c8=Conv2D(64,(3,3),activation='relu',padding='same')(u8); c8=BatchNormalization()(c8)
    c8=Conv2D(64,(3,3),activation='relu',padding='same')(c8); c8=BatchNormalization()(c8)
    u9=Dropout(0.1)(concatenate([Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c8),c1]))
    c9=Conv2D(32,(3,3),activation='relu',padding='same')(u9); c9=BatchNormalization()(c9)
    c9=Conv2D(32,(3,3),activation='relu',padding='same')(c9); c9=BatchNormalization()(c9)
    out=Conv2D(1,(1,1),activation='sigmoid')(c9)
    return Model(inputs=[inp], outputs=[out])

lane_model = build_unet()
lane_model.compile(optimizer=Adam(learning_rate=1e-3), loss=combined_loss,
                   metrics=['accuracy', dice_coefficient, iou_metric])
lane_model.summary()

print("\n  Training started...\n")
ln_train_start = time.time()
ln_history = lane_model.fit(
    X_ln_train, Y_ln_train, validation_data=(X_ln_test, Y_ln_test),
    epochs=LANE_EPOCHS, batch_size=LANE_BATCH,
    callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)],
    verbose=1)

ln_train_time = time.time() - ln_train_start
print(f"\n  Training done in {ln_train_time:.1f} seconds")
lane_model.save(os.path.join(LANE_OUTPUT_DIR, "lane_model.h5"))
print("  Model saved: lane_model.h5")

lt = ln_history.history; l_ep = range(1, len(lt['loss'])+1)


# #####################################################
#   C-STEP 4 :  EVALUATE LANE MODEL
# #####################################################

print("\n" + "=" * 70)
print("  C-STEP 4 :  LANE MODEL EVALUATION")
print("=" * 70)

Y_ln_pred = lane_model.predict(X_ln_test, verbose=0)
Y_ln_pred_bin = (Y_ln_pred > 0.5).astype(np.float32)

ytf = Y_ln_test.flatten().astype(int); ypf = Y_ln_pred_bin.flatten().astype(int)
px_acc  = accuracy_score(ytf, ypf)
px_prec = precision_score(ytf, ypf, zero_division=0)
px_rec  = recall_score(ytf, ypf, zero_division=0)
px_f1   = f1_score(ytf, ypf, zero_division=0)

inter = np.sum(Y_ln_test.flatten() * Y_ln_pred_bin.flatten())
union = np.sum(Y_ln_test.flatten()) + np.sum(Y_ln_pred_bin.flatten()) - inter
l_iou  = inter / (union + 1e-7)
l_dice = (2*inter) / (np.sum(Y_ln_test.flatten())+np.sum(Y_ln_pred_bin.flatten())+1e-7)

l_test_loss = lane_model.evaluate(X_ln_test, Y_ln_test, verbose=0)[0]

print(f"\n  Loss: {l_test_loss:.4f} | Acc: {px_acc*100:.2f}% | Prec: {px_prec*100:.2f}%")
print(f"  Recall: {px_rec*100:.2f}% | F1: {px_f1*100:.2f}% | IoU: {l_iou*100:.2f}% | Dice: {l_dice*100:.2f}%")

with open(os.path.join(LANE_OUTPUT_DIR, "segmentation_report.txt"), "w") as f:
    f.write("LANE DETECTION - U-NET SEGMENTATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Samples: {len(X_ln_train)+len(X_ln_test)} | Train: {len(X_ln_train)} | Test: {len(X_ln_test)}\n\n")
    f.write(f"Loss: {l_test_loss:.4f}\nPixel Acc: {px_acc*100:.2f}%\nPrecision: {px_prec*100:.2f}%\n")
    f.write(f"Recall: {px_rec*100:.2f}%\nF1: {px_f1*100:.2f}%\nIoU: {l_iou*100:.2f}%\nDice: {l_dice*100:.2f}%\n")
    f.write(f"\nTraining: {ln_train_time:.1f}s | Epochs: {len(lt['loss'])}\n")
print("  Saved: segmentation_report.txt")


# #####################################################
#   C-STEP 5 :  LANE DETECTION CHARTS
# #####################################################

print("\n" + "=" * 70)
print("  C-STEP 5 :  GENERATING LANE DETECTION CHARTS")
print("=" * 70)

# Combined 2x2 training progress
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(16,12)); fig.patch.set_facecolor('#FAFAFA')
for ax, tk, tv, c1c, c2c, ttl in [
    (ax1, lt['loss'], lt['val_loss'], '#E91E63', '#2196F3', 'Loss'),
    (ax2, lt['accuracy'], lt['val_accuracy'], '#4CAF50', '#FF9800', 'Pixel Accuracy'),
    (ax3, lt['iou_metric'], lt['val_iou_metric'], '#9C27B0', '#00BCD4', 'IoU'),
    (ax4, lt['dice_coefficient'], lt['val_dice_coefficient'], '#FF5722', '#607D8B', 'Dice')]:
    ax.set_facecolor('#FAFAFA')
    ax.plot(l_ep, tk, color=c1c, linewidth=2, marker='o', markersize=4, label='Train')
    ax.plot(l_ep, tv, color=c2c, linewidth=2, marker='s', markersize=4, label='Test')
    ax.set_title(ttl, fontsize=14, fontweight='bold'); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle("Training Progress — Lane Detection (U-Net)", fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "combined_training_progress.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: combined_training_progress.png")

# All metrics bar
lcfg = [{"name":"Pixel Acc","value":px_acc,"color":"#2196F3"},{"name":"Precision","value":px_prec,"color":"#FF9800"},
        {"name":"Recall","value":px_rec,"color":"#4CAF50"},{"name":"F1","value":px_f1,"color":"#E91E63"},
        {"name":"IoU","value":l_iou,"color":"#9C27B0"},{"name":"Dice","value":l_dice,"color":"#00BCD4"}]
fig, ax = plt.subplots(figsize=(12,6)); fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ln, lv, lc = [c["name"] for c in lcfg], [c["value"] for c in lcfg], [c["color"] for c in lcfg]
bars = ax.bar(ln, lv, color=lc, width=0.35, edgecolor='white', linewidth=2, zorder=3)
ax.axhline(y=1.0, color='#BDBDBD', linestyle='--', linewidth=1, alpha=0.7)
for b, v, c in zip(bars, lv, lc):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v*100:.2f}%', ha='center', fontsize=12, fontweight='bold', color=c)
ax.set_ylim(0, 1.15); ax.set_title("Segmentation Metrics — Lane Detection", fontsize=16, fontweight='bold', pad=20)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0]); ax.set_yticklabels(['0%','20%','40%','60%','80%','100%'])
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.yaxis.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "all_metrics_combined.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: all_metrics_combined.png")

# Radar
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True)); fig.patch.set_facecolor('#FAFAFA')
vr = lv + [lv[0]]; angs = np.linspace(0,2*np.pi,len(ln),endpoint=False).tolist()+[0]
ax.fill(angs, vr, color='#9C27B0', alpha=0.25); ax.plot(angs, vr, color='#9C27B0', linewidth=2.5, marker='o', markersize=8)
ax.set_xticks(angs[:-1]); ax.set_xticklabels(ln, fontsize=11, fontweight='bold'); ax.set_ylim(0, 1.1)
ax.set_title("Radar Chart — Lane Detection", fontsize=16, fontweight='bold', pad=30)
for a, v in zip(angs[:-1], lv): ax.text(a, v+0.08, f'{v*100:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#9C27B0')
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "metrics_radar_chart.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: metrics_radar_chart.png")

# Epoch table
fig, ax = plt.subplots(figsize=(14, max(4, len(lt['loss'])*0.4+1))); fig.patch.set_facecolor('#FAFAFA'); ax.axis('off')
td = [[f"Ep {e+1}", f"{lt['loss'][e]:.4f}", f"{lt['val_loss'][e]:.4f}",
       f"{lt['accuracy'][e]*100:.1f}%", f"{lt['val_accuracy'][e]*100:.1f}%",
       f"{lt['iou_metric'][e]:.4f}", f"{lt['val_iou_metric'][e]:.4f}"] for e in range(len(lt['loss']))]
cl = ["Epoch","TrLoss","TeLoss","TrAcc","TeAcc","TrIoU","TeIoU"]
tb = ax.table(cellText=td, colLabels=cl, cellLoc='center', loc='center')
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1.0, 1.5)
for j in range(len(cl)): tb[0,j].set_facecolor('#9C27B0'); tb[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(td)+1):
    for j in range(len(cl)): tb[i,j].set_facecolor('#F3E5F5' if i%2==0 else '#FFFFFF')
ax.set_title("Epoch Results — Lane Detection", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "epoch_table.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: epoch_table.png")


# #####################################################
#   C-STEP 6 :  LANE SAMPLE PREDICTIONS
# #####################################################

print("\n" + "=" * 70)
print("  C-STEP 6 :  LANE SAMPLE PREDICTIONS")
print("=" * 70)

np.random.seed(RANDOM_STATE)
l_sel = np.random.choice(len(X_ln_test), min(8, len(X_ln_test)), replace=False)

fig, axes = plt.subplots(8, 4, figsize=(20, 32)); fig.patch.set_facecolor('#FAFAFA')
for row, si in enumerate(l_sel):
    img = X_ln_test[si].astype(np.float32) / 255.0; gt = Y_ln_test[si,:,:,0]; pred = Y_ln_pred_bin[si,:,:,0]
    ov = img.copy()
    ov[gt>0] = ov[gt>0]*0.5 + np.array([0,0.8,0])*0.5
    ov[pred>0] = ov[pred>0]*0.5 + np.array([0,0.5,1.0])*0.5
    si_iou = np.sum(gt*pred) / (np.sum(gt)+np.sum(pred)-np.sum(gt*pred)+1e-7)

    axes[row,0].imshow(img); axes[row,0].axis('off')
    axes[row,1].imshow(gt, cmap='Greens', vmin=0, vmax=1); axes[row,1].axis('off')
    axes[row,2].imshow(pred, cmap='Blues', vmin=0, vmax=1); axes[row,2].axis('off')
    axes[row,3].imshow(ov); axes[row,3].axis('off')
    axes[row,3].set_title(f"IoU:{si_iou:.4f}", fontsize=10, fontweight='bold',
                          color='#2E7D32' if si_iou>0.5 else '#C62828')

for j, t in enumerate(["Original","Ground Truth","Predicted","Overlay"]):
    axes[0,j].set_title(f"{t}\n"+axes[0,j].get_title(), fontsize=12, fontweight='bold')

fig.suptitle("Lane Detection — GT vs Prediction\n(Green=GT, Blue=Predicted)", fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "sample_predictions.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  Saved: sample_predictions.png")

print("\n" + "=" * 70)
print("  ✓  PART C COMPLETE — Lane Detection Done!")
print("=" * 70)


# *************************************************************
#
#   ALL DONE — FINAL SUMMARY
#
# *************************************************************

total_time = time.time() - global_start

print("\n\n" + "=" * 70)
print("=" * 70)
print("  ALL 3 MODULES COMPLETE!")
print(f"  Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
print("=" * 70)
print("=" * 70)

print(f"""
  ╔═══════════════════════════════════════════════════════════════╗
  ║  PART A: OBSTACLE DETECTION                                  ║
  ║  Output: {OBS_OUTPUT_DIR:<52s}║
  ║  Files: model, report, confusion matrix, charts, predictions ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  PART B: STEERING PREDICTION                                 ║
  ║  Output: {STEER_OUTPUT_DIR:<52s}║
  ║  Files: model, report, scatter, distribution, charts         ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  PART C: LANE DETECTION                                      ║
  ║  Output: {LANE_OUTPUT_DIR:<52s}║
  ║  Files: model, report, IoU/Dice charts, predictions          ║
  ╚═══════════════════════════════════════════════════════════════╝
""")

print("=" * 70 + "\n")
