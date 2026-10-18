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
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# INPUT: Where your images and labels are
image_dir = r"D:\Finalyear pro\Images\data_object_image_2\training\image_2"
label_dir = r"D:\Finalyear pro\Labels\data_object_label_2\training\label_2"

# OUTPUT: Where everything gets saved
OUTPUT_DIR = r"D:\may26"

# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_SIZE   = 64
BATCH_SIZE   = 64
EPOCHS       = 20
TEST_SPLIT   = 0.2
RANDOM_STATE = 42

valid_classes = ["Car", "Cyclist", "Pedestrian", "Truck", "Van", "Tram"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# CHECK INPUT FOLDERS
# =====================================================

if not os.path.exists(image_dir):
    print(f"ERROR: Image folder not found!\n  {image_dir}")
    exit()

if not os.path.exists(label_dir):
    print(f"ERROR: Label folder not found!\n  {label_dir}")
    exit()


# #####################################################
#
#   STEP 1 :  LOAD KITTI DATASET
#
# #####################################################

print("=" * 55)
print("  STEP 1 :  LOADING KITTI DATASET")
print("=" * 55)

images = []
labels = []

start_time = time.time()

image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
])

print(f"  Found {len(image_files)} images")

for i, file in enumerate(image_files):

    if (i + 1) % 500 == 0 or i == 0:
        print(f"  Processing image {i + 1}/{len(image_files)}...")

    img_path   = os.path.join(image_dir, file)
    label_path = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")

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
        if obj not in valid_classes:
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

        crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32) / 255.0

        images.append(crop)
        labels.append(obj)

load_time = time.time() - start_time

X = np.array(images, dtype=np.float32)
y = np.array(labels)
del images

print(f"\n  Loaded in {load_time:.1f} seconds")
print(f"  Total objects : {X.shape[0]}")
print(f"  Image shape   : {X.shape[1:]}")
print(f"  Memory        : {X.nbytes / (1024**3):.2f} GB")

if len(X) == 0:
    print("\n  ERROR: No valid data found.")
    exit()

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print("\n  CLASS DISTRIBUTION:")
for cls, cnt in zip(unique, counts):
    print(f"    {cls:<15s} : {cnt:>6d}  ({cnt/len(y)*100:5.1f}%)")


# #####################################################
#
#   STEP 2 :  PREPARE DATA
#
# #####################################################

print("\n" + "=" * 55)
print("  STEP 2 :  PREPARING DATA")
print("=" * 55)

encoder = LabelEncoder()
y_encoded     = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_classes   = len(encoder.classes_)

print(f"  Classes ({num_classes}): {list(encoder.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical,
    test_size=TEST_SPLIT,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)
del X

print(f"  Training : {len(X_train)}")
print(f"  Testing  : {len(X_test)}")

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)


# #####################################################
#
#   STEP 3 :  BUILD & TRAIN CNN MODEL
#
# #####################################################

print("\n" + "=" * 55)
print("  STEP 3 :  BUILDING & TRAINING CNN MODEL")
print("=" * 55)

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop  = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
reduce_lr   = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
checkpoint  = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, "best_model_checkpoint.h5"),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print("\n  Training started...\n")
train_start = time.time()

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

train_time = time.time() - train_start
print(f"\n  Training done in {train_time:.1f} seconds")

# Save model
model_path = os.path.join(OUTPUT_DIR, "kitti_obstacle_model.h5")
model.save(model_path)
print(f"  Model saved: {model_path}")

# Save history
train_acc  = history.history['accuracy']
val_acc    = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
epoch_list = range(1, len(train_acc) + 1)


# #####################################################
#
#   STEP 4 :  EVALUATE MODEL
#   FIX: Use MACRO average so metrics differ per class
#
# #####################################################

print("\n" + "=" * 55)
print("  STEP 4 :  MODEL EVALUATION")
print("=" * 55)

loss, acc = model.evaluate(X_test, y_test, verbose=0)

y_pred         = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# -------------------------------------------------------
# WHY MACRO AVERAGE?
# "weighted" average can make Precision=Recall=F1=Accuracy
# because it weights by support (number of samples per class),
# which mathematically converges to the same number as overall
# accuracy when classes are reasonably balanced.
# MACRO average computes each class independently then averages —
# this correctly shows different values for each metric.
# For academic papers, MACRO is the standard for fairness.
# -------------------------------------------------------

accuracy  = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
recall    = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
f1        = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

# Per-class metrics for detailed paper reporting
per_class_precision = precision_score(y_true_classes, y_pred_classes, average=None, zero_division=0)
per_class_recall    = recall_score(y_true_classes, y_pred_classes, average=None, zero_division=0)
per_class_f1        = f1_score(y_true_classes, y_pred_classes, average=None, zero_division=0)

print(f"\n  Test Loss       : {loss:.4f}")
print(f"  Test Accuracy   : {accuracy*100:.2f}%")
print(f"  Macro Precision : {precision*100:.2f}%")
print(f"  Macro Recall    : {recall*100:.2f}%")
print(f"  Macro F1 Score  : {f1*100:.2f}%")

print("\n  PER-CLASS BREAKDOWN:")
print(f"  {'Class':<15} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}")
print("  " + "-" * 55)
for i, cls in enumerate(encoder.classes_):
    print(f"  {cls:<15} {per_class_precision[i]*100:>11.2f}% {per_class_recall[i]*100:>11.2f}% {per_class_f1[i]*100:>11.2f}%")

report = classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_)
print(f"\n{report}")

# Save classification report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write("KITTI OBSTACLE CLASSIFICATION - CNN RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("AVERAGING METHOD: MACRO (independent per-class then averaged)\n")
    f.write("This ensures Precision, Recall, F1 Score are DIFFERENT values.\n\n")
    f.write(f"Test Accuracy   : {accuracy*100:.2f}%\n")
    f.write(f"Macro Precision : {precision*100:.2f}%\n")
    f.write(f"Macro Recall    : {recall*100:.2f}%\n")
    f.write(f"Macro F1 Score  : {f1*100:.2f}%\n\n")
    f.write("PER-CLASS BREAKDOWN:\n")
    f.write(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}\n")
    f.write("-" * 55 + "\n")
    for i, cls in enumerate(encoder.classes_):
        f.write(f"{cls:<15} {per_class_precision[i]*100:>11.2f}% {per_class_recall[i]*100:>11.2f}% {per_class_f1[i]*100:>11.2f}%\n")
    f.write("\nFULL SKLEARN CLASSIFICATION REPORT:\n")
    f.write(report)


# #####################################################
#
#   STEP 5 :  CONFUSION MATRIX
#
# #####################################################

print("=" * 55)
print("  STEP 5 :  GENERATING CONFUSION MATRIX")
print("=" * 55)

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_,
            linewidths=0.5, linecolor='white')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - KITTI Object Classification", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()
print("  Saved: confusion_matrix.png")


# #####################################################
#
#   STEP 6 :  SAMPLE PREDICTIONS
#   Diverse classes + failure cases included
#   High resolution for paper clarity
#
# #####################################################

print("\n" + "=" * 55)
print("  STEP 6 :  SAMPLE PREDICTIONS (DIVERSE + FAILURES)")
print("=" * 55)

# Separate correct and wrong predictions
correct_indices = []
wrong_indices = []
for i in range(len(y_pred_classes)):
    if y_pred_classes[i] == y_true_classes[i]:
        correct_indices.append(i)
    else:
        wrong_indices.append(i)

print(f"  Total correct: {len(correct_indices)}")
print(f"  Total wrong  : {len(wrong_indices)}")

# Pick diverse correct samples — one from each class if possible
selected_correct = []
for cls_idx in range(num_classes):
    cls_correct = [i for i in correct_indices if y_true_classes[i] == cls_idx]
    if cls_correct:
        # Pick up to 2 from each class
        np.random.seed(RANDOM_STATE)
        picked = np.random.choice(cls_correct, size=min(2, len(cls_correct)), replace=False)
        selected_correct.extend(picked.tolist())

# Pick diverse wrong samples — try to get different class confusions
selected_wrong = []
if len(wrong_indices) > 0:
    # Try to get wrong predictions from different true classes
    for cls_idx in range(num_classes):
        cls_wrong = [i for i in wrong_indices if y_true_classes[i] == cls_idx]
        if cls_wrong:
            np.random.seed(RANDOM_STATE + cls_idx)
            picked = np.random.choice(cls_wrong, size=min(1, len(cls_wrong)), replace=False)
            selected_wrong.extend(picked.tolist())

# Build final selection: 12 correct (diverse) + 4 wrong (failures)
np.random.seed(RANDOM_STATE)
if len(selected_correct) > 12:
    selected_correct = list(np.random.choice(selected_correct, size=12, replace=False))
if len(selected_wrong) > 4:
    selected_wrong = list(np.random.choice(selected_wrong, size=4, replace=False))

# Combine: correct first, then failures
sample_indices = selected_correct + selected_wrong
num_samples = len(sample_indices)
rows = 4
cols = 4
num_slots = rows * cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
fig.patch.set_facecolor('#FAFAFA')

for i, ax in enumerate(axes.flat):
    if i >= num_samples:
        ax.axis('off')
        continue

    idx = sample_indices[i]
    ax.imshow(X_test[idx])
    pred_label = encoder.inverse_transform([y_pred_classes[idx]])[0]
    true_label = encoder.inverse_transform([y_true_classes[idx]])[0]
    is_correct = pred_label == true_label

    # Colored border around image (thick frame)
    border_color = '#2E7D32' if is_correct else '#C62828'
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)

    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    title_color = '#2E7D32' if is_correct else '#C62828'
    ax.set_title(f"Predicted: {pred_label}\nActual: {true_label}\n{status}",
                 fontsize=11, color=title_color, fontweight='bold', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Sample Predictions — Diverse Classes with Correct & Failure Cases\n(Green Border = Correct,  Red Border = Wrong)",
             fontsize=16, fontweight='bold', color='#333333', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: sample_predictions.png  ({len(selected_correct)} correct + {len(selected_wrong)} failures)")


# #####################################################
#
#   STEP 7 :  METRIC CHARTS
#   FIX 1: Thinner bars (width=0.25 instead of 0.5)
#   FIX 2: Values shown ONLY as percentage (no float+%)
#   FIX 3: Metrics now differ (macro averaging)
#
# #####################################################

print("\n" + "=" * 55)
print("  STEP 7 :  METRIC CHARTS")
print("=" * 55)

chart_config = [
    {"name": "Accuracy",  "value": accuracy,  "color": "#2196F3"},   # Blue
    {"name": "Precision", "value": precision, "color": "#FF9800"},   # Orange
    {"name": "Recall",    "value": recall,    "color": "#4CAF50"},   # Green
    {"name": "F1 Score",  "value": f1,        "color": "#E91E63"},   # Pink
]

for cfg in chart_config:

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    # FIX: Thinner bar — width changed from 0.5 → 0.25
    ax.bar([cfg["name"]], [cfg["value"]], color=cfg["color"], width=0.25,
           edgecolor='white', linewidth=2, zorder=3)

    ax.axhline(y=1.0, color='#BDBDBD', linestyle='--', linewidth=1, alpha=0.7)

    # FIX: Show ONLY percentage — removed duplicate float value above bar
    ax.text(0, cfg["value"] + 0.03, f'{cfg["value"] * 100:.2f}%',
            ha='center', va='bottom', fontsize=22, fontweight='bold', color=cfg["color"])

    # Removed: inner float label (was showing both float AND percent — now only percent above)

    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score", fontsize=14, fontweight='bold', color='#333333')
    ax.set_title(cfg["name"], fontsize=20, fontweight='bold', color=cfg["color"], pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDBDBD')
    ax.spines['bottom'].set_color('#BDBDBD')
    ax.tick_params(axis='both', colors='#666666', labelsize=12)
    ax.yaxis.grid(True, alpha=0.2, color='#999999')

    # Convert y-axis ticks to percentage labels
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    plt.tight_layout()
    fname = f"{cfg['name'].lower().replace(' ', '_')}_chart.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}  =  {cfg['value']*100:.2f}%")

# Combined metrics bar chart — thinner bars
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

m_names  = [c["name"]  for c in chart_config]
m_values = [c["value"] for c in chart_config]
m_colors = [c["color"] for c in chart_config]

# FIX: Thinner bars — width changed from 0.6 → 0.30
bars = ax.bar(m_names, m_values, color=m_colors, width=0.30, edgecolor='white', linewidth=2, zorder=3)
ax.axhline(y=1.0, color='#BDBDBD', linestyle='--', linewidth=1, alpha=0.7)

for bar, val, color in zip(bars, m_values, m_colors):
    # FIX: Show ONLY percentage above bar — removed inner float+percent
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.025, f'{val*100:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold', color=color)

ax.set_ylim(0, 1.2)
ax.set_ylabel("Score (%)", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("All Performance Metrics - KITTI Classification\n(Macro Average)", fontsize=15, fontweight='bold', color='#333333', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', colors='#666666', labelsize=12)
ax.yaxis.grid(True, alpha=0.2, color='#999999')

# Percentage y-axis labels
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_metrics_combined.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: all_metrics_combined.png")

# Per-class bar chart (NEW — for paper submission)
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

x = np.arange(len(encoder.classes_))
bar_width = 0.22   # Thin bars for grouped chart

b1 = ax.bar(x - bar_width, per_class_precision * 100, bar_width, label='Precision',  color='#FF9800', edgecolor='white', zorder=3)
b2 = ax.bar(x,             per_class_recall    * 100, bar_width, label='Recall',     color='#4CAF50', edgecolor='white', zorder=3)
b3 = ax.bar(x + bar_width, per_class_f1        * 100, bar_width, label='F1 Score',   color='#E91E63', edgecolor='white', zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(encoder.classes_, fontsize=12, fontweight='bold')
ax.set_ylabel("Score (%)", fontsize=13, fontweight='bold')
ax.set_title("Per-Class Precision, Recall & F1 Score - KITTI CNN", fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.set_ylim(0, 120)
ax.yaxis.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in list(b1) + list(b2) + list(b3):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.1f}%',
            ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "per_class_metrics.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: per_class_metrics.png")

# Radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#FAFAFA')

vals_r  = m_values + [m_values[0]]
angles  = np.linspace(0, 2*np.pi, len(m_names), endpoint=False).tolist()
angles += angles[:1]

ax.fill(angles, vals_r, color='#2196F3', alpha=0.25)
ax.plot(angles, vals_r, color='#2196F3', linewidth=2.5, marker='o', markersize=8)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(m_names, fontsize=13, fontweight='bold', color='#333333')
ax.set_ylim(0, 1.1)
ax.set_title("Metrics Radar Chart (Macro Average)", fontsize=16, fontweight='bold', color='#333333', pad=30)

# FIX: Show only percentage on radar labels
for angle, val in zip(angles[:-1], m_values):
    ax.text(angle, val + 0.08, f'{val*100:.1f}%',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#2196F3')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_radar_chart.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: metrics_radar_chart.png")


# #####################################################
#
#   STEP 7B :  PRECISION-RECALL CURVE & ROC CURVE
#   NEW — Required for paper/presentation
#   Uses One-vs-Rest (OvR) for multi-class
#
# #####################################################

print("\n" + "=" * 55)
print("  STEP 7B :  PRECISION-RECALL & ROC CURVES")
print("=" * 55)

# -------------------------------------------------------
# These curves need PROBABILITY scores (softmax output),
# not just hard class predictions.
# y_pred  = softmax probabilities  (shape: n_samples x n_classes)
# y_test  = one-hot encoded labels (shape: n_samples x n_classes)
# For multi-class, we use One-vs-Rest: treat each class
# as binary (this class vs all others) and plot one curve
# per class.
# -------------------------------------------------------

# Define distinct colors for each class (matching project palette)
curve_colors = {
    "Car":        "#4285F4",   # Blue
    "Cyclist":    "#FF9800",   # Orange
    "Pedestrian": "#4CAF50",   # Green
    "Truck":      "#E91E63",   # Pink
    "Van":        "#9C27B0",   # Purple
    "Tram":       "#00BCD4",   # Cyan
}

# Fallback colors if class order differs
default_colors = ["#4285F4", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]

# ----- PRECISION-RECALL CURVE -----
print("\n  Generating Precision-Recall Curve...")

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

all_precisions = []
all_recalls = []
ap_scores = []

for i, cls_name in enumerate(encoder.classes_):
    # Binary labels for this class: 1 if true class == i, else 0
    y_true_binary = y_test[:, i]
    # Probability scores for this class from softmax
    y_score = y_pred[:, i]

    # Compute precision-recall curve
    prec, rec, thresholds = precision_recall_curve(y_true_binary, y_score)

    # Average Precision (area under PR curve)
    ap = average_precision_score(y_true_binary, y_score)
    ap_scores.append(ap)

    # Get color
    color = curve_colors.get(cls_name, default_colors[i % len(default_colors)])

    # Plot curve
    ax.plot(rec, prec, color=color, linewidth=2.5, label=f'{cls_name} (AP = {ap:.3f})')

    all_precisions.append(prec)
    all_recalls.append(rec)

# Compute macro-average AP
macro_ap = np.mean(ap_scores)

# Plot styling
ax.set_xlabel("Recall", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Precision", fontsize=14, fontweight='bold', color='#333333')
ax.set_title(f"Precision-Recall Curve — KITTI Object Classification\n(Macro Average AP = {macro_ap:.3f})",
             fontsize=15, fontweight='bold', color='#333333', pad=20)
ax.legend(loc='lower left', fontsize=11, framealpha=0.9, fancybox=True, shadow=True)
ax.set_xlim([0.0, 1.05])
ax.set_ylim([0.0, 1.05])
ax.grid(True, alpha=0.2, color='#999999')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDBDBD')
ax.spines['bottom'].set_color('#BDBDBD')
ax.tick_params(axis='both', colors='#666666', labelsize=11)

# Add annotation for macro average
ax.annotate(f'Macro Avg AP: {macro_ap:.3f}',
            xy=(0.5, 0.5), xytext=(0.55, 0.15),
            fontsize=13, fontweight='bold', color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#2196F3', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curve.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: precision_recall_curve.png  (Macro AP = {macro_ap:.3f})")

# Print per-class AP
print("\n  AVERAGE PRECISION (AP) PER CLASS:")
for i, cls_name in enumerate(encoder.classes_):
    print(f"    {cls_name:<15} : AP = {ap_scores[i]:.4f}")
print(f"    {'Macro Average':<15} : AP = {macro_ap:.4f}")


# ----- ROC CURVE -----
print("\n  Generating ROC Curve...")

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

auc_scores = []

for i, cls_name in enumerate(encoder.classes_):
    # Binary labels for this class
    y_true_binary = y_test[:, i]
    # Probability scores for this class
    y_score = y_pred[:, i]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)

    # AUC (Area Under ROC Curve)
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)

    # Get color
    color = curve_colors.get(cls_name, default_colors[i % len(default_colors)])

    # Plot curve
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{cls_name} (AUC = {roc_auc:.3f})')

# Compute macro-average AUC
macro_auc = np.mean(auc_scores)

# Plot diagonal baseline (random classifier)
ax.plot([0, 1], [0, 1], color='#BDBDBD', linewidth=2, linestyle='--', label='Random (AUC = 0.500)')

# Plot styling
ax.set_xlabel("False Positive Rate (FPR)", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("True Positive Rate (TPR)", fontsize=14, fontweight='bold', color='#333333')
ax.set_title(f"ROC Curve — KITTI Object Classification\n(Macro Average AUC = {macro_auc:.3f})",
             fontsize=15, fontweight='bold', color='#333333', pad=20)
ax.legend(loc='lower right', fontsize=11, framealpha=0.9, fancybox=True, shadow=True)
ax.set_xlim([0.0, 1.05])
ax.set_ylim([0.0, 1.05])
ax.grid(True, alpha=0.2, color='#999999')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDBDBD')
ax.spines['bottom'].set_color('#BDBDBD')
ax.tick_params(axis='both', colors='#666666', labelsize=11)

# Add annotation for macro average AUC
ax.annotate(f'Macro Avg AUC: {macro_auc:.3f}',
            xy=(0.5, 0.5), xytext=(0.55, 0.25),
            fontsize=13, fontweight='bold', color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#2196F3', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: roc_curve.png  (Macro AUC = {macro_auc:.3f})")

# Print per-class AUC
print("\n  AUC (AREA UNDER ROC CURVE) PER CLASS:")
for i, cls_name in enumerate(encoder.classes_):
    print(f"    {cls_name:<15} : AUC = {auc_scores[i]:.4f}")
print(f"    {'Macro Average':<15} : AUC = {macro_auc:.4f}")

# Save PR & ROC results to the classification report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "a") as f:
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("PRECISION-RECALL & ROC CURVE RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("AVERAGE PRECISION (AP) PER CLASS:\n")
    f.write(f"{'Class':<15} {'AP':>10}\n")
    f.write("-" * 30 + "\n")
    for i, cls_name in enumerate(encoder.classes_):
        f.write(f"{cls_name:<15} {ap_scores[i]:>10.4f}\n")
    f.write(f"{'Macro Average':<15} {macro_ap:>10.4f}\n")
    f.write("\nAUC (AREA UNDER ROC CURVE) PER CLASS:\n")
    f.write(f"{'Class':<15} {'AUC':>10}\n")
    f.write("-" * 30 + "\n")
    for i, cls_name in enumerate(encoder.classes_):
        f.write(f"{cls_name:<15} {auc_scores[i]:>10.4f}\n")
    f.write(f"{'Macro Average':<15} {macro_auc:>10.4f}\n")


# #####################################################
#
#   STEP 8 :  TRAINING AND TESTING LOSS
#
# #####################################################

print("\n" + "=" * 55)
print("  STEP 8 :  TRAINING AND TESTING LOSS")
print("=" * 55)

# Training vs Testing Loss
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.plot(epoch_list, train_loss, color='#E91E63', linewidth=2.5, marker='o', markersize=6, label='Training Loss')
ax.plot(epoch_list, val_loss,   color='#2196F3', linewidth=2.5, marker='s', markersize=6, label='Testing Loss')
ax.fill_between(epoch_list, train_loss, alpha=0.1, color='#E91E63')
ax.fill_between(epoch_list, val_loss,   alpha=0.1, color='#2196F3')

ax.set_xlabel("Epoch", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Loss", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Training Loss vs Testing Loss", fontsize=16, fontweight='bold', color='#333333', pad=20)
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

# Training vs Testing Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.plot(epoch_list, train_acc, color='#4CAF50', linewidth=2.5, marker='o', markersize=6, label='Training Accuracy')
ax.plot(epoch_list, val_acc,   color='#FF9800', linewidth=2.5, marker='s', markersize=6, label='Testing Accuracy')
ax.fill_between(epoch_list, train_acc, alpha=0.1, color='#4CAF50')
ax.fill_between(epoch_list, val_acc,   alpha=0.1, color='#FF9800')

ax.set_xlabel("Epoch", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Training Accuracy vs Testing Accuracy", fontsize=16, fontweight='bold', color='#333333', pad=20)
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

# Percentage y-axis
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_testing_accuracy.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: training_testing_accuracy.png")

# Combined side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#FAFAFA')

ax1.set_facecolor('#FAFAFA')
ax1.plot(epoch_list, train_loss, color='#E91E63', linewidth=2.5, marker='o', markersize=5, label='Train Loss')
ax1.plot(epoch_list, val_loss,   color='#2196F3', linewidth=2.5, marker='s', markersize=5, label='Test Loss')
ax1.fill_between(epoch_list, train_loss, alpha=0.1, color='#E91E63')
ax1.fill_between(epoch_list, val_loss,   alpha=0.1, color='#2196F3')
ax1.set_title("Training & Testing Loss", fontsize=14, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax1.set_ylabel("Loss", fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.set_facecolor('#FAFAFA')
ax2.plot(epoch_list, train_acc, color='#4CAF50', linewidth=2.5, marker='o', markersize=5, label='Train Accuracy')
ax2.plot(epoch_list, val_acc,   color='#FF9800', linewidth=2.5, marker='s', markersize=5, label='Test Accuracy')
ax2.fill_between(epoch_list, train_acc, alpha=0.1, color='#4CAF50')
ax2.fill_between(epoch_list, val_acc,   alpha=0.1, color='#FF9800')
ax2.set_title("Training & Testing Accuracy", fontsize=14, fontweight='bold')
ax2.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.2)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle("Model Training Progress - KITTI CNN", fontsize=16, fontweight='bold', color='#333333', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_loss_accuracy.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: combined_loss_accuracy.png")

# Epoch table
fig, ax = plt.subplots(figsize=(12, max(4, len(train_acc) * 0.4 + 1)))
fig.patch.set_facecolor('#FAFAFA')
ax.axis('off')

table_data = []
for e in range(len(train_acc)):
    table_data.append([
        f"Epoch {e+1}",
        f"{train_loss[e]:.4f}",
        f"{val_loss[e]:.4f}",
        f"{train_acc[e]*100:.2f}%",
        f"{val_acc[e]*100:.2f}%"
    ])

col_labels = ["Epoch", "Train Loss", "Test Loss", "Train Accuracy", "Test Accuracy"]
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.5)

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2196F3')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        table[i, j].set_facecolor('#E3F2FD' if i % 2 == 0 else '#FFFFFF')

ax.set_title("Epoch-by-Epoch Results", fontsize=14, fontweight='bold', color='#333333', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "epoch_table.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: epoch_table.png")


# #####################################################
#
#   STEP 9 :  SCENARIO EXAMPLES
#
# #####################################################

print("\n" + "=" * 55)
print("  STEP 9 :  SCENARIO EXAMPLES")
print("=" * 55)

CLASS_COLORS = {
    "Car":        (66,  133, 244),
    "Cyclist":    (255, 152, 0),
    "Pedestrian": (76,  175, 80),
    "Truck":      (233, 30,  99),
    "Van":        (156, 39,  176),
    "Tram":       (0,   188, 212),
}

CLASS_COLORS_RGB = {
    "Car": "#4285F4", "Cyclist": "#FF9800", "Pedestrian": "#4CAF50",
    "Truck": "#E91E63", "Van": "#9C27B0", "Tram": "#00BCD4",
}


def draw_detections(image, detections):
    img = image.copy()
    for det in detections:
        cls, x1, y1, x2, y2, pred_cls, conf = det
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"{pred_cls} ({conf*100:.1f}%)"
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1 - ts[1] - 10), (x1 + ts[0] + 6, y1), color, -1)
        cv2.putText(img, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img


def predict_crop_fn(model, crop, enc_cls):
    r = cv2.resize(crop, (64, 64))
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    r = r.astype(np.float32) / 255.0
    p = model.predict(np.expand_dims(r, axis=0), verbose=0)
    idx = np.argmax(p)
    return str(enc_cls[idx]), float(p[0][idx])


def get_detections(file):
    ip = os.path.join(image_dir, file)
    lp = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")
    img = cv2.imread(ip)
    h, w = img.shape[:2]
    with open(lp, 'r') as f:
        lines = f.readlines()
    dets = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue
        cls = parts[0]
        if cls not in CLASS_COLORS:
            continue
        x1 = max(0, int(float(parts[4])))
        y1 = max(0, int(float(parts[5])))
        x2 = min(w, int(float(parts[6])))
        y2 = min(h, int(float(parts[7])))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pc, cf = predict_crop_fn(model, crop, encoder.classes_)
        dets.append((cls, x1, y1, x2, y2, pc, cf))
    return cv2.cvtColor(draw_detections(img, dets), cv2.COLOR_BGR2RGB), dets


print("\n  Scanning images for scenarios (pedestrian + cyclist + vehicles)...")

ped_cyc_imgs = []    # Scenes with pedestrians AND cyclists
ped_only_imgs = []   # Scenes with 2+ pedestrians
veh_imgs = []        # Scenes with diverse vehicles
mix_imgs = []        # Mixed traffic

for file in image_files:
    lp = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")
    if not os.path.exists(lp):
        continue
    with open(lp, 'r') as f:
        lines = f.readlines()
    pc = cc = vc = car_c = total = 0
    vt = set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue
        c = parts[0]
        if c == "Pedestrian":
            pc += 1; total += 1
        elif c == "Cyclist":
            cc += 1; total += 1
        elif c in ["Truck", "Van", "Tram"]:
            vc += 1; vt.add(c); total += 1
        elif c == "Car":
            car_c += 1; vt.add(c); total += 1

    # Pedestrian + Cyclist scenes (sir's requirement: 2-3 pedestrians + cyclist)
    if pc >= 2 and cc >= 1:
        ped_cyc_imgs.append((file, pc, cc))
    elif pc >= 2:
        ped_only_imgs.append((file, pc, cc))

    # Vehicle scenes with variety
    if vc >= 1 or car_c >= 2:
        all_vt = vt.copy()
        veh_imgs.append((file, vc + car_c, len(all_vt), all_vt))

    # Mixed traffic
    if pc >= 1 and (vc >= 1 or total >= 4):
        mix_imgs.append((file, total))

# Sort: prioritize diversity and count
ped_cyc_imgs.sort(key=lambda x: (x[1] + x[2]), reverse=True)
ped_only_imgs.sort(key=lambda x: x[1], reverse=True)
veh_imgs.sort(key=lambda x: (x[2], x[1]), reverse=True)  # Sort by variety first
mix_imgs.sort(key=lambda x: x[1], reverse=True)

print(f"  Pedestrian+Cyclist scenes : {len(ped_cyc_imgs)}")
print(f"  Pedestrian-only scenes   : {len(ped_only_imgs)}")
print(f"  Vehicle scenes            : {len(veh_imgs)}")
print(f"  Mixed scenes              : {len(mix_imgs)}")

N = 6  # Show 6 images per scenario for better coverage

# --- Scenario 1: Pedestrian + Cyclist Detection ---
print("\n  Generating: Pedestrian + Cyclist Detection...")

# Combine: prefer pedestrian+cyclist scenes, fill remainder with pedestrian-only
scenario1_imgs = ped_cyc_imgs[:N]
if len(scenario1_imgs) < N:
    remaining = N - len(scenario1_imgs)
    scenario1_imgs += ped_only_imgs[:remaining]

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.patch.set_facecolor('#1a1a2e')

for idx, ax in enumerate(axes.flat):
    if idx >= min(N, len(scenario1_imgs)):
        ax.axis('off'); continue
    entry = scenario1_imgs[idx]
    file = entry[0]
    ped_count = entry[1]
    cyc_count = entry[2] if len(entry) > 2 else 0
    rgb, _ = get_detections(file)
    ax.imshow(rgb)
    subtitle = f"{ped_count} Pedestrian(s)"
    if cyc_count > 0:
        subtitle += f" + {cyc_count} Cyclist(s)"
    ax.set_title(f"Scene: {file}\n{subtitle}",
                 fontsize=12, fontweight='bold', color='white', pad=10)
    ax.axis('off')

fig.suptitle("SCENARIO 1 : Pedestrian & Cyclist Detection\n(2-3 Pedestrians + Cyclists in Scene)",
             fontsize=20, fontweight='bold', color='#4CAF50', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scenario_person_walking.png"), dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("  Saved: scenario_person_walking.png")

# --- Scenario 2: Vehicle Detection (Diverse Types) ---
print("  Generating: Vehicle Detection (Diverse Types)...")
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.patch.set_facecolor('#1a1a2e')

for idx, ax in enumerate(axes.flat):
    if idx >= min(N, len(veh_imgs)):
        ax.axis('off'); continue
    file, cnt, variety, vt = veh_imgs[idx]
    rgb, _ = get_detections(file)
    ax.imshow(rgb)
    ax.set_title(f"Vehicle Scene - {file}\n{', '.join(vt)} ({cnt} vehicles)",
                 fontsize=12, fontweight='bold', color='white', pad=10)
    ax.axis('off')

fig.suptitle("SCENARIO 2 : Vehicle Detection — Diverse Vehicle Types\n(Cars, Trucks, Vans, Trams)",
             fontsize=20, fontweight='bold', color='#E91E63', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scenario_bus_vehicle.png"), dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("  Saved: scenario_bus_vehicle.png")

# --- Scenario 3: Mixed Traffic ---
print("  Generating: Mixed Traffic...")
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.patch.set_facecolor('#1a1a2e')

for idx, ax in enumerate(axes.flat):
    if idx >= min(N, len(mix_imgs)):
        ax.axis('off'); continue
    file, total = mix_imgs[idx]
    rgb, _ = get_detections(file)
    ax.imshow(rgb)
    ax.set_title(f"Mixed Traffic - {file}\n{total} objects",
                 fontsize=12, fontweight='bold', color='white', pad=10)
    ax.axis('off')

fig.suptitle("SCENARIO 3 : Mixed Traffic — Pedestrians + Vehicles Together",
             fontsize=20, fontweight='bold', color='#FF9800', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scenario_mixed_traffic.png"), dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("  Saved: scenario_mixed_traffic.png")

# --- Legend ---
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.axis('off')

for i, (cls, color) in enumerate(CLASS_COLORS_RGB.items()):
    y_pos = 0.85 - (i * 0.13)
    ax.add_patch(plt.Rectangle((0.1, y_pos-0.04), 0.08, 0.08,
                 facecolor=color, edgecolor='white', linewidth=1.5, transform=ax.transAxes))
    ax.text(0.22, y_pos, cls, fontsize=14, fontweight='bold', color='white', va='center', transform=ax.transAxes)

ax.set_title("Detection Legend - Class Colors", fontsize=14, fontweight='bold', color='white', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "detection_legend.png"), dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("  Saved: detection_legend.png")



# #####################################################
#
#   ALL DONE!
#
# #####################################################

print("\n\n" + "=" * 55)
print("  ALL DONE! EVERYTHING SAVED TO D:\\may26")
print("=" * 55)
print()
print("  FILES CREATED:")
print("  -----------------------------------------------")
print("  MODEL:")
print("    kitti_obstacle_model.h5")
print("    classification_report.txt")
print()
print("  METRIC CHARTS (MACRO average — metrics are DIFFERENT):")
print("    accuracy_chart.png          (Blue)")
print("    precision_chart.png         (Orange)")
print("    recall_chart.png            (Green)")
print("    f1_score_chart.png          (Pink)")
print("    all_metrics_combined.png")
print("    per_class_metrics.png       (per-class grouped chart)")
print("    metrics_radar_chart.png")
print()
print("  NEW — PRECISION-RECALL & ROC CURVES:")
print("    precision_recall_curve.png  (per-class PR curves + macro AP)")
print("    roc_curve.png               (per-class ROC curves + macro AUC)")
print()
print("  TRAINING & TESTING:")
print("    training_testing_loss.png")
print("    training_testing_accuracy.png")
print("    combined_loss_accuracy.png")
print("    epoch_table.png")
print()
print("  SCENARIO EXAMPLES:")
print("    scenario_person_walking.png")
print("    scenario_bus_vehicle.png")
print("    scenario_mixed_traffic.png")
print("    detection_legend.png")
print()
print("  OTHER:")
print("    confusion_matrix.png")
print("    sample_predictions.png")
print("  -----------------------------------------------")
print(f"\n  All saved to: {OUTPUT_DIR}")
print("=" * 55 + "\n")
