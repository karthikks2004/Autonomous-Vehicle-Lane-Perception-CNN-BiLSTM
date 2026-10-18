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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# =============================================================
#
#   STEERING WHEEL PREDICTION - CNN REGRESSION
#   Dataset: Self-Driving Car Simulator (Jungle + Make)
#
# =============================================================


# =====================================================
# INPUT & OUTPUT CONFIGURATION
# =====================================================

# Dataset directories
JUNGLE_DIR = r"D:\Finalyear pro\steering prediction\archive\self_driving_car_dataset_jungle"
MAKE_DIR   = r"D:\Finalyear pro\steering prediction\archive\self_driving_car_dataset_make"

# Output directory
OUTPUT_DIR = r"D:\STEERING PREDICTION PART"

# Training configuration
IMAGE_HEIGHT = 66       # NVIDIA paper standard
IMAGE_WIDTH  = 200      # NVIDIA paper standard
BATCH_SIZE   = 64
EPOCHS       = 20
TEST_SPLIT   = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# CHECK INPUT FOLDERS
# =====================================================

for d, name in [(JUNGLE_DIR, "Jungle"), (MAKE_DIR, "Make")]:
    if not os.path.exists(d):
        print(f"ERROR: {name} dataset folder not found!\n  {d}")
        exit()

print("=" * 60)
print("  STEERING WHEEL PREDICTION - CNN REGRESSION")
print("=" * 60)


# #####################################################
#
#   STEP 1 :  LOAD DATASET
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 1 :  LOADING STEERING DATASET")
print("=" * 60)

start_time = time.time()


def load_driving_data(dataset_dir, dataset_name):
    """
    Load driving log CSV and corresponding center camera images.
    CSV columns: center_path, left_path, right_path, steering, throttle, reverse, speed
    The paths in CSV point to original PC — we remap to local IMG folder.
    """
    csv_path = os.path.join(dataset_dir, "driving_log.csv")
    img_dir  = os.path.join(dataset_dir, "IMG")

    print(f"\n  Loading {dataset_name} dataset...")
    print(f"    CSV  : {csv_path}")
    print(f"    IMG  : {img_dir}")

    # Read CSV (no header row in these files)
    df = pd.read_csv(csv_path, header=None,
                     names=["center", "left", "right", "steering", "throttle", "reverse", "speed"])

    images = []
    angles = []
    skipped = 0

    for idx, row in df.iterrows():
        if (idx + 1) % 2000 == 0 or idx == 0:
            print(f"    Processing {idx + 1}/{len(df)}...")

        # Extract just the filename from the full path in CSV
        center_filename = os.path.basename(row["center"].strip())
        img_path = os.path.join(img_dir, center_filename)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Read and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        # Crop sky and car hood (top 60px for sky, bottom 25px for hood)
        img = img[60:135, :, :]

        # Resize to NVIDIA standard
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        images.append(img)
        angles.append(float(row["steering"]))

    print(f"    Loaded : {len(images)} images")
    if skipped > 0:
        print(f"    Skipped: {skipped} (file not found or unreadable)")

    return images, angles


# Load both datasets
images_jungle, angles_jungle = load_driving_data(JUNGLE_DIR, "Jungle")
images_make,   angles_make   = load_driving_data(MAKE_DIR,   "Make")

# Combine both datasets
all_images = images_jungle + images_make
all_angles = angles_jungle + angles_make
del images_jungle, images_make, angles_jungle, angles_make

X = np.array(all_images, dtype=np.float32)
y = np.array(all_angles, dtype=np.float32)
del all_images

load_time = time.time() - start_time

print(f"\n  Combined Dataset Stats:")
print(f"  Total samples  : {len(X)}")
print(f"  Image shape    : {X.shape[1:]}")
print(f"  Memory         : {X.nbytes / (1024**3):.2f} GB")
print(f"  Loaded in      : {load_time:.1f} seconds")
print(f"  Steering range : [{y.min():.4f}, {y.max():.4f}]")
print(f"  Steering mean  : {y.mean():.4f}")
print(f"  Steering std   : {y.std():.4f}")

if len(X) == 0:
    print("\n  ERROR: No valid data found.")
    exit()


# #####################################################
#
#   STEP 2 :  PREPARE DATA
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 2 :  PREPARING DATA")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SPLIT,
    random_state=RANDOM_STATE
)
del X

print(f"  Training samples : {len(X_train)}")
print(f"  Testing samples  : {len(X_test)}")
print(f"  Train steering range : [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"  Test  steering range : [{y_test.min():.4f}, {y_test.max():.4f}]")


# #####################################################
#
#   STEP 3 :  BUILD & TRAIN CNN MODEL
#   Architecture inspired by NVIDIA's self-driving paper
#   "End to End Learning for Self-Driving Cars"
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 3 :  BUILDING & TRAINING NVIDIA CNN MODEL")
print("=" * 60)

model = Sequential()

# Normalization layer (center around zero)
model.add(Lambda(lambda x: x - 0.5, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

# NVIDIA Architecture - 5 Convolutional layers
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
model.add(BatchNormalization())

model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
model.add(BatchNormalization())

model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(BatchNormalization())

# Flatten and Dense layers
model.add(Flatten())

model.add(Dense(100, activation='elu'))
model.add(Dropout(0.3))

model.add(Dense(50, activation='elu'))
model.add(Dropout(0.3))

model.add(Dense(10, activation='elu'))

# Output: single value (steering angle)
model.add(Dense(1))

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
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
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

train_time = time.time() - train_start
print(f"\n  Training done in {train_time:.1f} seconds")

# Save model
model_path = os.path.join(OUTPUT_DIR, "steering_model.h5")
model.save(model_path)
print(f"  Model saved: {model_path}")

# Extract history
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
train_mae  = history.history['mae']
val_mae    = history.history['val_mae']
epoch_list = range(1, len(train_loss) + 1)


# #####################################################
#
#   STEP 4 :  EVALUATE MODEL
#   Regression metrics: MAE, MSE, RMSE, R²
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 4 :  MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test, verbose=0).flatten()

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n  Mean Absolute Error (MAE)  : {mae:.6f}")
print(f"  Mean Squared Error  (MSE)  : {mse:.6f}")
print(f"  Root Mean Sq Error  (RMSE) : {rmse:.6f}")
print(f"  R² Score                   : {r2:.6f}")

# Save regression report
with open(os.path.join(OUTPUT_DIR, "regression_report.txt"), "w") as f:
    f.write("STEERING WHEEL PREDICTION - CNN REGRESSION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model: NVIDIA-inspired CNN (5 Conv + 3 Dense)\n")
    f.write(f"Dataset: Jungle ({len(images_jungle) if 'images_jungle' in dir() else 'N/A'}) + Make\n")
    f.write(f"Total Samples: {len(X_train) + len(X_test)}\n")
    f.write(f"Training: {len(X_train)} | Testing: {len(X_test)}\n\n")
    f.write("REGRESSION METRICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"  MAE   : {mae:.6f}\n")
    f.write(f"  MSE   : {mse:.6f}\n")
    f.write(f"  RMSE  : {rmse:.6f}\n")
    f.write(f"  R²    : {r2:.6f}\n\n")
    f.write(f"Training Time: {train_time:.1f} seconds\n")
    f.write(f"Epochs Trained: {len(train_loss)}\n")

print("  Saved: regression_report.txt")


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
ax.set_ylabel("Loss (MSE)", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Training Loss vs Testing Loss — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.annotate(f'Train: {train_loss[-1]:.6f}', xy=(len(train_loss), train_loss[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#E91E63',
            arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))
ax.annotate(f'Test: {val_loss[-1]:.6f}', xy=(len(val_loss), val_loss[-1]),
            xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold', color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_testing_loss.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: training_testing_loss.png")


# --- 5b: Training vs Testing MAE ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.plot(epoch_list, train_mae, color='#4CAF50', linewidth=2.5, marker='o', markersize=6, label='Training MAE')
ax.plot(epoch_list, val_mae,   color='#FF9800', linewidth=2.5, marker='s', markersize=6, label='Testing MAE')
ax.fill_between(epoch_list, train_mae, alpha=0.1, color='#4CAF50')
ax.fill_between(epoch_list, val_mae,   alpha=0.1, color='#FF9800')

ax.set_xlabel("Epoch", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("MAE", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Training MAE vs Testing MAE — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.annotate(f'Train: {train_mae[-1]:.6f}', xy=(len(train_mae), train_mae[-1]),
            xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))
ax.annotate(f'Test: {val_mae[-1]:.6f}', xy=(len(val_mae), val_mae[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#FF9800',
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_testing_mae.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: training_testing_mae.png")


# --- 5c: Combined Loss + MAE side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#FAFAFA')

ax1.set_facecolor('#FAFAFA')
ax1.plot(epoch_list, train_loss, color='#E91E63', linewidth=2.5, marker='o', markersize=5, label='Train Loss')
ax1.plot(epoch_list, val_loss,   color='#2196F3', linewidth=2.5, marker='s', markersize=5, label='Test Loss')
ax1.fill_between(epoch_list, train_loss, alpha=0.1, color='#E91E63')
ax1.fill_between(epoch_list, val_loss,   alpha=0.1, color='#2196F3')
ax1.set_title("Training & Testing Loss", fontsize=14, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax1.set_ylabel("Loss (MSE)", fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.set_facecolor('#FAFAFA')
ax2.plot(epoch_list, train_mae, color='#4CAF50', linewidth=2.5, marker='o', markersize=5, label='Train MAE')
ax2.plot(epoch_list, val_mae,   color='#FF9800', linewidth=2.5, marker='s', markersize=5, label='Test MAE')
ax2.fill_between(epoch_list, train_mae, alpha=0.1, color='#4CAF50')
ax2.fill_between(epoch_list, val_mae,   alpha=0.1, color='#FF9800')
ax2.set_title("Training & Testing MAE", fontsize=14, fontweight='bold')
ax2.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax2.set_ylabel("MAE", fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle("Model Training Progress — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_loss_mae.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: combined_loss_mae.png")


# --- 5d: Actual vs Predicted Scatter Plot ---
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='#2196F3', edgecolors='none', label='Predictions')

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction', color='#E91E63')

ax.set_xlabel("Actual Steering Angle", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Predicted Steering Angle", fontsize=14, fontweight='bold', color='#333333')
ax.set_title(f"Actual vs Predicted Steering Angle\nR² = {r2:.4f}", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: actual_vs_predicted.png")


# --- 5e: Steering Angle Distribution ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.hist(y_test, bins=50, alpha=0.6, color='#2196F3', edgecolor='white', linewidth=1, label='Actual', density=True)
ax.hist(y_pred, bins=50, alpha=0.6, color='#FF9800', edgecolor='white', linewidth=1, label='Predicted', density=True)

ax.set_xlabel("Steering Angle", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Density", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Steering Angle Distribution — Actual vs Predicted", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "steering_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: steering_distribution.png")


# --- 5f: Prediction Error Distribution ---
errors = y_pred - y_test

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.hist(errors, bins=50, alpha=0.7, color='#4CAF50', edgecolor='white', linewidth=1)
ax.axvline(x=0, color='#E91E63', linestyle='--', linewidth=2, label='Zero Error')
ax.axvline(x=errors.mean(), color='#FF9800', linestyle='--', linewidth=2, label=f'Mean Error: {errors.mean():.4f}')

ax.set_xlabel("Prediction Error (Predicted - Actual)", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Frequency", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Prediction Error Distribution", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "error_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: error_distribution.png")


# --- 5g: Residual Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

ax.scatter(y_pred, errors, alpha=0.3, s=10, color='#9C27B0', edgecolors='none')
ax.axhline(y=0, color='#E91E63', linestyle='--', linewidth=2)

ax.set_xlabel("Predicted Steering Angle", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Residual (Error)", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Residual Plot — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: residual_plot.png")


# --- 5h: All Metrics Combined Bar Chart ---
metric_config = [
    {"name": "MAE",  "value": mae,  "color": "#2196F3"},
    {"name": "MSE",  "value": mse,  "color": "#FF9800"},
    {"name": "RMSE", "value": rmse, "color": "#4CAF50"},
    {"name": "R²",   "value": r2,   "color": "#E91E63"},
]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

m_names  = [c["name"]  for c in metric_config]
m_values = [c["value"] for c in metric_config]
m_colors = [c["color"] for c in metric_config]

bars = ax.bar(m_names, m_values, color=m_colors, width=0.35, edgecolor='white', linewidth=2, zorder=3)

for bar, val, color in zip(bars, m_values, m_colors):
    ax.text(bar.get_x() + bar.get_width()/2, val + max(m_values)*0.03,
            f'{val:.6f}', ha='center', va='bottom', fontsize=13, fontweight='bold', color=color)

ax.set_ylabel("Score", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Regression Metrics — Steering Wheel Prediction", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', colors='#666666', labelsize=12)
ax.yaxis.grid(True, alpha=0.2, color='#999999')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_metrics_combined.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: all_metrics_combined.png")


# --- 5i: Radar Chart ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#FAFAFA')

# Normalize metrics to [0,1] for radar display
# For MAE/MSE/RMSE: lower=better so use 1-normalized; For R²: higher=better
max_error = max(mae, mse, rmse) if max(mae, mse, rmse) > 0 else 1
radar_values = [
    1 - mae / max_error,    # MAE (inverted)
    1 - mse / max_error,    # MSE (inverted)
    1 - rmse / max_error,   # RMSE (inverted)
    max(0, r2),             # R² (already 0-1)
]
radar_labels = ["MAE\n(lower=better)", "MSE\n(lower=better)", "RMSE\n(lower=better)", "R² Score"]

vals_r  = radar_values + [radar_values[0]]
angles  = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
angles += angles[:1]

ax.fill(angles, vals_r, color='#2196F3', alpha=0.25)
ax.plot(angles, vals_r, color='#2196F3', linewidth=2.5, marker='o', markersize=8)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11, fontweight='bold', color='#333333')
ax.set_ylim(0, 1.1)
ax.set_title("Metrics Radar Chart — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', pad=30)

# Display actual values
actual_vals = [mae, mse, rmse, r2]
for angle, val, actual in zip(angles[:-1], radar_values, actual_vals):
    ax.text(angle, val + 0.08, f'{actual:.4f}',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#2196F3')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_radar_chart.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: metrics_radar_chart.png")


# --- 5j: Epoch Table ---
fig, ax = plt.subplots(figsize=(12, max(4, len(train_loss) * 0.4 + 1)))
fig.patch.set_facecolor('#FAFAFA')
ax.axis('off')

table_data = []
for e in range(len(train_loss)):
    table_data.append([
        f"Epoch {e+1}",
        f"{train_loss[e]:.6f}",
        f"{val_loss[e]:.6f}",
        f"{train_mae[e]:.6f}",
        f"{val_mae[e]:.6f}"
    ])

col_labels = ["Epoch", "Train Loss", "Test Loss", "Train MAE", "Test MAE"]
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

ax.set_title("Epoch-by-Epoch Results — Steering Prediction", fontsize=14, fontweight='bold', color='#333333', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "epoch_table.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: epoch_table.png")


# #####################################################
#
#   STEP 6 :  SAMPLE PREDICTIONS
#   Show images with predicted vs actual steering angle
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 6 :  SAMPLE PREDICTIONS")
print("=" * 60)

# Find indices for different steering scenarios
straight_idx = [i for i in range(len(y_test)) if abs(y_test[i]) < 0.05]
left_idx     = [i for i in range(len(y_test)) if y_test[i] < -0.1]
right_idx    = [i for i in range(len(y_test)) if y_test[i] > 0.1]

# Also find some good and bad predictions
abs_errors = np.abs(y_pred - y_test)
good_pred_idx = np.argsort(abs_errors)[:50].tolist()        # Best predictions
bad_pred_idx  = np.argsort(abs_errors)[-50:].tolist()[::-1]  # Worst predictions

# Pick diverse samples: 4 straight + 3 left + 3 right + 3 good + 3 bad = 16
np.random.seed(RANDOM_STATE)
selected = []

if len(straight_idx) >= 4:
    selected += list(np.random.choice(straight_idx, 4, replace=False))
elif len(straight_idx) > 0:
    selected += straight_idx[:4]

if len(left_idx) >= 3:
    selected += list(np.random.choice(left_idx, 3, replace=False))
elif len(left_idx) > 0:
    selected += left_idx[:3]

if len(right_idx) >= 3:
    selected += list(np.random.choice(right_idx, 3, replace=False))
elif len(right_idx) > 0:
    selected += right_idx[:3]

# Add good predictions (not already selected)
for idx in good_pred_idx:
    if idx not in selected and len(selected) < 13:
        selected.append(idx)

# Add bad predictions (failures)
for idx in bad_pred_idx:
    if idx not in selected and len(selected) < 16:
        selected.append(idx)

# Fill remaining if needed
remaining = [i for i in range(len(y_test)) if i not in selected]
while len(selected) < 16 and remaining:
    selected.append(remaining.pop(0))

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.patch.set_facecolor('#FAFAFA')

for i, ax in enumerate(axes.flat):
    if i >= len(selected):
        ax.axis('off')
        continue

    idx = selected[i]
    ax.imshow(X_test[idx])

    actual = y_test[idx]
    predicted = y_pred[idx]
    error = abs(predicted - actual)

    # Color based on error quality
    if error < 0.05:
        border_color = '#2E7D32'  # Green = good
        status = "GOOD"
    elif error < 0.15:
        border_color = '#FF9800'  # Orange = okay
        status = "FAIR"
    else:
        border_color = '#C62828'  # Red = poor
        status = "POOR"

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)

    # Determine direction
    if actual > 0.05:
        direction = "→ RIGHT"
    elif actual < -0.05:
        direction = "← LEFT"
    else:
        direction = "↑ STRAIGHT"

    ax.set_title(f"Actual: {actual:.3f}  |  Pred: {predicted:.3f}\n{direction} | {status} (err: {error:.3f})",
                 fontsize=9, fontweight='bold', color=border_color, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Sample Steering Predictions — Diverse Scenarios\n(Green=Good, Orange=Fair, Red=Poor Prediction)",
             fontsize=16, fontweight='bold', color='#333333', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: sample_predictions.png ({len(selected)} samples)")


# #####################################################
#
#   STEP 7 :  SCENARIO EXAMPLES
#   Straight driving, Left turn, Right turn
#
# #####################################################

print("\n" + "=" * 60)
print("  STEP 7 :  SCENARIO EXAMPLES")
print("=" * 60)


def draw_steering_indicator(img, angle, pred_angle, img_size=(200, 66)):
    """Draw a steering direction indicator on the image."""
    display = (img * 255).astype(np.uint8).copy()

    h, w = display.shape[:2]

    # Draw actual steering arrow (green)
    center_x, center_y = w // 2, h - 5
    end_x = int(center_x + angle * w * 0.4)
    end_y = center_y - 15
    cv2.arrowedLine(display, (center_x, center_y), (end_x, end_y), (0, 200, 0), 2, tipLength=0.3)

    # Draw predicted steering arrow (blue)
    pred_end_x = int(center_x + pred_angle * w * 0.4)
    cv2.arrowedLine(display, (center_x, center_y - 2), (pred_end_x, end_y - 2), (0, 100, 255), 2, tipLength=0.3)

    return display


def generate_scenario(indices, title, color, filename, scenario_label, n=6):
    """Generate a scenario visualization with n images."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
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
        display = draw_steering_indicator(X_test[i], y_test[i], y_pred[i])
        ax.imshow(display)

        actual = y_test[i]
        predicted = y_pred[i]
        error = abs(predicted - actual)

        ax.set_title(f"Actual: {actual:.3f}  |  Predicted: {predicted:.3f}\nError: {error:.4f}",
                     fontsize=11, fontweight='bold', color='white', pad=8)
        ax.axis('off')

    fig.suptitle(f"{scenario_label}\n(Green Arrow = Actual, Blue Arrow = Predicted)",
                 fontsize=20, fontweight='bold', color=color, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {filename}")


# Scenario 1: Straight Driving
straight_indices = [i for i in range(len(y_test)) if abs(y_test[i]) < 0.03]
generate_scenario(straight_indices, "Straight Driving", '#4CAF50',
                  "scenario_straight.png", "SCENARIO 1 : Straight Road Driving")

# Scenario 2: Left Turn
left_indices = [i for i in range(len(y_test)) if y_test[i] < -0.15]
if len(left_indices) < 3:
    left_indices = [i for i in range(len(y_test)) if y_test[i] < -0.05]
generate_scenario(left_indices, "Left Turn", '#2196F3',
                  "scenario_left_turn.png", "SCENARIO 2 : Left Turn / Curve")

# Scenario 3: Right Turn
right_indices = [i for i in range(len(y_test)) if y_test[i] > 0.15]
if len(right_indices) < 3:
    right_indices = [i for i in range(len(y_test)) if y_test[i] > 0.05]
generate_scenario(right_indices, "Right Turn", '#E91E63',
                  "scenario_right_turn.png", "SCENARIO 3 : Right Turn / Curve")


# #####################################################
#
#   ALL DONE!
#
# #####################################################

print("\n\n" + "=" * 60)
print("  ALL DONE! STEERING PREDICTION RESULTS SAVED")
print("=" * 60)
print()
print("  FILES CREATED:")
print("  " + "-" * 50)
print("  MODEL:")
print("    steering_model.h5")
print("    regression_report.txt")
print()
print("  TRAINING CHARTS:")
print("    training_testing_loss.png")
print("    training_testing_mae.png")
print("    combined_loss_mae.png")
print("    epoch_table.png")
print()
print("  EVALUATION CHARTS:")
print("    actual_vs_predicted.png")
print("    steering_distribution.png")
print("    error_distribution.png")
print("    residual_plot.png")
print("    all_metrics_combined.png")
print("    metrics_radar_chart.png")
print()
print("  PREDICTIONS:")
print("    sample_predictions.png")
print("    scenario_straight.png")
print("    scenario_left_turn.png")
print("    scenario_right_turn.png")
print("  " + "-" * 50)
print(f"\n  All saved to: {OUTPUT_DIR}")
print("=" * 60 + "\n")
