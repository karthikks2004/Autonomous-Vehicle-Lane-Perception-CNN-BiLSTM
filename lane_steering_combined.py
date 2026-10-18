import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as snsy
import time
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


# =============================================================
#
#   COMBINED SCRIPT: LANE DETECTION + STEERING PREDICTION
#
#   PART A :  STEERING WHEEL PREDICTION  (CNN Regression)
#   PART B :  LANE DETECTION             (U-Net Segmentation)
#
#   Results saved to separate folders under D:\New parttss all comined results\
#
# =============================================================


# =====================================================
# GLOBAL CONFIGURATION
# =====================================================

RANDOM_STATE = 42
DPI          = 300      # High resolution for all figures

# --- Steering Prediction Config ---
STEERING_JUNGLE_DIR = r"D:\Finalyear pro\steering prediction\archive\self_driving_car_dataset_jungle"
STEERING_MAKE_DIR   = r"D:\Finalyear pro\steering prediction\archive\self_driving_car_dataset_make"
STEERING_OUTPUT_DIR = r"D:\New parttss all comined results\steering_prediction"

STEER_IMAGE_H  = 66        # NVIDIA paper standard
STEER_IMAGE_W  = 200
STEER_BATCH    = 64
STEER_EPOCHS   = 20
STEER_SPLIT    = 0.2

# --- Lane Detection Config ---
TUSIMPLE_DIR     = r"D:\Finalyear pro\lane detection\TUSimple"
LANE_TRAIN_DIR   = os.path.join(TUSIMPLE_DIR, "train_set")
LANE_OUTPUT_DIR  = r"D:\New parttss all comined results\lane_detection"

LANE_IMAGE_H  = 256
LANE_IMAGE_W  = 512
LANE_BATCH    = 8           # Smaller batch for segmentation (larger images)
LANE_EPOCHS   = 20
LANE_SPLIT    = 0.2
LANE_WIDTH    = 8           # Pixel width of lane line in mask

# Create output directories
os.makedirs(STEERING_OUTPUT_DIR, exist_ok=True)
os.makedirs(LANE_OUTPUT_DIR, exist_ok=True)


# =====================================================
# VERIFY ALL INPUT FOLDERS EXIST
# =====================================================

print("=" * 65)
print("  VERIFYING INPUT FOLDERS")
print("=" * 65)

all_ok = True
for d, name in [
    (STEERING_JUNGLE_DIR, "Steering - Jungle dataset"),
    (STEERING_MAKE_DIR,   "Steering - Make dataset"),
    (LANE_TRAIN_DIR,      "Lane Detection - TUSimple train"),
]:
    if os.path.exists(d):
        print(f"  ✓  {name}")
    else:
        print(f"  ✗  {name}  NOT FOUND: {d}")
        all_ok = False

if not all_ok:
    print("\n  ERROR: Some folders are missing. Please check paths.")
    exit()

print("\n  All input folders verified!\n")


# *************************************************************
# *************************************************************
#
#   PART A :  STEERING WHEEL PREDICTION
#             CNN Regression using NVIDIA Architecture
#
# *************************************************************
# *************************************************************

print("\n" + "#" * 65)
print("#" * 65)
print("##")
print("##   PART A :  STEERING WHEEL PREDICTION")
print("##   Architecture: NVIDIA-inspired CNN")
print("##   Dataset: Self-Driving Car Simulator (Jungle + Make)")
print(f"##   Output: {STEERING_OUTPUT_DIR}")
print("##")
print("#" * 65)
print("#" * 65)


# #####################################################
#
#   A-STEP 1 :  LOAD STEERING DATASET
#
# #####################################################

print("\n" + "=" * 65)
print("  A-STEP 1 :  LOADING STEERING DATASET")
print("=" * 65)

steer_start = time.time()


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

    df = pd.read_csv(csv_path, header=None,
                     names=["center", "left", "right", "steering", "throttle", "reverse", "speed"])

    images = []
    angles = []
    skipped = 0

    for idx, row in df.iterrows():
        if (idx + 1) % 2000 == 0 or idx == 0:
            print(f"    Processing {idx + 1}/{len(df)}...")

        center_filename = os.path.basename(row["center"].strip())
        img_path = os.path.join(img_dir, center_filename)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        # Crop sky (top 60px) and car hood (bottom 25px)
        img = img[60:135, :, :]
        img = cv2.resize(img, (STEER_IMAGE_W, STEER_IMAGE_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        images.append(img)
        angles.append(float(row["steering"]))

    print(f"    Loaded : {len(images)} images")
    if skipped > 0:
        print(f"    Skipped: {skipped} (file not found or unreadable)")

    return images, angles


images_jungle, angles_jungle = load_driving_data(STEERING_JUNGLE_DIR, "Jungle")
images_make,   angles_make   = load_driving_data(STEERING_MAKE_DIR,   "Make")

steer_images = images_jungle + images_make
steer_angles = angles_jungle + angles_make
del images_jungle, images_make, angles_jungle, angles_make

X_steer = np.array(steer_images, dtype=np.float32)
y_steer = np.array(steer_angles, dtype=np.float32)
del steer_images

steer_load_time = time.time() - steer_start

print(f"\n  Combined Steering Dataset:")
print(f"  Total samples  : {len(X_steer)}")
print(f"  Image shape    : {X_steer.shape[1:]}")
print(f"  Memory         : {X_steer.nbytes / (1024**3):.2f} GB")
print(f"  Loaded in      : {steer_load_time:.1f} seconds")
print(f"  Steering range : [{y_steer.min():.4f}, {y_steer.max():.4f}]")
print(f"  Steering mean  : {y_steer.mean():.4f}")
print(f"  Steering std   : {y_steer.std():.4f}")

if len(X_steer) == 0:
    print("\n  ERROR: No valid steering data found.")
    exit()


# #####################################################
#
#   A-STEP 2 :  PREPARE STEERING DATA
#
# #####################################################

print("\n" + "=" * 65)
print("  A-STEP 2 :  PREPARING STEERING DATA")
print("=" * 65)

X_steer_train, X_steer_test, y_steer_train, y_steer_test = train_test_split(
    X_steer, y_steer,
    test_size=STEER_SPLIT,
    random_state=RANDOM_STATE
)
del X_steer

print(f"  Training : {len(X_steer_train)}")
print(f"  Testing  : {len(X_steer_test)}")


# #####################################################
#
#   A-STEP 3 :  BUILD & TRAIN NVIDIA CNN
#
# #####################################################

print("\n" + "=" * 65)
print("  A-STEP 3 :  BUILDING & TRAINING NVIDIA CNN")
print("=" * 65)

steer_model = Sequential()

steer_model.add(Lambda(lambda x: x - 0.5, input_shape=(STEER_IMAGE_H, STEER_IMAGE_W, 3)))

# NVIDIA Architecture - 5 Convolutional layers
steer_model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
steer_model.add(BatchNormalization())

steer_model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
steer_model.add(BatchNormalization())

steer_model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
steer_model.add(BatchNormalization())

steer_model.add(Conv2D(64, (3, 3), activation='elu'))
steer_model.add(BatchNormalization())

steer_model.add(Conv2D(64, (3, 3), activation='elu'))
steer_model.add(BatchNormalization())

steer_model.add(Flatten())
steer_model.add(Dense(100, activation='elu'))
steer_model.add(Dropout(0.3))
steer_model.add(Dense(50, activation='elu'))
steer_model.add(Dropout(0.3))
steer_model.add(Dense(10, activation='elu'))
steer_model.add(Dense(1))  # Single output: steering angle

steer_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
steer_model.summary()

steer_early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
steer_reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n  Training started...\n")
steer_train_start = time.time()

steer_history = steer_model.fit(
    X_steer_train, y_steer_train,
    validation_data=(X_steer_test, y_steer_test),
    epochs=STEER_EPOCHS,
    batch_size=STEER_BATCH,
    callbacks=[steer_early_stop, steer_reduce_lr],
    verbose=1
)

steer_train_time = time.time() - steer_train_start
print(f"\n  Training done in {steer_train_time:.1f} seconds")

steer_model.save(os.path.join(STEERING_OUTPUT_DIR, "steering_model.h5"))
print(f"  Model saved: steering_model.h5")

s_train_loss = steer_history.history['loss']
s_val_loss   = steer_history.history['val_loss']
s_train_mae  = steer_history.history['mae']
s_val_mae    = steer_history.history['val_mae']
s_epoch_list = range(1, len(s_train_loss) + 1)


# #####################################################
#
#   A-STEP 4 :  EVALUATE STEERING MODEL
#
# #####################################################

print("\n" + "=" * 65)
print("  A-STEP 4 :  STEERING MODEL EVALUATION")
print("=" * 65)

y_steer_pred = steer_model.predict(X_steer_test, verbose=0).flatten()

s_mae  = mean_absolute_error(y_steer_test, y_steer_pred)
s_mse  = mean_squared_error(y_steer_test, y_steer_pred)
s_rmse = np.sqrt(s_mse)
s_r2   = r2_score(y_steer_test, y_steer_pred)

print(f"\n  MAE  : {s_mae:.6f}")
print(f"  MSE  : {s_mse:.6f}")
print(f"  RMSE : {s_rmse:.6f}")
print(f"  R²   : {s_r2:.6f}")

with open(os.path.join(STEERING_OUTPUT_DIR, "regression_report.txt"), "w") as f:
    f.write("STEERING WHEEL PREDICTION - CNN REGRESSION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model: NVIDIA-inspired CNN (5 Conv + 3 Dense)\n")
    f.write(f"Total Samples: {len(X_steer_train) + len(X_steer_test)}\n")
    f.write(f"Training: {len(X_steer_train)} | Testing: {len(X_steer_test)}\n\n")
    f.write("REGRESSION METRICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"  MAE   : {s_mae:.6f}\n")
    f.write(f"  MSE   : {s_mse:.6f}\n")
    f.write(f"  RMSE  : {s_rmse:.6f}\n")
    f.write(f"  R²    : {s_r2:.6f}\n\n")
    f.write(f"Training Time: {steer_train_time:.1f} seconds\n")
    f.write(f"Epochs Trained: {len(s_train_loss)}\n")
print("  Saved: regression_report.txt")


# #####################################################
#
#   A-STEP 5 :  STEERING CHARTS
#
# #####################################################

print("\n" + "=" * 65)
print("  A-STEP 5 :  GENERATING STEERING CHARTS")
print("=" * 65)

# --- Training vs Testing Loss ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.plot(s_epoch_list, s_train_loss, color='#E91E63', linewidth=2.5, marker='o', markersize=6, label='Training Loss')
ax.plot(s_epoch_list, s_val_loss,   color='#2196F3', linewidth=2.5, marker='s', markersize=6, label='Testing Loss')
ax.fill_between(s_epoch_list, s_train_loss, alpha=0.1, color='#E91E63')
ax.fill_between(s_epoch_list, s_val_loss,   alpha=0.1, color='#2196F3')
ax.set_xlabel("Epoch", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Loss (MSE)", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Training Loss vs Testing Loss — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.annotate(f'Train: {s_train_loss[-1]:.6f}', xy=(len(s_train_loss), s_train_loss[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#E91E63',
            arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))
ax.annotate(f'Test: {s_val_loss[-1]:.6f}', xy=(len(s_val_loss), s_val_loss[-1]),
            xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold', color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5))
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "training_testing_loss.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: training_testing_loss.png")

# --- Training vs Testing MAE ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.plot(s_epoch_list, s_train_mae, color='#4CAF50', linewidth=2.5, marker='o', markersize=6, label='Training MAE')
ax.plot(s_epoch_list, s_val_mae,   color='#FF9800', linewidth=2.5, marker='s', markersize=6, label='Testing MAE')
ax.fill_between(s_epoch_list, s_train_mae, alpha=0.1, color='#4CAF50')
ax.fill_between(s_epoch_list, s_val_mae,   alpha=0.1, color='#FF9800')
ax.set_xlabel("Epoch", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("MAE", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Training MAE vs Testing MAE — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.annotate(f'Train: {s_train_mae[-1]:.6f}', xy=(len(s_train_mae), s_train_mae[-1]),
            xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))
ax.annotate(f'Test: {s_val_mae[-1]:.6f}', xy=(len(s_val_mae), s_val_mae[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#FF9800',
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5))
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "training_testing_mae.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: training_testing_mae.png")

# --- Combined Loss + MAE side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#FAFAFA')
ax1.set_facecolor('#FAFAFA')
ax1.plot(s_epoch_list, s_train_loss, color='#E91E63', linewidth=2.5, marker='o', markersize=5, label='Train Loss')
ax1.plot(s_epoch_list, s_val_loss,   color='#2196F3', linewidth=2.5, marker='s', markersize=5, label='Test Loss')
ax1.fill_between(s_epoch_list, s_train_loss, alpha=0.1, color='#E91E63')
ax1.fill_between(s_epoch_list, s_val_loss,   alpha=0.1, color='#2196F3')
ax1.set_title("Training & Testing Loss", fontsize=14, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=12, fontweight='bold'); ax1.set_ylabel("Loss (MSE)", fontsize=12, fontweight='bold')
ax1.legend(fontsize=11); ax1.grid(True, alpha=0.2)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.set_facecolor('#FAFAFA')
ax2.plot(s_epoch_list, s_train_mae, color='#4CAF50', linewidth=2.5, marker='o', markersize=5, label='Train MAE')
ax2.plot(s_epoch_list, s_val_mae,   color='#FF9800', linewidth=2.5, marker='s', markersize=5, label='Test MAE')
ax2.fill_between(s_epoch_list, s_train_mae, alpha=0.1, color='#4CAF50')
ax2.fill_between(s_epoch_list, s_val_mae,   alpha=0.1, color='#FF9800')
ax2.set_title("Training & Testing MAE", fontsize=14, fontweight='bold')
ax2.set_xlabel("Epoch", fontsize=12, fontweight='bold'); ax2.set_ylabel("MAE", fontsize=12, fontweight='bold')
ax2.legend(fontsize=11); ax2.grid(True, alpha=0.2)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.suptitle("Model Training Progress — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "combined_loss_mae.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: combined_loss_mae.png")

# --- Actual vs Predicted Scatter ---
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.scatter(y_steer_test, y_steer_pred, alpha=0.3, s=10, color='#2196F3', edgecolors='none', label='Predictions')
min_v = min(y_steer_test.min(), y_steer_pred.min())
max_v = max(y_steer_test.max(), y_steer_pred.max())
ax.plot([min_v, max_v], [min_v, max_v], '--', linewidth=2, color='#E91E63', label='Perfect Prediction')
ax.set_xlabel("Actual Steering Angle", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Predicted Steering Angle", fontsize=14, fontweight='bold', color='#333333')
ax.set_title(f"Actual vs Predicted Steering Angle\nR² = {s_r2:.4f}", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "actual_vs_predicted.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: actual_vs_predicted.png")

# --- Steering Distribution ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.hist(y_steer_test, bins=50, alpha=0.6, color='#2196F3', edgecolor='white', linewidth=1, label='Actual', density=True)
ax.hist(y_steer_pred, bins=50, alpha=0.6, color='#FF9800', edgecolor='white', linewidth=1, label='Predicted', density=True)
ax.set_xlabel("Steering Angle", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Density", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Steering Angle Distribution — Actual vs Predicted", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "steering_distribution.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: steering_distribution.png")

# --- Error Distribution ---
s_errors = y_steer_pred - y_steer_test
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.hist(s_errors, bins=50, alpha=0.7, color='#4CAF50', edgecolor='white', linewidth=1)
ax.axvline(x=0, color='#E91E63', linestyle='--', linewidth=2, label='Zero Error')
ax.axvline(x=s_errors.mean(), color='#FF9800', linestyle='--', linewidth=2, label=f'Mean Error: {s_errors.mean():.4f}')
ax.set_xlabel("Prediction Error (Predicted - Actual)", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Frequency", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Prediction Error Distribution", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12, framealpha=0.9); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "error_distribution.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: error_distribution.png")

# --- Residual Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.scatter(y_steer_pred, s_errors, alpha=0.3, s=10, color='#9C27B0', edgecolors='none')
ax.axhline(y=0, color='#E91E63', linestyle='--', linewidth=2)
ax.set_xlabel("Predicted Steering Angle", fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel("Residual (Error)", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Residual Plot — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.grid(True, alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "residual_plot.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: residual_plot.png")

# --- All Metrics Combined ---
s_metric_cfg = [
    {"name": "MAE",  "value": s_mae,  "color": "#2196F3"},
    {"name": "MSE",  "value": s_mse,  "color": "#FF9800"},
    {"name": "RMSE", "value": s_rmse, "color": "#4CAF50"},
    {"name": "R²",   "value": s_r2,   "color": "#E91E63"},
]
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
sm_names  = [c["name"] for c in s_metric_cfg]
sm_values = [c["value"] for c in s_metric_cfg]
sm_colors = [c["color"] for c in s_metric_cfg]
bars = ax.bar(sm_names, sm_values, color=sm_colors, width=0.35, edgecolor='white', linewidth=2, zorder=3)
for bar, val, color in zip(bars, sm_values, sm_colors):
    ax.text(bar.get_x() + bar.get_width()/2, val + max(sm_values)*0.03,
            f'{val:.6f}', ha='center', va='bottom', fontsize=13, fontweight='bold', color=color)
ax.set_ylabel("Score", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("Regression Metrics — Steering Wheel Prediction", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', colors='#666666', labelsize=12); ax.yaxis.grid(True, alpha=0.2, color='#999999')
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "all_metrics_combined.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: all_metrics_combined.png")

# --- Radar Chart ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#FAFAFA')
max_err = max(s_mae, s_mse, s_rmse) if max(s_mae, s_mse, s_rmse) > 0 else 1
radar_vals = [1 - s_mae/max_err, 1 - s_mse/max_err, 1 - s_rmse/max_err, max(0, s_r2)]
radar_labs = ["MAE\n(lower=better)", "MSE\n(lower=better)", "RMSE\n(lower=better)", "R² Score"]
vr = radar_vals + [radar_vals[0]]
angs = np.linspace(0, 2*np.pi, len(radar_labs), endpoint=False).tolist() + [0]
ax.fill(angs, vr, color='#2196F3', alpha=0.25)
ax.plot(angs, vr, color='#2196F3', linewidth=2.5, marker='o', markersize=8)
ax.set_xticks(angs[:-1])
ax.set_xticklabels(radar_labs, fontsize=11, fontweight='bold', color='#333333')
ax.set_ylim(0, 1.1)
ax.set_title("Metrics Radar Chart — Steering Prediction", fontsize=16, fontweight='bold', color='#333333', pad=30)
actual_v = [s_mae, s_mse, s_rmse, s_r2]
for a, v, av in zip(angs[:-1], radar_vals, actual_v):
    ax.text(a, v + 0.08, f'{av:.4f}', ha='center', va='center', fontsize=10, fontweight='bold', color='#2196F3')
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "metrics_radar_chart.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: metrics_radar_chart.png")

# --- Epoch Table ---
fig, ax = plt.subplots(figsize=(12, max(4, len(s_train_loss) * 0.4 + 1)))
fig.patch.set_facecolor('#FAFAFA'); ax.axis('off')
tbl_data = []
for e in range(len(s_train_loss)):
    tbl_data.append([f"Epoch {e+1}", f"{s_train_loss[e]:.6f}", f"{s_val_loss[e]:.6f}",
                     f"{s_train_mae[e]:.6f}", f"{s_val_mae[e]:.6f}"])
col_labs = ["Epoch", "Train Loss", "Test Loss", "Train MAE", "Test MAE"]
tbl = ax.table(cellText=tbl_data, colLabels=col_labs, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.5)
for j in range(len(col_labs)):
    tbl[0, j].set_facecolor('#2196F3'); tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(tbl_data) + 1):
    for j in range(len(col_labs)):
        tbl[i, j].set_facecolor('#E3F2FD' if i % 2 == 0 else '#FFFFFF')
ax.set_title("Epoch-by-Epoch Results — Steering Prediction", fontsize=14, fontweight='bold', color='#333333', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "epoch_table.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: epoch_table.png")


# #####################################################
#
#   A-STEP 6 :  SAMPLE PREDICTIONS
#
# #####################################################

print("\n" + "=" * 65)
print("  A-STEP 6 :  STEERING SAMPLE PREDICTIONS")
print("=" * 65)

straight_idx = [i for i in range(len(y_steer_test)) if abs(y_steer_test[i]) < 0.05]
left_idx     = [i for i in range(len(y_steer_test)) if y_steer_test[i] < -0.1]
right_idx    = [i for i in range(len(y_steer_test)) if y_steer_test[i] > 0.1]

abs_errors = np.abs(y_steer_pred - y_steer_test)
good_idx = np.argsort(abs_errors)[:50].tolist()
bad_idx  = np.argsort(abs_errors)[-50:].tolist()[::-1]

np.random.seed(RANDOM_STATE)
selected = []
if len(straight_idx) >= 4: selected += list(np.random.choice(straight_idx, 4, replace=False))
elif straight_idx: selected += straight_idx[:4]
if len(left_idx) >= 3: selected += list(np.random.choice(left_idx, 3, replace=False))
elif left_idx: selected += left_idx[:3]
if len(right_idx) >= 3: selected += list(np.random.choice(right_idx, 3, replace=False))
elif right_idx: selected += right_idx[:3]

for i in good_idx:
    if i not in selected and len(selected) < 13: selected.append(i)
for i in bad_idx:
    if i not in selected and len(selected) < 16: selected.append(i)

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.patch.set_facecolor('#FAFAFA')
for i, ax in enumerate(axes.flat):
    if i >= len(selected): ax.axis('off'); continue
    idx = selected[i]
    ax.imshow(X_steer_test[idx])
    actual, predicted = y_steer_test[idx], y_steer_pred[idx]
    error = abs(predicted - actual)
    if error < 0.05: bc, status = '#2E7D32', "GOOD"
    elif error < 0.15: bc, status = '#FF9800', "FAIR"
    else: bc, status = '#C62828', "POOR"
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor(bc); sp.set_linewidth(4)
    direction = "→ RIGHT" if actual > 0.05 else ("← LEFT" if actual < -0.05 else "↑ STRAIGHT")
    ax.set_title(f"Actual: {actual:.3f} | Pred: {predicted:.3f}\n{direction} | {status} (err: {error:.3f})",
                 fontsize=9, fontweight='bold', color=bc, pad=6)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Steering Predictions — Diverse Scenarios\n(Green=Good, Orange=Fair, Red=Poor)",
             fontsize=16, fontweight='bold', color='#333333', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(STEERING_OUTPUT_DIR, "sample_predictions.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  Saved: sample_predictions.png ({len(selected)} samples)")


# #####################################################
#
#   A-STEP 7 :  STEERING SCENARIOS
#
# #####################################################

print("\n" + "=" * 65)
print("  A-STEP 7 :  STEERING SCENARIOS")
print("=" * 65)


def draw_steering_arrow(img, angle, pred_angle):
    display = (img * 255).astype(np.uint8).copy()
    h, w = display.shape[:2]
    cx, cy = w // 2, h - 5
    ex = int(cx + angle * w * 0.4)
    cv2.arrowedLine(display, (cx, cy), (ex, cy - 15), (0, 200, 0), 2, tipLength=0.3)
    pex = int(cx + pred_angle * w * 0.4)
    cv2.arrowedLine(display, (cx, cy - 2), (pex, cy - 17), (0, 100, 255), 2, tipLength=0.3)
    return display


def gen_steer_scenario(indices, title, color, filename, n=6):
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    fig.patch.set_facecolor('#1a1a2e')
    np.random.seed(RANDOM_STATE)
    chosen = np.random.choice(indices, min(n, len(indices)), replace=False) if len(indices) > n else indices[:n]
    for ii, ax in enumerate(axes.flat):
        if ii >= len(chosen): ax.axis('off'); continue
        i = chosen[ii]
        display = draw_steering_arrow(X_steer_test[i], y_steer_test[i], y_steer_pred[i])
        ax.imshow(display)
        ax.set_title(f"Actual: {y_steer_test[i]:.3f} | Pred: {y_steer_pred[i]:.3f}\nError: {abs(y_steer_pred[i]-y_steer_test[i]):.4f}",
                     fontsize=11, fontweight='bold', color='white', pad=8)
        ax.axis('off')
    fig.suptitle(f"{title}\n(Green Arrow = Actual, Blue Arrow = Predicted)",
                 fontsize=20, fontweight='bold', color=color, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(STEERING_OUTPUT_DIR, filename), dpi=DPI, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {filename}")


s_straight = [i for i in range(len(y_steer_test)) if abs(y_steer_test[i]) < 0.03]
s_left     = [i for i in range(len(y_steer_test)) if y_steer_test[i] < -0.05]
s_right    = [i for i in range(len(y_steer_test)) if y_steer_test[i] > 0.05]

if s_straight: gen_steer_scenario(s_straight, "SCENARIO 1 : Straight Road Driving", '#4CAF50', "scenario_straight.png")
if s_left:     gen_steer_scenario(s_left,     "SCENARIO 2 : Left Turn / Curve",     '#2196F3', "scenario_left_turn.png")
if s_right:    gen_steer_scenario(s_right,    "SCENARIO 3 : Right Turn / Curve",    '#E91E63', "scenario_right_turn.png")

print("\n" + "=" * 65)
print("  ✓  PART A COMPLETE — Steering Prediction Done!")
print("=" * 65)

# Free steering data memory
del X_steer_train, X_steer_test, y_steer_train, y_steer_test, y_steer_pred
del steer_model


# *************************************************************
# *************************************************************
#
#   PART B :  LANE DETECTION
#             U-Net Semantic Segmentation
#
# *************************************************************
# *************************************************************

print("\n\n" + "#" * 65)
print("#" * 65)
print("##")
print("##   PART B :  LANE DETECTION")
print("##   Architecture: U-Net (Encoder-Decoder)")
print("##   Dataset: TUSimple Lane Detection Challenge")
print(f"##   Output: {LANE_OUTPUT_DIR}")
print("##")
print("#" * 65)
print("#" * 65)


# =====================================================
# Custom metrics for segmentation
# =====================================================

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_cast = K.cast(y_true, K.floatx())
    y_true_f = K.flatten(y_true_cast)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    y_true_cast = K.cast(y_true, K.floatx())
    bce = K.mean(K.binary_crossentropy(y_true_cast, y_pred))
    return bce + dice_loss(y_true_cast, y_pred)

def iou_metric(y_true, y_pred, smooth=1.0):
    y_true_cast = K.cast(y_true, K.floatx())
    y_pred_bin = K.cast(K.greater(y_pred, 0.5), K.floatx())
    intersection = K.sum(y_true_cast * y_pred_bin)
    union = K.sum(y_true_cast) + K.sum(y_pred_bin) - intersection
    return (intersection + smooth) / (union + smooth)


# #####################################################
#
#   B-STEP 1 :  LOAD TUSIMPLE DATASET
#
# #####################################################

print("\n" + "=" * 65)
print("  B-STEP 1 :  LOADING TUSIMPLE DATASET")
print("=" * 65)

lane_start = time.time()


def generate_lane_mask(lanes, h_samples, img_shape, lane_width=LANE_WIDTH):
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    for lane in lanes:
        points = [(int(x), int(y)) for x, y in zip(lane, h_samples) if x != -2]
        if len(points) >= 2:
            cv2.polylines(mask, [np.array(points, dtype=np.int32)], False, 255, lane_width)
    return mask


def load_tusimple_labels(label_file):
    data = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


label_files = [
    os.path.join(LANE_TRAIN_DIR, "label_data_0313.json"),
    os.path.join(LANE_TRAIN_DIR, "label_data_0531.json"),
    os.path.join(LANE_TRAIN_DIR, "label_data_0601.json"),
]

all_labels = []
for lf in label_files:
    if os.path.exists(lf):
        labels = load_tusimple_labels(lf)
        print(f"  Loaded {len(labels)} entries from {os.path.basename(lf)}")
        all_labels.extend(labels)

print(f"\n  Total label entries: {len(all_labels)}")

lane_images = []
lane_masks  = []
skipped = 0

for i, entry in enumerate(all_labels):
    if (i + 1) % 500 == 0 or i == 0:
        print(f"  Processing {i + 1}/{len(all_labels)}...")

    img_path = os.path.join(LANE_TRAIN_DIR, entry["raw_file"])
    if not os.path.exists(img_path):
        skipped += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue

    orig_h, orig_w = img.shape[:2]
    mask = generate_lane_mask(entry["lanes"], entry["h_samples"], (orig_h, orig_w))

    img  = cv2.resize(img, (LANE_IMAGE_W, LANE_IMAGE_H))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.resize(mask, (LANE_IMAGE_W, LANE_IMAGE_H), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)

    lane_images.append(img)
    lane_masks.append(mask)

lane_load_time = time.time() - lane_start

X_lane = np.array(lane_images, dtype=np.uint8)
Y_lane = np.array(lane_masks,  dtype=np.uint8)[..., np.newaxis]
del lane_images, lane_masks

print(f"\n  Loaded in {lane_load_time:.1f} seconds")
print(f"  Total samples : {X_lane.shape[0]}")
print(f"  Image shape   : {X_lane.shape[1:]}")
print(f"  Mask shape    : {Y_lane.shape[1:]}")
print(f"  Memory        : {(X_lane.nbytes + Y_lane.nbytes) / (1024**3):.2f} GB")
if skipped > 0:
    print(f"  Skipped       : {skipped}")

lane_pix = Y_lane.sum()
total_pix = Y_lane.size
print(f"  Lane pixels   : {lane_pix:,.0f} / {total_pix:,.0f} ({lane_pix/total_pix*100:.2f}%)")


# #####################################################
#
#   B-STEP 2 :  PREPARE LANE DATA
#
# #####################################################

print("\n" + "=" * 65)
print("  B-STEP 2 :  PREPARING LANE DATA")
print("=" * 65)

X_lane_train, X_lane_test, Y_lane_train, Y_lane_test = train_test_split(
    X_lane, Y_lane,
    test_size=LANE_SPLIT,
    random_state=RANDOM_STATE
)
del X_lane, Y_lane

print(f"  Training : {len(X_lane_train)}")
print(f"  Testing  : {len(X_lane_test)}")


# #####################################################
#
#   B-STEP 3 :  BUILD & TRAIN U-NET
#
# #####################################################

print("\n" + "=" * 65)
print("  B-STEP 3 :  BUILDING & TRAINING U-NET")
print("=" * 65)


def build_unet(input_shape=(LANE_IMAGE_H, LANE_IMAGE_W, 3)):
    inputs = Input(input_shape)

    # Scale from [0, 255] to [0.0, 1.0] dynamically
    rescaled = Lambda(lambda x: x / 255.0)(inputs)

    # Encoder Block 1
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(rescaled)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2,2))(c1)
    p1 = Dropout(0.1)(p1)

    # Encoder Block 2
    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2,2))(c2)
    p2 = Dropout(0.1)(p2)

    # Encoder Block 3
    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2,2))(c3)
    p3 = Dropout(0.2)(p3)

    # Encoder Block 4
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2,2))(c4)
    p4 = Dropout(0.2)(p4)

    # Bottleneck
    c5 = Conv2D(512, (3,3), activation='relu', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(512, (3,3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    # Decoder Block 6
    u6 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(0.2)(u6)
    c6 = Conv2D(256, (3,3), activation='relu', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(256, (3,3), activation='relu', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    # Decoder Block 7
    u7 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(0.2)(u7)
    c7 = Conv2D(128, (3,3), activation='relu', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(128, (3,3), activation='relu', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    # Decoder Block 8
    u8 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.1)(u8)
    c8 = Conv2D(64, (3,3), activation='relu', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(64, (3,3), activation='relu', padding='same')(c8)
    c8 = BatchNormalization()(c8)

    # Decoder Block 9
    u9 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(0.1)(u9)
    c9 = Conv2D(32, (3,3), activation='relu', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(32, (3,3), activation='relu', padding='same')(c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    return Model(inputs=[inputs], outputs=[outputs])


lane_model = build_unet()
lane_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=combined_loss,
    metrics=['accuracy', dice_coefficient, iou_metric]
)
lane_model.summary()

lane_early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
lane_reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n  Training started...\n")
lane_train_start = time.time()

lane_history = lane_model.fit(
    X_lane_train, Y_lane_train,
    validation_data=(X_lane_test, Y_lane_test),
    epochs=LANE_EPOCHS,
    batch_size=LANE_BATCH,
    callbacks=[lane_early_stop, lane_reduce_lr],
    verbose=1
)

lane_train_time = time.time() - lane_train_start
print(f"\n  Training done in {lane_train_time:.1f} seconds")

lane_model.save(os.path.join(LANE_OUTPUT_DIR, "lane_model.h5"))
print(f"  Model saved: lane_model.h5")

l_train_loss = lane_history.history['loss']
l_val_loss   = lane_history.history['val_loss']
l_train_acc  = lane_history.history['accuracy']
l_val_acc    = lane_history.history['val_accuracy']
l_train_dice = lane_history.history['dice_coefficient']
l_val_dice   = lane_history.history['val_dice_coefficient']
l_train_iou  = lane_history.history['iou_metric']
l_val_iou    = lane_history.history['val_iou_metric']
l_epoch_list = range(1, len(l_train_loss) + 1)


# #####################################################
#
#   B-STEP 4 :  EVALUATE LANE MODEL
#
# #####################################################

print("\n" + "=" * 65)
print("  B-STEP 4 :  LANE MODEL EVALUATION")
print("=" * 65)

Y_lane_pred = lane_model.predict(X_lane_test, verbose=0)
Y_lane_pred_bin = (Y_lane_pred > 0.5).astype(np.float32)

y_true_flat = Y_lane_test.flatten().astype(int)
y_pred_flat = Y_lane_pred_bin.flatten().astype(int)

pixel_accuracy  = accuracy_score(y_true_flat, y_pred_flat)
pixel_precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
pixel_recall    = recall_score(y_true_flat, y_pred_flat, zero_division=0)
pixel_f1        = f1_score(y_true_flat, y_pred_flat, zero_division=0)

intersection = np.sum(Y_lane_test.flatten() * Y_lane_pred_bin.flatten())
union = np.sum(Y_lane_test.flatten()) + np.sum(Y_lane_pred_bin.flatten()) - intersection
l_iou_score  = intersection / (union + 1e-7)
l_dice_score = (2 * intersection) / (np.sum(Y_lane_test.flatten()) + np.sum(Y_lane_pred_bin.flatten()) + 1e-7)

eval_res = lane_model.evaluate(X_lane_test, Y_lane_test, verbose=0)
l_test_loss = eval_res[0]

print(f"\n  Test Loss        : {l_test_loss:.4f}")
print(f"  Pixel Accuracy   : {pixel_accuracy*100:.2f}%")
print(f"  Pixel Precision  : {pixel_precision*100:.2f}%")
print(f"  Pixel Recall     : {pixel_recall*100:.2f}%")
print(f"  Pixel F1 Score   : {pixel_f1*100:.2f}%")
print(f"  IoU (Jaccard)    : {l_iou_score*100:.2f}%")
print(f"  Dice Coefficient : {l_dice_score*100:.2f}%")

with open(os.path.join(LANE_OUTPUT_DIR, "segmentation_report.txt"), "w") as f:
    f.write("LANE DETECTION - U-NET SEGMENTATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model: U-Net (4 Encoder + 4 Decoder blocks)\n")
    f.write(f"Dataset: TUSimple ({len(X_lane_train) + len(X_lane_test)} samples)\n")
    f.write(f"Training: {len(X_lane_train)} | Testing: {len(X_lane_test)}\n")
    f.write(f"Image Size: {LANE_IMAGE_H}x{LANE_IMAGE_W}\n\n")
    f.write("SEGMENTATION METRICS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Test Loss        : {l_test_loss:.4f}\n")
    f.write(f"  Pixel Accuracy   : {pixel_accuracy*100:.2f}%\n")
    f.write(f"  Pixel Precision  : {pixel_precision*100:.2f}%\n")
    f.write(f"  Pixel Recall     : {pixel_recall*100:.2f}%\n")
    f.write(f"  Pixel F1 Score   : {pixel_f1*100:.2f}%\n")
    f.write(f"  IoU (Jaccard)    : {l_iou_score*100:.2f}%\n")
    f.write(f"  Dice Coefficient : {l_dice_score*100:.2f}%\n\n")
    f.write(f"Training Time: {lane_train_time:.1f} seconds\n")
    f.write(f"Epochs Trained: {len(l_train_loss)}\n")
print("  Saved: segmentation_report.txt")


# #####################################################
#
#   B-STEP 5 :  LANE DETECTION CHARTS
#
# #####################################################

print("\n" + "=" * 65)
print("  B-STEP 5 :  GENERATING LANE DETECTION CHARTS")
print("=" * 65)

# --- Loss ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.plot(l_epoch_list, l_train_loss, color='#E91E63', linewidth=2.5, marker='o', markersize=6, label='Training Loss')
ax.plot(l_epoch_list, l_val_loss,   color='#2196F3', linewidth=2.5, marker='s', markersize=6, label='Testing Loss')
ax.fill_between(l_epoch_list, l_train_loss, alpha=0.1, color='#E91E63')
ax.fill_between(l_epoch_list, l_val_loss,   alpha=0.1, color='#2196F3')
ax.set_xlabel("Epoch", fontsize=14, fontweight='bold'); ax.set_ylabel("Loss", fontsize=14, fontweight='bold')
ax.set_title("Training Loss vs Testing Loss — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.annotate(f'Train: {l_train_loss[-1]:.4f}', xy=(len(l_train_loss), l_train_loss[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#E91E63',
            arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))
ax.annotate(f'Test: {l_val_loss[-1]:.4f}', xy=(len(l_val_loss), l_val_loss[-1]),
            xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold', color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5))
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "training_testing_loss.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: training_testing_loss.png")

# --- Accuracy ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.plot(l_epoch_list, l_train_acc, color='#4CAF50', linewidth=2.5, marker='o', markersize=6, label='Training Acc')
ax.plot(l_epoch_list, l_val_acc,   color='#FF9800', linewidth=2.5, marker='s', markersize=6, label='Testing Acc')
ax.fill_between(l_epoch_list, l_train_acc, alpha=0.1, color='#4CAF50')
ax.fill_between(l_epoch_list, l_val_acc,   alpha=0.1, color='#FF9800')
ax.set_xlabel("Epoch", fontsize=14, fontweight='bold'); ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
ax.set_title("Training vs Testing Accuracy — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(['0%','20%','40%','60%','80%','100%'])
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "training_testing_accuracy.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: training_testing_accuracy.png")

# --- IoU per Epoch ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.plot(l_epoch_list, l_train_iou, color='#9C27B0', linewidth=2.5, marker='o', markersize=6, label='Training IoU')
ax.plot(l_epoch_list, l_val_iou,   color='#00BCD4', linewidth=2.5, marker='s', markersize=6, label='Testing IoU')
ax.fill_between(l_epoch_list, l_train_iou, alpha=0.1, color='#9C27B0')
ax.fill_between(l_epoch_list, l_val_iou,   alpha=0.1, color='#00BCD4')
ax.set_xlabel("Epoch", fontsize=14, fontweight='bold'); ax.set_ylabel("IoU", fontsize=14, fontweight='bold')
ax.set_title("IoU per Epoch — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "iou_per_epoch.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: iou_per_epoch.png")

# --- Dice per Epoch ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
ax.plot(l_epoch_list, l_train_dice, color='#FF5722', linewidth=2.5, marker='o', markersize=6, label='Training Dice')
ax.plot(l_epoch_list, l_val_dice,   color='#607D8B', linewidth=2.5, marker='s', markersize=6, label='Testing Dice')
ax.fill_between(l_epoch_list, l_train_dice, alpha=0.1, color='#FF5722')
ax.fill_between(l_epoch_list, l_val_dice,   alpha=0.1, color='#607D8B')
ax.set_xlabel("Epoch", fontsize=14, fontweight='bold'); ax.set_ylabel("Dice", fontsize=14, fontweight='bold')
ax.set_title("Dice Coefficient per Epoch — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.legend(fontsize=12); ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "dice_per_epoch.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: dice_per_epoch.png")

# --- Combined 2x2 Training Progress ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#FAFAFA')

ax1.set_facecolor('#FAFAFA')
ax1.plot(l_epoch_list, l_train_loss, color='#E91E63', linewidth=2, marker='o', markersize=4, label='Train')
ax1.plot(l_epoch_list, l_val_loss,   color='#2196F3', linewidth=2, marker='s', markersize=4, label='Test')
ax1.set_title("Loss", fontsize=14, fontweight='bold'); ax1.legend(fontsize=10); ax1.grid(True, alpha=0.2)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.set_facecolor('#FAFAFA')
ax2.plot(l_epoch_list, l_train_acc, color='#4CAF50', linewidth=2, marker='o', markersize=4, label='Train')
ax2.plot(l_epoch_list, l_val_acc,   color='#FF9800', linewidth=2, marker='s', markersize=4, label='Test')
ax2.set_title("Pixel Accuracy", fontsize=14, fontweight='bold'); ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

ax3.set_facecolor('#FAFAFA')
ax3.plot(l_epoch_list, l_train_iou, color='#9C27B0', linewidth=2, marker='o', markersize=4, label='Train')
ax3.plot(l_epoch_list, l_val_iou,   color='#00BCD4', linewidth=2, marker='s', markersize=4, label='Test')
ax3.set_title("IoU Score", fontsize=14, fontweight='bold'); ax3.legend(fontsize=10); ax3.grid(True, alpha=0.2)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

ax4.set_facecolor('#FAFAFA')
ax4.plot(l_epoch_list, l_train_dice, color='#FF5722', linewidth=2, marker='o', markersize=4, label='Train')
ax4.plot(l_epoch_list, l_val_dice,   color='#607D8B', linewidth=2, marker='s', markersize=4, label='Test')
ax4.set_title("Dice Coefficient", fontsize=14, fontweight='bold'); ax4.legend(fontsize=10); ax4.grid(True, alpha=0.2)
ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

fig.suptitle("Model Training Progress — Lane Detection (U-Net)", fontsize=18, fontweight='bold', color='#333333', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "combined_training_progress.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: combined_training_progress.png")

# --- All Metrics Combined Bar Chart ---
l_metric_cfg = [
    {"name": "Pixel Acc",  "value": pixel_accuracy,  "color": "#2196F3"},
    {"name": "Precision",  "value": pixel_precision,  "color": "#FF9800"},
    {"name": "Recall",     "value": pixel_recall,     "color": "#4CAF50"},
    {"name": "F1 Score",   "value": pixel_f1,         "color": "#E91E63"},
    {"name": "IoU",        "value": l_iou_score,      "color": "#9C27B0"},
    {"name": "Dice",       "value": l_dice_score,     "color": "#00BCD4"},
]
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
lm_names  = [c["name"] for c in l_metric_cfg]
lm_values = [c["value"] for c in l_metric_cfg]
lm_colors = [c["color"] for c in l_metric_cfg]
bars = ax.bar(lm_names, lm_values, color=lm_colors, width=0.35, edgecolor='white', linewidth=2, zorder=3)
ax.axhline(y=1.0, color='#BDBDBD', linestyle='--', linewidth=1, alpha=0.7)
for bar, val, color in zip(bars, lm_values, lm_colors):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val*100:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=14, fontweight='bold', color='#333333')
ax.set_title("All Segmentation Metrics — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=20)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0]); ax.set_yticklabels(['0%','20%','40%','60%','80%','100%'])
ax.yaxis.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "all_metrics_combined.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: all_metrics_combined.png")

# --- Radar Chart ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#FAFAFA')
vr = lm_values + [lm_values[0]]
angs = np.linspace(0, 2*np.pi, len(lm_names), endpoint=False).tolist() + [0]
ax.fill(angs, vr, color='#9C27B0', alpha=0.25)
ax.plot(angs, vr, color='#9C27B0', linewidth=2.5, marker='o', markersize=8)
ax.set_xticks(angs[:-1])
ax.set_xticklabels(lm_names, fontsize=11, fontweight='bold', color='#333333')
ax.set_ylim(0, 1.1)
ax.set_title("Metrics Radar Chart — Lane Detection", fontsize=16, fontweight='bold', color='#333333', pad=30)
for a, v in zip(angs[:-1], lm_values):
    ax.text(a, v + 0.08, f'{v*100:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='#9C27B0')
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "metrics_radar_chart.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: metrics_radar_chart.png")

# --- Epoch Table ---
fig, ax = plt.subplots(figsize=(14, max(4, len(l_train_loss)*0.4+1)))
fig.patch.set_facecolor('#FAFAFA'); ax.axis('off')
tbl_data = []
for e in range(len(l_train_loss)):
    tbl_data.append([f"Epoch {e+1}", f"{l_train_loss[e]:.4f}", f"{l_val_loss[e]:.4f}",
                     f"{l_train_acc[e]*100:.2f}%", f"{l_val_acc[e]*100:.2f}%",
                     f"{l_train_iou[e]:.4f}", f"{l_val_iou[e]:.4f}"])
col_labs = ["Epoch", "Train Loss", "Test Loss", "Train Acc", "Test Acc", "Train IoU", "Test IoU"]
tbl = ax.table(cellText=tbl_data, colLabels=col_labs, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.5)
for j in range(len(col_labs)):
    tbl[0, j].set_facecolor('#9C27B0'); tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(tbl_data)+1):
    for j in range(len(col_labs)):
        tbl[i, j].set_facecolor('#F3E5F5' if i%2==0 else '#FFFFFF')
ax.set_title("Epoch-by-Epoch Results — Lane Detection", fontsize=14, fontweight='bold', color='#333333', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "epoch_table.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: epoch_table.png")


# #####################################################
#
#   B-STEP 6 :  LANE SAMPLE PREDICTIONS
#
# #####################################################

print("\n" + "=" * 65)
print("  B-STEP 6 :  LANE SAMPLE PREDICTIONS")
print("=" * 65)

np.random.seed(RANDOM_STATE)
sample_idx = np.random.choice(len(X_lane_test), size=min(8, len(X_lane_test)), replace=False)

fig, axes = plt.subplots(8, 4, figsize=(20, 32))
fig.patch.set_facecolor('#FAFAFA')
col_titles = ["Original Image", "Ground Truth Mask", "Predicted Mask", "Overlay"]

for row, si in enumerate(sample_idx):
    img       = X_lane_test[si].astype(np.float32) / 255.0
    gt_mask   = Y_lane_test[si, :, :, 0]
    pred_mask = Y_lane_pred_bin[si, :, :, 0]

    overlay = img.copy()
    overlay[gt_mask > 0]   = overlay[gt_mask > 0]   * 0.5 + np.array([0, 0.8, 0]) * 0.5
    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.5 + np.array([0, 0.5, 1.0]) * 0.5

    inter = np.sum(gt_mask * pred_mask)
    uni   = np.sum(gt_mask) + np.sum(pred_mask) - inter
    s_iou = inter / (uni + 1e-7)

    axes[row, 0].imshow(img);                                          axes[row, 0].axis('off')
    axes[row, 1].imshow(gt_mask, cmap='Greens', vmin=0, vmax=1);      axes[row, 1].axis('off')
    axes[row, 2].imshow(pred_mask, cmap='Blues', vmin=0, vmax=1);     axes[row, 2].axis('off')
    axes[row, 3].imshow(overlay);                                      axes[row, 3].axis('off')
    axes[row, 3].set_title(f"IoU: {s_iou:.4f}", fontsize=10, fontweight='bold',
                           color='#2E7D32' if s_iou > 0.5 else '#C62828')

for j, title in enumerate(col_titles):
    axes[0, j].set_title(f"{title}\n" + axes[0, j].get_title(), fontsize=12, fontweight='bold', color='#333333')

fig.suptitle("Lane Detection — Ground Truth vs Model Prediction\n(Green=GT, Blue=Predicted, Overlay=Combined)",
             fontsize=18, fontweight='bold', color='#333333', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(LANE_OUTPUT_DIR, "sample_predictions.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  Saved: sample_predictions.png ({len(sample_idx)} samples)")


# #####################################################
#
#   B-STEP 7 :  LANE SCENARIOS
#
# #####################################################

print("\n" + "=" * 65)
print("  B-STEP 7 :  LANE DETECTION SCENARIOS")
print("=" * 65)


def count_lanes(mask, threshold=100):
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_uint8)
    return max(0, num_labels - 1)


lane_counts = np.array([count_lanes(Y_lane_test[i, :, :, 0]) for i in range(len(Y_lane_test))])
two_lane   = np.where(lane_counts == 2)[0]
multi_lane = np.where(lane_counts >= 3)[0]
one_lane   = np.where(lane_counts == 1)[0]

print(f"  1-lane scenes  : {len(one_lane)}")
print(f"  2-lane scenes  : {len(two_lane)}")
print(f"  3+ lane scenes : {len(multi_lane)}")


def gen_lane_scenario(indices, title, color, filename, n=6):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.patch.set_facecolor('#1a1a2e')
    np.random.seed(RANDOM_STATE)
    chosen = np.random.choice(indices, min(n, len(indices)), replace=False) if len(indices) > n else indices[:n]
    for ii, ax in enumerate(axes.flat):
        if ii >= len(chosen): ax.axis('off'); continue
        i = chosen[ii]
        img = X_lane_test[i].astype(np.float32) / 255.0
        pred = Y_lane_pred_bin[i, :, :, 0]
        overlay = img.copy()
        lane_ov = np.zeros_like(img)
        lane_ov[:, :, 1] = pred * 0.8
        lane_ov[:, :, 2] = pred * 0.3
        overlay = np.clip(overlay + lane_ov * 0.6, 0, 1)
        ax.imshow(overlay)
        ax.set_title(f"Detected {count_lanes(pred)} lane(s)", fontsize=12, fontweight='bold', color='white', pad=8)
        ax.axis('off')
    fig.suptitle(title, fontsize=20, fontweight='bold', color=color, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(LANE_OUTPUT_DIR, filename), dpi=DPI, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {filename}")


if len(two_lane) > 0:   gen_lane_scenario(two_lane,   "SCENARIO 1 : Standard 2-Lane Road",    '#4CAF50', "scenario_2lane.png")
if len(multi_lane) > 0:  gen_lane_scenario(multi_lane, "SCENARIO 2 : Multi-Lane Highway",      '#2196F3', "scenario_multilane.png")
if len(one_lane) > 0:    gen_lane_scenario(one_lane,   "SCENARIO 3 : Challenging Single Lane", '#FF9800', "scenario_singlelane.png")

# Calculate IoU for all test samples to group by performance
best_iou_list = []
for i in range(len(Y_lane_test)):
    gt = Y_lane_test[i, :, :, 0]; pred = Y_lane_pred_bin[i, :, :, 0]
    inter = np.sum(gt * pred); uni = np.sum(gt) + np.sum(pred) - inter
    best_iou_list.append((i, inter/(uni+1e-7)))
best_iou_list.sort(key=lambda x: x[1], reverse=True)

# SCENARIO 4: Curved Roads & Edge Cases (Failure Analysis)
# We choose the worst performing predictions to demonstrate how the model reacts to tricky edge cases
worst_idx = [x[0] for x in best_iou_list[-6:]]
gen_lane_scenario(worst_idx, "SCENARIO 4 : Curved Roads & Edge Cases (Failure Analysis)", '#F44336', "scenario_failures.png")

# Best predictions
best_idx = [x[0] for x in best_iou_list[:6]]
gen_lane_scenario(best_idx, "BEST PREDICTIONS — Highest IoU Samples", '#E91E63', "scenario_best_predictions.png")

print("\n" + "=" * 65)
print("  ✓  PART B COMPLETE — Lane Detection Done!")
print("=" * 65)


# *************************************************************
# *************************************************************
#
#   ALL DONE — FINAL SUMMARY
#
# *************************************************************
# *************************************************************

total_time = time.time() - steer_start

print("\n\n" + "=" * 65)
print("=" * 65)
print("  ALL DONE! BOTH MODULES COMPLETE")
print("=" * 65)
print("=" * 65)

print(f"\n  Total execution time: {total_time/60:.1f} minutes")

print("\n  ╔══════════════════════════════════════════════════════╗")
print("  ║  PART A: STEERING PREDICTION                        ║")
print(f"  ║  Output: {STEERING_OUTPUT_DIR:<43s} ║")
print("  ╠══════════════════════════════════════════════════════╣")
print("  ║  MODEL:                                             ║")
print("  ║    steering_model.h5                                ║")
print("  ║    regression_report.txt                            ║")
print("  ║  CHARTS:                                            ║")
print("  ║    training_testing_loss.png                        ║")
print("  ║    training_testing_mae.png                         ║")
print("  ║    combined_loss_mae.png                            ║")
print("  ║    actual_vs_predicted.png                          ║")
print("  ║    steering_distribution.png                        ║")
print("  ║    error_distribution.png                           ║")
print("  ║    residual_plot.png                                ║")
print("  ║    all_metrics_combined.png                         ║")
print("  ║    metrics_radar_chart.png                          ║")
print("  ║    epoch_table.png                                  ║")
print("  ║    sample_predictions.png                           ║")
print("  ║    scenario_straight.png                            ║")
print("  ║    scenario_left_turn.png                           ║")
print("  ║    scenario_right_turn.png                          ║")
print("  ╠══════════════════════════════════════════════════════╣")
print("  ║  PART B: LANE DETECTION                             ║")
print(f"  ║  Output: {LANE_OUTPUT_DIR:<43s} ║")
print("  ╠══════════════════════════════════════════════════════╣")
print("  ║  MODEL:                                             ║")
print("  ║    lane_model.h5                                    ║")
print("  ║    segmentation_report.txt                          ║")
print("  ║  CHARTS:                                            ║")
print("  ║    training_testing_loss.png                        ║")
print("  ║    training_testing_accuracy.png                    ║")
print("  ║    iou_per_epoch.png                                ║")
print("  ║    dice_per_epoch.png                               ║")
print("  ║    combined_training_progress.png                   ║")
print("  ║    all_metrics_combined.png                         ║")
print("  ║    metrics_radar_chart.png                          ║")
print("  ║    epoch_table.png                                  ║")
print("  ║    sample_predictions.png                           ║")
print("  ║    scenario_2lane.png                               ║")
print("  ║    scenario_multilane.png                           ║")
print("  ║    scenario_singlelane.png                          ║")
print("  ║    scenario_best_predictions.png                    ║")
print("  ╚══════════════════════════════════════════════════════╝")
print()
print("=" * 65 + "\n")
