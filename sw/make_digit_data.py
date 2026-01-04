import os
import json
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Project settings
# -----------------------------
OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)

N_INPUT = 64      # 8x8 image flattened
HIDDEN  = 32      # hidden layer size
N_CLASS = 10      # digits 0..9
NTEST   = 50      # number of test images to export

W_SCALE = 128     # scale float weights -> int8 (fixed-point idea)

# -----------------------------
# Helper functions
# -----------------------------
def clamp_int8(x):
    """Clamp to signed int8 range and return np.int8."""
    return np.clip(x, -128, 127).astype(np.int8)

def to_hex_u8(v):
    """Convert signed byte value to 2-digit hex (two's complement)."""
    return f"{(v & 0xFF):02x}"

def to_hex_u32(v):
    """Convert signed int32 value to 8-digit hex (two's complement)."""
    return f"{(v & 0xFFFFFFFF):08x}"

def write_mem_u8(path, arr_int8):
    """Write int8 array as one 2-hex-digit value per line."""
    with open(path, "w") as f:
        for v in arr_int8.flatten():
            f.write(to_hex_u8(int(v)) + "\n")

def write_mem_u32(path, arr_int32):
    """Write int32 array as one 8-hex-digit value per line."""
    with open(path, "w") as f:
        for v in arr_int32.flatten():
            f.write(to_hex_u32(int(v)) + "\n")

def print_ascii_digit(img64, label=None):
    """Print an 8x8 digit with ASCII shading so it feels tangible."""
    grid = img64.reshape(8, 8)
    if label is not None:
        print(f"Label: {label}")
    for r in range(8):
        row = ""
        for c in range(8):
            val = grid[r, c]  # 0..16
            if val < 4:   ch = " "
            elif val < 8: ch = "."
            elif val < 12: ch = "*"
            else:         ch = "#"
            row += ch
        print(row)

def int_infer_one(x_i8, W1_q, b1_q, W2_q, b2_q, shift1):
    """
    Hardware-like integer inference:
      FC1: int32 accumulate -> shift -> clamp -> ReLU -> int8 activations
      FC2: int32 logits -> argmax
    Returns predicted digit.
    """
    x32 = x_i8.astype(np.int32)

    # FC1 (32 outputs)
    fc1 = b1_q.astype(np.int32) + (W1_q.astype(np.int32) @ x32)  # shape (32,)
    a1 = (fc1 >> shift1)  # scale down
    a1 = np.clip(a1, -128, 127).astype(np.int32)
    a1[a1 < 0] = 0        # ReLU
    a1_i8 = a1.astype(np.int8)

    # FC2 (10 outputs)
    logits = b2_q.astype(np.int32) + (W2_q.astype(np.int32) @ a1_i8.astype(np.int32))  # (10,)
    pred = int(np.argmax(logits))
    return pred

# -----------------------------
# Main script
# -----------------------------
def main():
    # 1) Load dataset (8x8 digits)
    digits = load_digits()
    X = digits.data   # (n_samples, 64), each value 0..16
    y = digits.target # (n_samples,), labels 0..9

    print("Example digit from dataset (ASCII):")
    print_ascii_digit(X[0], label=y[0])
    print()

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # 3) Train a small float neural net: 64 -> 32 -> 10
    clf = MLPClassifier(
        hidden_layer_sizes=(HIDDEN,),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=0
    )
    clf.fit(X_train, y_train)

    # Float accuracy (just to sanity check training)
    y_float_pred = clf.predict(X_test)
    float_acc = accuracy_score(y_test, y_float_pred)
    print(f"Float model accuracy: {float_acc*100:.2f}%")

    # 4) Convert inputs to int8 for hardware
    # Pixel values are 0..16. We center by subtracting 8 so inputs are ~[-8..+8]
    X_train_i8 = clamp_int8(np.rint(X_train - 8.0))
    X_test_i8  = clamp_int8(np.rint(X_test  - 8.0))

    # 5) Extract float weights and quantize to integers
    # sklearn weight shapes:
    #   coefs_[0] is (64, 32)  input->hidden
    #   coefs_[1] is (32, 10)  hidden->output
    #
    # For hardware, we prefer:
    #   W1: (32, 64)  so W1[i,j] = weight for hidden neuron i, input j
    #   W2: (10, 32)
    W1_f = clf.coefs_[0].T            # (32, 64)
    b1_f = clf.intercepts_[0]         # (32,)
    W2_f = clf.coefs_[1].T            # (10, 32)
    b2_f = clf.intercepts_[1]         # (10,)

    W1_q = clamp_int8(np.rint(W1_f * W_SCALE))
    W2_q = clamp_int8(np.rint(W2_f * W_SCALE))
    b1_q = np.rint(b1_f * W_SCALE).astype(np.int32)
    b2_q = np.rint(b2_f * W_SCALE).astype(np.int32)

    # 6) Choose SHIFT1 so FC1 outputs don't overflow int8 too often
    # We'll look at max absolute FC1 sum on a sample and choose a shift
    sample_count = min(200, X_train_i8.shape[0])
    max_abs_fc1 = 0
    for i in range(sample_count):
        x32 = X_train_i8[i].astype(np.int32)
        fc1 = b1_q + (W1_q.astype(np.int32) @ x32)
        max_abs_fc1 = max(max_abs_fc1, int(np.max(np.abs(fc1))))

    shift1 = 0
    while (max_abs_fc1 >> shift1) > 120 and shift1 < 16:
        shift1 += 1

    print(f"Auto-chosen SHIFT1 = {shift1} (max abs FC1 sum before shift was {max_abs_fc1})")

    # 7) Integer (hardware-like) accuracy (what RTL should match)
    y_int_pred = []
    for i in range(X_test_i8.shape[0]):
        y_int_pred.append(int_infer_one(X_test_i8[i], W1_q, b1_q, W2_q, b2_q, shift1))
    y_int_pred = np.array(y_int_pred, dtype=np.int32)
    int_acc = float(np.mean(y_int_pred == y_test))
    print(f"Integer (hardware-like) accuracy: {int_acc*100:.2f}%")

    # 8) Export model parameters as .mem files (hex per line)
    # Row-major layout:
    #   W1[i,j] stored at addr = i*64 + j  (total 32*64 = 2048 bytes)
    #   W2[k,t] stored at addr = k*32 + t  (total 10*32 = 320 bytes)
    write_mem_u8 (os.path.join(OUTDIR, "w1.mem"), W1_q.reshape(-1))
    write_mem_u32(os.path.join(OUTDIR, "b1.mem"), b1_q.reshape(-1))
    write_mem_u8 (os.path.join(OUTDIR, "w2.mem"), W2_q.reshape(-1))
    write_mem_u32(os.path.join(OUTDIR, "b2.mem"), b2_q.reshape(-1))

    # Export test images and references
    X_small = X_test_i8[:NTEST]                  # (NTEST,64)
    y_label = y_test[:NTEST].astype(np.int32)    # true labels (for accuracy reporting)

    y_ref = []
    for i in range(NTEST):
        y_ref.append(int_infer_one(X_small[i], W1_q, b1_q, W2_q, b2_q, shift1))
    y_ref = np.array(y_ref, dtype=np.int32)      # "golden" outputs for RTL

    write_mem_u8(os.path.join(OUTDIR, "x_test.mem"), X_small.reshape(-1))

    with open(os.path.join(OUTDIR, "y_label.txt"), "w") as f:
        for v in y_label:
            f.write(str(int(v)) + "\n")

    with open(os.path.join(OUTDIR, "y_ref.txt"), "w") as f:
        for v in y_ref:
            f.write(str(int(v)) + "\n")

    meta = {
        "N_INPUT": N_INPUT,
        "HIDDEN": HIDDEN,
        "N_CLASS": N_CLASS,
        "NTEST": NTEST,
        "W_SCALE": W_SCALE,
        "SHIFT1": shift1
    }
    with open(os.path.join(OUTDIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nExported to ./data/:")
    for fn in ["w1.mem","b1.mem","w2.mem","b2.mem","x_test.mem","y_label.txt","y_ref.txt","meta.json"]:
        print(" ", fn)

    print("\nWhat to use later:")
    print("  - Use y_ref.txt to verify RTL is correct (RTL vs Python-integer).")
    print("  - Use y_label.txt only to report the model's accuracy.")

if __name__ == "__main__":
    main()
