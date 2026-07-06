import os
import pickle
import numpy as np
 
# =====================================================
# CONFIG - عدّلي القيم دي
# =====================================================
 
# مسار مجلد بيانات WESAD الخام (فيه S2, S3, ... إلخ)
RAW_DATA_DIR = "data"
 
# أسماء الـ subjects (WESAD رسميًا بيستبعد S1 و S12 لمشاكل في الأجهزة)
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
 
# طول الـ window بالعينات (700 = ثانية واحدة عند معدل 700Hz لجهاز الصدر)
WINDOW_SIZE = 700
 
# الخطوة بين كل window والتاني (لو STEP = WINDOW_SIZE يبقى مفيش overlap)
# لو عايزة overlap 50% خليها WINDOW_SIZE // 2
STEP = 700
 
# الكلاسات اللي عايزين نحتفظ بيها بس (1=Baseline, 2=Stress, 3=Amusement)
KEEP_LABELS = [1, 2, 3]
 
# مسار حفظ الملفات الناتجة
OUTPUT_DIR = "data"
 
# =====================================================
# FUNCTIONS
# =====================================================
 
def load_subject_data(subject_id, raw_data_dir):
    """
    بتحمّل ملف .pkl بتاع subject واحد وترجع chest signals + labels
    """
    path = os.path.join(raw_data_dir, f"S{subject_id}", f"S{subject_id}.pkl")
 
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
 
    chest = data["signal"]["chest"]
    labels = data["label"]
 
    # ترتيب القنوات: ACC_x, ACC_y, ACC_z, ECG, EMG, EDA, Temp, Resp
    acc = chest["ACC"]          # shape (N, 3)
    ecg = chest["ECG"]          # shape (N, 1)
    emg = chest["EMG"]          # shape (N, 1)
    eda = chest["EDA"]          # shape (N, 1)
    temp = chest["Temp"]        # shape (N, 1)
    resp = chest["Resp"]        # shape (N, 1)
 
    signals = np.hstack([acc, ecg, emg, eda, temp, resp])  # (N, 8)
 
    return signals, labels
 
 
def segment_windows(signals, labels, subject_id, window_size, step, keep_labels):
    
    X_list = []
    y_list = []
    subj_list = []
 
    n_samples = signals.shape[0]
 
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
 
        window_signal = signals[start:end]
        window_labels = labels[start:end]
 
        unique_labels = np.unique(window_labels)
 
        # نتجاهل الـ window لو فيها أكتر من label واحد (يعني فترة انتقال)
        if len(unique_labels) != 1:
            continue
 
        label = unique_labels[0]
 
        if label not in keep_labels:
            continue
 
        X_list.append(window_signal)
        y_list.append(label)
        subj_list.append(f"S{subject_id}")
 
    return X_list, y_list, subj_list
 
 
 
def main():
    all_X = []
    all_y = []
    all_subjects = []
 
    for subject_id in SUBJECT_IDS:
        print(f"Processing S{subject_id} ...")
 
        signals, labels = load_subject_data(subject_id, RAW_DATA_DIR)
 
        X_list, y_list, subj_list = segment_windows(
            signals, labels, subject_id,
            WINDOW_SIZE, STEP, KEEP_LABELS
        )
 
        print(f"  -> {len(X_list)} windows")
 
        all_X.extend(X_list)
        all_y.extend(y_list)
        all_subjects.extend(subj_list)
 
    X = np.array(all_X)              # (total_windows, WINDOW_SIZE, 8)
    y = np.array(all_y)              # (total_windows,)
    subjects = np.array(all_subjects)  # (total_windows,)  <-- ده الجديد المهم
 
    print("\nFinal shapes:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("subjects:", subjects.shape)
    print("\nWindows per subject:")
    for s in SUBJECT_IDS:
        print(f"  S{s}: {(subjects == f'S{s}').sum()}")
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "subjects.npy"), subjects)  # <-- مهم جدًا
 
    print(" OUTPUT_DIR")
    print(" X.npy, y.npy, subjects.npy")
 
 
if __name__ == "__main__":
    main()
 