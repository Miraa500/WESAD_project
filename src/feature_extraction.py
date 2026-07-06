import numpy as np
import pandas as pd

X = np.load("data/X.npy")          # (samples, 700, 8)
y = np.load("data/y.npy")          # (samples,)
subjects = np.load("data/subjects.npy")  # (samples,)

print("X:", X.shape, "| y:", y.shape, "| subjects:", subjects.shape)


ACC_X, ACC_Y, ACC_Z = 0, 1, 2
ECG, EMG, EDA, TEMP, RESP = 3, 4, 5, 6, 7



acc_mag = np.sqrt(X[:, :, ACC_X]**2 + X[:, :, ACC_Y]**2 + X[:, :, ACC_Z]**2)

features = pd.DataFrame({
    "ACC_mag_mean": acc_mag.mean(axis=1),
    "ECG_mean": X[:, :, ECG].mean(axis=1),
    "EMG_mean": X[:, :, EMG].mean(axis=1),
    "EDA_mean": X[:, :, EDA].mean(axis=1),
    "Temp_mean": X[:, :, TEMP].mean(axis=1),
    "Resp_mean": X[:, :, RESP].mean(axis=1),
    "label": y,
    "subject": subjects
})

print("\nShape of features table:", features.shape)
print(features.head())

features.to_pickle("data/processed_df.pkl")

