import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import matplotlib.pyplot as plt



X = np.load("data/X.npy")
y = np.load("data/y.npy")
subjects = np.load("data/subjects.npy")   # <-- ده الجديد اللي كان ناقص هنا

print("X shape:", X.shape)
print("y shape:", y.shape)
print("subjects shape:", subjects.shape)



le = LabelEncoder()
y = le.fit_transform(y)



TEST_SUBJECTS = ["S15", "S16", "S17"]   # عدّلي الأسماء حسب اللي عندك بالظبط

train_mask = ~np.isin(subjects, TEST_SUBJECTS)
test_mask = np.isin(subjects, TEST_SUBJECTS)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("\nTrain:", X_train.shape, " | Subjects:", np.unique(subjects[train_mask]))
print("Test:", X_test.shape, " | Subjects:", np.unique(subjects[test_mask]))



weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(weights))

print("Class weights:", class_weights)


model = models.Sequential()

model.add(layers.Conv1D(64, kernel_size=5, activation='relu',
                        input_shape=(700, 8)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(2))

model.add(layers.Conv1D(128, kernel_size=5, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(2))

model.add(layers.Conv1D(256, kernel_size=3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.GlobalAveragePooling1D())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.4))

model.add(layers.Dense(3, activation='softmax'))


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()



early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,              # لو 10 epochs من غير تحسن، يوقف
    restore_best_weights=True # يرجع لأفضل نسخة كانت وصلها، مش آخر واحدة
)

checkpoint = ModelCheckpoint(
    "cnn_wesad_best.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,                
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)



loss, acc = model.evaluate(X_test, y_test)

print("\nTest Accuracy (subject-independent):", acc)



plt.figure()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Model Accuracy (Subject-Based Split)")
plt.legend()
plt.show()


model.save("cnn_wesad_model.h5")
