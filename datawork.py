import librosa
import librosa.display
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import Sequential  # 用于创建顺序模型
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation  # 用于构建层
from joblib import dump
# 读取音频数据
data, sampling_rate = librosa.load(r'E:\大创\大创数据处理\Data\03-01-01-01-01-01-01.wav')
# 绘制音频波形
plt.figure(figsize=(15, 5))
librosa.display.waveshow(data, sr=sampling_rate, color="blue")
plt.show()
# 获取数据列表
file_path = r'E:\大创\大创数据处理\Data'
file_list = os.listdir(file_path)


# 构建情感标签
def get_emotion_label(file_name):
    gender = 'female' if int(file_name[18:-4]) % 2 == 0 else 'male'
    emotion_code = file_name[6:-16]

    emotion_map = {
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful'
    }

    if emotion_code in emotion_map:
        return f'{gender}_{emotion_map[emotion_code]}'
    else:
        return 'unknown'


labels = [get_emotion_label(file) for file in file_list]
labels_df = pd.DataFrame(labels, columns=['label'])

# 提取 MFCC 特征
df = pd.DataFrame(columns=['feature'])
for i, file in enumerate(file_list):
    file_path_full = os.path.join(file_path, file)
    X, sr = librosa.load(file_path_full, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=25)
    feature = np.mean(mfccs, axis=1)
    df.loc[i] = [feature]

# 转换特征为 DataFrame
df_features = pd.DataFrame(df['feature'].values.tolist())
final_df = pd.concat([df_features, labels_df], axis=1)
final_df = final_df.dropna()

# 处理标签为 one-hot 编码
labels = final_df['label'].astype(str)
labels_onehot = pd.get_dummies(labels, dtype=int).values
num_classes = labels_onehot.shape[1]
label_mapping = {i: label for i, label in enumerate(pd.get_dummies(labels).columns)}

# 处理特征
df_features = final_df.drop(columns=['label'])
features = df_features.values
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(scaled_features, labels_onehot, test_size=0.2, random_state=42,
                                                    stratify=labels_onehot)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 调整 CNN 输入数据格式
X_train = X_train.reshape(-1, 25, 1)
X_val = X_val.reshape(-1, 25, 1)
X_test = X_test.reshape(-1, 25, 1)

# 超参数调节
LEARNING_RATE = 0.002  # 学习率
BATCH_SIZE = 32  # 批量大小
EPOCHS = 30  # 训练轮数
DROPOUT_RATE = 0.1  # Dropout 率

# 扩充维度
x_traincnn = np.expand_dims(X_train, axis=2)  # 形状: (num_samples, 13, 1)
x_testcnn = np.expand_dims(X_test, axis=2)    # 形状: (num_samples, 13, 1)

# 构建CNN序贯模型
model = Sequential()

# 卷积层+激活层
model.add(Conv1D(256, 5, padding='same', input_shape=(25, 1)))  # 修改 input_shape
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))

# Dropout防止过拟合
model.add(Dropout(0.1))

# 池化层降维
model.add(MaxPooling1D(pool_size=(2)))

# 卷积层+激活层
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))

# 展平+全连接层
model.add(Flatten())

# 输出层（假设类别数为 num_classes）
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val),
                    callbacks=[early_stop], verbose=1)

# 保存训练好的模型
model.save("trained_model.h5")

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

# 获取预测结果
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# 转换分类报告的索引格式
y_test_labels_named = [label_mapping[label] for label in y_test_labels]
y_pred_labels_named = [label_mapping[label] for label in y_pred_labels]

# 输出分类报告和混淆矩阵
print(classification_report(y_test_labels_named, y_pred_labels_named))
print(confusion_matrix(y_test_labels_named, y_pred_labels_named))

dump(scaler, 'standard_scaler.pkl')
