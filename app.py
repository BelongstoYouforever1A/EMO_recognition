import os
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from joblib import load
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Flask 初始化
app = Flask(__name__)

# ============ 音频情感识别模型加载 ============
audio_model = load_model("trained_model.h5")
scaler = load("standard_scaler.pkl")

label_mapping = {
    0: 'female_calm',
    1: 'female_happy',
    2: 'female_sad',
    3: 'female_angry',
    4: 'female_fearful',
    5: 'male_calm',
    6: 'male_happy',
    7: 'male_sad',
    8: 'male_angry',
    9: 'male_fearful',
    10: '不明情感'
}

# ============ 文本情感识别模型加载 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, output_dim, dropout=0.3):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)

    def forward(self, x):
        x = self.embedding(x)          # (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)           # (batch_size, embed_dim, seq_len)
        x = [torch.relu(conv(x)) for conv in self.convs]  # list of (batch_size, num_filters, L_out)
        x = [torch.max(pool, dim=2)[0] for pool in x]       # list of (batch_size, num_filters)
        x = torch.cat(x, dim=1)          # (batch_size, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        return self.fc(x)

# 加载词典
word_to_idx = torch.load("word_to_idx.pt")

# 创建模型并加载权重（注意参数与训练代码保持一致）
text_model = CNNClassifier(
    vocab_size=len(word_to_idx),
    embed_dim=200,
    num_filters=128,
    kernel_sizes=[2, 3, 4, 5],
    output_dim=2,
    dropout=0.3
)
text_model.load_state_dict(torch.load("lstm_model.pt", map_location=device))
text_model.to(device)
text_model.eval()

# ============ 音频处理函数 ============
def preprocess_audio(file_path, target_duration=2.5, sr=22050 * 2, n_mfcc=25):
    y, _ = librosa.load(file_path, sr=sr)
    target_length = int(target_duration * sr)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feature = np.mean(mfccs, axis=1).reshape(1, -1)
    feature_scaled = scaler.transform(feature)
    return feature_scaled.reshape(1, 25, 1)

# ============ 文本预处理函数 ============
def preprocess_text(text, max_len=50):
    words = text.strip().split()
    indices = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words]
    if len(indices) < max_len:
        indices += [word_to_idx["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor([indices], dtype=torch.long).to(device)

# ============ 路由部分 ============

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    features = preprocess_audio(file_path)

    raw_prediction = audio_model.predict(features)[0]
    penalty_weights = np.ones_like(raw_prediction)
    penalty_weights[0] *= 0.7
    penalty_weights[5] *= 0.7
    adjusted_prediction = raw_prediction * penalty_weights
    adjusted_prediction /= np.sum(adjusted_prediction)
    predicted_label = label_mapping[np.argmax(adjusted_prediction)]

    return jsonify({
        'emotion': predicted_label,
        'confidence': float(np.max(adjusted_prediction)),
        'raw_top': label_mapping[np.argmax(raw_prediction)],
    })

@app.route('/text_predict', methods=['POST'])
def predict_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': '文本为空'})
    input_tensor = preprocess_text(text)
    with torch.no_grad():
        output = text_model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        emotion = 'positive' if pred == 1 else 'negative'
    return jsonify({'emotion': emotion})

# ============ 启动 ============
if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
