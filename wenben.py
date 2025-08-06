# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from collections import Counter
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import os
#
# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # 修改后的数据加载函数，直接加载positive和negative文件
# def load_data_from_files(positive_path, negative_path):
#     # 读取positive文件
#     with open(positive_path, 'r', encoding='utf-8') as f:
#         positive_lines = f.readlines()
#
#     # 读取negative文件
#     with open(negative_path, 'r', encoding='utf-8') as f:
#         negative_lines = f.readlines()
#
#     # 创建数据和标签列表
#     data = []
#     labels = []
#
#     # 处理positive文件
#     for text in positive_lines:
#         text = text.strip()  # 去除换行符和空格
#         if not text:  # 跳过空行
#             continue
#         data.append(text.split())  # 分词
#         labels.append(1)  # 标签为1
#
#     # 处理negative文件
#     for text in negative_lines:
#         text = text.strip()  # 去除换行符和空格
#         if not text:  # 跳过空行
#             continue
#         data.append(text.split())  # 分词
#         labels.append(0)  # 标签为0
#
#     return data, labels
#
#
# # 构建词典
# def build_vocab(data, max_vocab_size=5000):
#     all_words = [word for sentence in data for word in sentence]
#     word_freq = Counter(all_words).most_common(max_vocab_size - 2)  # -2 for <PAD> and <UNK>
#     word_to_idx = {word: idx + 2 for idx, (word, _) in enumerate(word_freq)}
#     word_to_idx["<PAD>"] = 0
#     word_to_idx["<UNK>"] = 1
#     return word_to_idx
#
#
# # 文本转索引
# def text_to_index(data, word_to_idx):
#     result = []
#     for sentence in data:
#         idxs = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in sentence]
#         result.append(idxs)
#     return result
#
#
# # 补齐序列
# def pad_sequences(data, max_len=50):
#     return [seq[:max_len] + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in data]
#
#
# # LSTM 模型
# class LSTMClassifier(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
#         super(LSTMClassifier, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(0.2)
#
#     def forward(self, x):
#         x = self.embedding(x)
#         _, (h_n, _) = self.lstm(x)
#         h_n = h_n[-1]
#         out = self.fc(self.dropout(h_n))
#         return out
#
#
# # 主训练函数
# def train_model(positive_path, negative_path, test_path, model_save_path="lstm_model.pt"):
#     # 加载数据
#     train_data, train_labels = load_data_from_files(positive_path, negative_path)
#
#     # 构建词典
#     word_to_idx = build_vocab(train_data)
#     indexed_train_data = text_to_index(train_data, word_to_idx)
#     padded_train_data = pad_sequences(indexed_train_data)
#
#     # 划分训练和验证集
#     x_train, x_val, y_train, y_val = train_test_split(
#         padded_train_data, train_labels, test_size=0.2, random_state=42
#     )
#
#     # 转 tensor
#     x_train = torch.tensor(x_train, dtype=torch.long)
#     y_train = torch.tensor(y_train, dtype=torch.long)
#     x_val = torch.tensor(x_val, dtype=torch.long)
#     y_val = torch.tensor(y_val, dtype=torch.long)
#
#     # 构造 DataLoader
#     train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
#     val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64)
#
#     # 加载测试数据
#     test_data, test_labels = load_data_from_files(test_path, test_path)  # 传入同一个路径作为测试文件
#     indexed_test_data = text_to_index(test_data, word_to_idx)
#     padded_test_data = pad_sequences(indexed_test_data)
#     x_test = torch.tensor(padded_test_data, dtype=torch.long)
#     y_test = torch.tensor(test_labels, dtype=torch.long)
#     test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)
#
#     # 模型定义
#     model = LSTMClassifier(vocab_size=len(word_to_idx), embed_dim=100, hidden_dim=256, output_dim=2)
#     model.to(device)
#
#     # 损失函数和优化器
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#
#     # 训练循环
#     for epoch in range(10):
#         model.train()
#         total_loss = 0
#         for x_batch, y_batch in train_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             output = model(x_batch)
#             loss = criterion(output, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         # 验证
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for x_batch, y_batch in val_loader:
#                 x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#                 output = model(x_batch)
#                 preds = torch.argmax(output, dim=1)
#                 correct += (preds == y_batch).sum().item()
#                 total += y_batch.size(0)
#
#         print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Val Acc: {correct / total:.4f}")
#
#     # 保存模型
#     torch.save(model.state_dict(), model_save_path)
#     torch.save(word_to_idx, "word_to_idx.pt")
#     print("✅ 模型和词典保存完成！")
#
#     # 在测试集上进行评估
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for x_batch, y_batch in test_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             output = model(x_batch)
#             preds = torch.argmax(output, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y_batch.cpu().numpy())
#
#     # 生成分类报告
#     report = classification_report(all_labels, all_preds, target_names=['negative', 'positive'], digits=4)
#     print("\n分类报告：")
#     print(report)
#
#     # 保存分类报告到文件
#     with open("classification_report.txt", "w", encoding="utf-8") as f:
#         f.write(report)
#
#     # 计算混淆矩阵
#     cm = confusion_matrix(all_labels, all_preds)
#     print("\n混淆矩阵：")
#     print(cm)
#
#     # 保存混淆矩阵到文件
#     np.savetxt("confusion_matrix.txt", cm, fmt="%d")
#
#
# # 调用训练函数，使用positive.txt和negative.txt
# positive_path = "E:/大创/大创数据处理/xunlian/positive.txt"
# negative_path = "E:/大创/大创数据处理/xunlian/negative.txt"
# test_path = "E:/大创/大创数据处理/xunlian/test.txt"  # 假设你有一个测试集文件
# train_model(positive_path, negative_path, test_path, model_save_path="lstm_model.pt")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据加载函数
def load_data_from_files(positive_path, negative_path):
    with open(positive_path, 'r', encoding='utf-8') as f:
        positive_lines = f.readlines()

    with open(negative_path, 'r', encoding='utf-8') as f:
        negative_lines = f.readlines()

    data, labels = [], []
    for line in positive_lines:
        line = line.strip()
        if line:
            data.append(line.split())
            labels.append(1)
    for line in negative_lines:
        line = line.strip()
        if line:
            data.append(line.split())
            labels.append(0)
    return data, labels


# 构建词典
def build_vocab(data, max_vocab_size=5000):
    all_words = [word for sentence in data for word in sentence]
    word_freq = Counter(all_words).most_common(max_vocab_size - 2)
    word_to_idx = {word: idx + 2 for idx, (word, _) in enumerate(word_freq)}
    word_to_idx["<PAD>"] = 0
    word_to_idx["<UNK>"] = 1
    return word_to_idx


# 文本转索引
def text_to_index(data, word_to_idx):
    result = []
    for sentence in data:
        idxs = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in sentence]
        result.append(idxs)
    return result


# 补齐序列
def pad_sequences(data, max_len=50):
    return [seq[:max_len] + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in data]


# CNN 模型
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, output_dim, dropout=0.5):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        x = [torch.relu(conv(x)) for conv in self.convs]  # [(batch_size, num_filters, L_out), ...]
        x = [torch.max(pool, dim=2)[0] for pool in x]     # [(batch_size, num_filters), ...]
        x = torch.cat(x, dim=1)  # (batch_size, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        return self.fc(x)


# 主训练函数
def train_model(positive_path, negative_path, test_positive_path, test_negative_path, model_save_path="lstm_model.pt"):
    # 加载训练数据
    train_data, train_labels = load_data_from_files(positive_path, negative_path)

    # 构建词典
    word_to_idx = build_vocab(train_data)
    indexed_train_data = text_to_index(train_data, word_to_idx)
    padded_train_data = pad_sequences(indexed_train_data)

    # 划分验证集
    x_train, x_val, y_train, y_val = train_test_split(
        padded_train_data, train_labels, test_size=0.2, random_state=42
    )

    # 转为 tensor
    x_train = torch.tensor(x_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64)

    # 加载测试集
    test_data, test_labels = load_data_from_files(test_positive_path, test_negative_path)
    indexed_test_data = text_to_index(test_data, word_to_idx)
    padded_test_data = pad_sequences(indexed_test_data)
    x_test = torch.tensor(padded_test_data, dtype=torch.long)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)

    # 构建 CNN 模型
    model = CNNClassifier(
        vocab_size=len(word_to_idx),
        embed_dim=200,
        num_filters=128,
        kernel_sizes=[2, 3, 4, 5],
        output_dim=2,
        dropout=0.3
    ).to(device)

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # 训练循环
    for epoch in range(20):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证准确率
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = torch.argmax(model(x_batch), dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Val Acc: {correct / total:.4f}")

    # 保存模型和词典
    torch.save(model.state_dict(), model_save_path)
    torch.save(word_to_idx, "word_to_idx.pt")
    print("✅ 模型和词典保存完成！")

    # 测试集评估
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = torch.argmax(model(x_batch), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # 报告与混淆矩阵
    report = classification_report(all_labels, all_preds, target_names=['negative', 'positive'], digits=4)
    print("\n分类报告：")
    print(report)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵：")
    print(cm)
    np.savetxt("confusion_matrix.txt", cm, fmt="%d")


# === 调用主函数 ===
positive_path = "E:/大创/大创数据处理/xunlian/positive.txt"
negative_path = "E:/大创/大创数据处理/xunlian/negative.txt"
test_positive_path = "E:/大创/大创数据处理/xunlian/positive_test.txt"
test_negative_path = "E:/大创/大创数据处理/xunlian/negative_test.txt"

train_model(positive_path, negative_path, test_positive_path, test_negative_path, model_save_path="lstm_model.pt")
