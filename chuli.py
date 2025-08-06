# 定义源文件路径
file_path = "E:\\大创\\大创数据处理\\xunlian\\test.txt"

# 定义输出文件路径
positive_file = "E:\\大创\\大创数据处理\\xunlian\\positive2.txt"
negative_file = "E:\\大创\\大创数据处理\\xunlian\\negative2.txt"

try:
    # 读取源文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    positive_lines = []
    negative_lines = []

    # 处理每一行
    for line in lines:
        line = line.strip()  # 去除换行符和空格
        if not line:
            continue  # 跳过空行
        parts = line.rsplit(maxsplit=1)  # 从右边分割一次
        if len(parts) != 2:
            continue  # 跳过格式不正确的行
        text, label = parts
        if label == '1':
            positive_lines.append(text + '\n')
        elif label == '0':
            negative_lines.append(text + '\n')

    # 写入positive文件
    with open(positive_file, 'w', encoding='utf-8') as pf:
        pf.writelines(positive_lines)

    # 写入negative文件
    with open(negative_file, 'w', encoding='utf-8') as nf:
        nf.writelines(negative_lines)

    print("处理完成！")
except Exception as e:
    print(f"错误：{e}")
