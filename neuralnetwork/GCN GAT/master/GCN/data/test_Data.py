def load_labels(file_path):
    label_map = {}  # 类别映射
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label_name = line.strip().split()[1]
            if label_name not in label_map:
                label_map[label_name] = len(label_map)
            labels.append(label_map[label_name])
    return labels, label_map

# 示例使用
file_path = 'r52-label.txt'
labels, label_map = load_labels(file_path)

# 输出类别映射
print("Class Map:", label_map)

# 输出提取的标签
print("Labels:", labels)
