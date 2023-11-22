# 读取原始数据文件
with open('r8-test.txt', 'r', encoding='latin1') as file:
    data = file.readlines()

# 处理数据并写入带有编号的标签文件
with open('r8-test-label.txt', 'w', encoding='utf-8') as label_file:
    for idx, line in enumerate(data, start=0):
        # 提取标签
        label = line.split('\t')[0]
        # 写入带有编号的标签文件
        label_file.write(f"{idx} {label}\n")
