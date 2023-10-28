import matplotlib.pyplot as plt

# 从文本文件中读取数据
with open('1200℃.txt', 'r') as file:
    lines = file.readlines()

data = []
count_dict = {}

cnt = 0

# 解析数据并提取有效数字组合
for line in lines:
    try:
        cnt += 1
        parts = line.strip().split()
        data_row = list(map(int, parts[1:]))
        data.append(data_row)
        combination = tuple(data_row[1:])
        if combination in count_dict:
            count_dict[combination] += 1
        else:
            count_dict[combination] = 1
    except ValueError:
        print(f'line{cnt}:{line}')

# 计算每个组合的平均值
averages = {}
for combination, count in count_dict.items():
    if count > 1:
        avg_frame_count = count / cnt
        averages[combination] = avg_frame_count
        # print(combination)
        # print(count)
        # print(averages[combination])

# 将最后三行数据写入文本文件
with open('output.txt', 'w') as output_file:
    for combination, count in count_dict.items():
        if count > 1:
            avg_frame_count = count / cnt
            output_file.write(f'{combination} {count} {avg_frame_count}\n')

# # 提取combination和averages[combination]为两个列表以便绘图
# combinations = list(averages.keys())
# avg_values = list(averages.values())
#
# # 创建图形
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(combinations)), avg_values, tick_label=[str(c) for c in combinations])
# plt.xlabel('Combination')
# plt.ylabel('Averages')
# plt.title('Combination vs. Averages')
#
# # 旋转x轴标签以便更好地显示
# plt.xticks(rotation=90)
#
# plt.show()
