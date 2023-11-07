# 从文本文件中读取数据
data_list = ["1200", "1400", "1600", "1800", "2000", "2200", "2400"]
for data_name in data_list:
    file_name = data_name
    with open(f'data/{file_name}.txt', 'r') as file:
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
            print(combination)
            print(count)
            print(averages[combination])

    # 将最后三行数据写入文本文件
    with open(f'output/{file_name}.txt', 'w') as output_file:
        for combination, count in count_dict.items():
            if count > 1:
                avg_frame_count = count / cnt
                output_file.write(f'{combination} {count} {avg_frame_count}\n')

    # 打开要操作的文件
    file_path = f'output/{file_name}.txt'

    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 删除第一行
    if len(lines) > 0:
        del lines[0]

import matplotlib.pyplot as plt
import pandas as pd

# 从文本文件中读取数据
data_list = ["1200", "1400", "1600", "1800", "2000", "2200", "2400"]

for data_name in data_list:
    with open(f'data/{data_name}.txt', 'r') as file:
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

    # 构建pandas数据框
    combinations = []
    counts = []
    averages = []
    for combination, count in count_dict.items():
        avg_frame_count = count / cnt
        combinations.append(combination)
        counts.append(count)
        averages.append(avg_frame_count)

    df = pd.DataFrame({
        'Combination': combinations,
        'Count': counts,
        'Average': averages
    })
    df = df.sort_values(by='Count', ascending=False).head(15)

    # 绘制表格
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    plt.title(f"Top 15 Combinations for {data_name}")
    plt.show()

    # 绘制柱状图
    df.plot(kind='bar', x='Combination', y='Count', title=f"Top 15 Combinations for {data_name}", figsize=(10, 6))
    plt.ylabel('Count')
    plt.xlabel('Combination')
    plt.show()
