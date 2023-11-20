tar = 1200
for i in range(1200, 2401, 200):
    # 路径到源文件和目标文件
    source_file_path = f'1113_data/Cu.MSD.{i}.dat'
    target_file_path = source_file_path

    # 读取源文件
    with open(source_file_path, 'r') as file:
        lines = file.readlines()

    # 分离注释行和数据行
    comment_lines = [line for line in lines if line.startswith("#")]
    data_lines = [line for line in lines if not line.startswith("#")]

    # 处理数据行
    split_data_lines = [line.split() for line in data_lines]
    first_value = int(split_data_lines[0][0])
    modified_lines = [[int(line[0]) - first_value, line[1]] for line in split_data_lines]

    # 将处理后的数据转换回字符串格式，并包含注释行
    modified_data = "\n".join(comment_lines + [" ".join(map(str, line)) for line in modified_lines])

    # 写入到目标文件
    with open(target_file_path, 'w') as file:
        file.write(modified_data)
