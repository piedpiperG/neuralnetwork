from method import *

alphabet_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
alphabet_list_re = [chr(i) for i in range(ord('a'), ord('z') + 1)]
res = []

# 进行正向的训练
# train()
# for alpha in alphabet_list:
#     output_name, top5_each_step = predict('male', alpha)
#     res.append(output_name)
#     print(output_name)
# print(res)
# plot_predictions(top5_each_step)

# 进行反向的训练
# train_reverse()
# for alpha in alphabet_list:
#     output_name_reverse, top5_each_step_reverse = predict_reverse_re('female', alpha)
#     res.append(output_name_reverse)
#     print(output_name_reverse)
# print(res)
# plot_predictions(top5_each_step_reverse)

# 生成从中间开始的名字
# print(predict_full_name('male', 'ak'))

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def generate_name():
    input_text = entry.get()
    direction = direction_var.get()
    category = category_var.get().lower()

    # 根据方向和类别调用相应的模型预测函数
    if direction == 'Forward':
        output_name, top5_each_step = predict(category, input_text)
    elif direction == 'Backward':
        output_name, top5_each_step = predict_reverse(category, input_text)
    else:
        output_name, top5_each_step = predict_full_name(category, input_text)

    # 显示结果
    result_label.config(text=f"Generated Name: {output_name}")

    # 调用绘图函数
    plot_predictions(top5_each_step)


def plot_predictions(top5_each_step):
    # 创建新窗口用于绘制图表
    new_window = tk.Toplevel(root)
    new_window.title("Prediction Visualization")

    # 创建一个matplotlib图表
    fig, ax = plt.subplots(figsize=(14, 8))  # 调整图表大小

    # 设置颜色列表，以便区分不同的步骤
    colors = plt.cm.viridis(np.linspace(0, 1, len(top5_each_step[0][0][0])))

    # 绘制条形图的宽度
    bar_width = 0.1

    # 获取步骤数量
    num_steps = len(top5_each_step)

    # 对于每个步骤中的每个字母，绘制一个条形图
    for step, (topi, topv) in enumerate(top5_each_step):
        # 如果是Tensor，则转换为numpy数组
        if hasattr(topv, 'cpu'):
            topv = topv.cpu().numpy()
        if hasattr(topi, 'cpu'):
            topi = topi.cpu().numpy()

        # 确保字符索引列表中的每个索引都在all_letters的有效范围内
        characters = ['EOF' if i == 58 else all_letters[i] for i in topi[0] if i < len(all_letters)]

        # 如果概率是对数形式，转换回标准形式
        probabilities = np.exp(topv[0])

        # 绘制当前步骤的条形图，确保条形图的数量与字符列表长度相匹配
        for idx, (char, prob) in enumerate(zip(characters, probabilities)):
            # 计算条形图的x坐标
            x = step + idx * bar_width
            ax.bar(x, prob, bar_width, label=f'{char} (Step {step + 1})')

    # 设置x轴标签
    ax.set_xticks([r + bar_width for r in range(num_steps)])
    ax.set_xticklabels([f'Step {r + 1}' for r in range(num_steps)])
    ax.set_xlabel('Steps')
    ax.set_ylabel('Probability')
    ax.set_title('Character Probability at Each Step')
    ax.set_ylim(0, 1)

    # 添加图例
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

    # 调整布局以适应图例
    plt.tight_layout()

    # 将matplotlib图表嵌入到Tkinter窗口中
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


# 创建主窗口
root = tk.Tk()
root.title("Name Generation Visualizer")

# 创建输入框
entry_label = tk.Label(root, text="Enter Starting Letters:")
entry_label.pack()
entry = tk.Entry(root)
entry.pack()

# 创建类别下拉菜单
category_var = tk.StringVar(value='Male')
category_label = tk.Label(root, text="Choose Category:")
category_label.pack()
category_menu = ttk.Combobox(root, textvariable=category_var, values=('Male', 'Female', 'Pet'))
category_menu.pack()

# 创建方向下拉菜单
direction_var = tk.StringVar(value='Forward')
direction_label = tk.Label(root, text="Choose Direction:")
direction_label.pack()
direction_menu = ttk.Combobox(root, textvariable=direction_var, values=('Forward', 'Backward'))
direction_menu.pack()

# 创建按钮
generate_button = tk.Button(root, text="Generate", command=generate_name)
generate_button.pack()

# 创建结果标签
result_label = tk.Label(root, text="")
result_label.pack()

# 运行主循环
root.mainloop()
