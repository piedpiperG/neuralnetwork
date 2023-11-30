from eralchemy import render_er
from IPython.display import Image

# Define the schema for the ER diagram
schema = """
[用户] {bgcolor: "#D5E8D4"}
*用户名
*密码

[角色] {bgcolor: "#D5E8D4"}
*角色名

[课程] {bgcolor: "#D5E8D4"}

[知识点] {bgcolor: "#D5E8D4"}

[试题] {bgcolor: "#D5E8D4"}
*内容
*分值
备选答案
正确答案

[试卷] {bgcolor: "#D5E8D4"}
*试卷标题

[考试] {bgcolor: "#D5E8D4"}
*考试标题
*考试时间
任教老师

用户 |o--o| 角色
教师 |o--o{ 试题
试题 ||--o{ 知识点
试题 }|--|{ 试卷
试卷 ||--o{ 考试
考试 }|--|{ 学生
学生 |o--o{ 成绩
"""

# Render the ER diagram
er_diagram_path = "/mnt/data/online_exam_system_er_diagram.png"
render_er(schema, er_diagram_path)

# Display the ER diagram
Image(er_diagram_path)
