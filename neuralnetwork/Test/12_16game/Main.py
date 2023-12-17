import random
import matplotlib.pyplot as plt

def generate_lottery_numbers():
    """
    生成彩票号码。
    返回一个包含6个从1到21的随机不重复数字的集合。
    """
    return set(random.sample(range(1, 22), 6))

def calculate_prize(matches):
    """
    根据匹配的数字数量计算奖金。
    :param matches: 匹配的数字数量
    :return: 奖金金额
    """
    prize_dict = {0: 0, 1: 5, 2: 5, 3: 10, 4: 100, 5: 100, 6: 100000}
    return prize_dict.get(matches, 0)

def simulate_lottery_plays(num_plays):
    """
    模拟多次游戏，并计算总花费和总中奖金额。
    :param num_plays: 游戏次数
    :return: 总花费和总中奖金额
    """
    total_cost = num_plays * 10  # 每次游戏的花费是$10
    total_prize = 0

    for _ in range(num_plays):
        lottery_numbers = generate_lottery_numbers()
        player_numbers = generate_lottery_numbers()
        matches = len(lottery_numbers.intersection(player_numbers))
        total_prize += calculate_prize(matches)

    return total_cost, total_prize

def plot_results(num_plays_list, cost_list, prize_list):
    """
    绘制模拟结果的图表。
    :param num_plays_list: 游戏次数列表
    :param cost_list: 总花费列表
    :param prize_list: 总奖金列表
    """
    plt.figure(figsize=(10, 5))
    plt.plot(num_plays_list, cost_list, label='Total Cost', color='red')
    plt.plot(num_plays_list, prize_list, label='Total Prize', color='green')
    plt.xlabel('Number of Plays')
    plt.ylabel('Amount ($)')
    plt.title('Lottery Simulation Results')
    plt.legend()
    plt.grid(True)
    plt.show()

# 模拟不同次数的游戏
num_plays_list = [100, 1000, 10000, 100000]
cost_list = []
prize_list = []

for num_plays in num_plays_list:
    cost, prize = simulate_lottery_plays(num_plays)
    cost_list.append(cost)
    prize_list.append(prize)

# 绘制结果图表
plot_results(num_plays_list, cost_list, prize_list)
