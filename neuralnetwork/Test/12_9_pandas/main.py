import pandas as pd


# 读取文件
def readfile():
    return pd.read_csv('fastFood_NJ.csv')


def Canteen_chose(file, name='null', category='null'):
    # 检查是否要筛选连锁店
    if name != 'null':
        # 筛选出包含特定名称的店
        fast_food_file = file[
            file['categories'].str.contains(category, case=False) & file['name'].str.contains(name,
                                                                                              case=False)].copy()
    else:
        # 筛选出所有店
        fast_food_file = file[file['categories'].str.contains(category, case=False)].copy()
    return fast_food_file


def City_Maxfast(file, name='null'):
    fast_food_file = Canteen_chose(file, name, 'Fast Food')
    # 提取城市信息
    fast_food_file['city'] = fast_food_file['address'].apply(lambda x: x.split(',')[1])
    # 按城市分组并计数
    city_counts = fast_food_file['city'].value_counts()
    # print(city_counts)
    # 找出快餐店数量最多的城市
    most_fast_food_city = city_counts.idxmax()
    return most_fast_food_city


def average_rating_for_chains(file, chains):
    ratings = {}
    for chain in chains:
        # 使用 Canteen_chose 函数筛选特定连锁店的数据
        chain_data = Canteen_chose(file, name=chain, category='Fast Food')
        # 计算并存储平均评分
        average_rating = chain_data['rating'].mean()
        ratings[chain] = average_rating
    return ratings


def Rating_Canteen(file, category):
    # 筛选出“咖啡与茶”类别的餐厅
    coffee_tea_restaurants = Canteen_chose(file, category=category)
    # 找出评分最高的餐厅
    highest_rated_restaurant = coffee_tea_restaurants.loc[coffee_tea_restaurants['rating'].idxmax()]
    return highest_rated_restaurant


def analyze_rating_review_correlation(file):
    # 计算评分和评论数量之间的相关系数
    correlation = file['rating'].corr(file['reviews'])
    return correlation


if __name__ == '__main__':
    # 读取文件
    df = readfile()

    print('***********************************问题1***********************************')
    # 问题1：哪个城市有最多的快餐店？
    most_fast_food_city = City_Maxfast(df)
    print(f"快餐店数量最多的城市是: {most_fast_food_city}")

    print('***********************************问题2***********************************')
    # 问题2：哪个城市拥有最多特定快餐连锁店？
    Mc_city = City_Maxfast(df, 'McDonald\'s')
    print(f'McDonald最多的城市是: {Mc_city}')
    BurgerKing_city = City_Maxfast(df, 'Burger King')
    print(f'Burger King最多的城市是: {BurgerKing_city}')
    Subway_city = City_Maxfast(df, 'Subway')
    print(f'Subway最多的城市是: {Subway_city}')
    We_city = City_Maxfast(df, 'Wendy\'s')
    print(f'Wendy\'s最多的城市是: {We_city}')
    KFC_city = City_Maxfast(df, 'KFC')
    print(f'KFC最多的城市是: {KFC_city}')
    PizzaHut_city = City_Maxfast(df, 'Pizza Hut')
    print(f'Pizza Hut最多的城市是: {PizzaHut_city}')

    print('***********************************问题3***********************************')
    # 问题3：比较特定快餐连锁店的评分
    chains = ['McDonald\'s', 'Burger King', 'Subway', 'Wendy\'s', 'KFC', 'Pizza Hut']
    # 获取每个连锁店的平均评分
    chain_ratings = average_rating_for_chains(df, chains)
    # 找出评分最高和最低的连锁店
    highest_rated_chain = max(chain_ratings, key=chain_ratings.get)
    lowest_rated_chain = min(chain_ratings, key=chain_ratings.get)
    print(f"评分最高的连锁店是: {highest_rated_chain}，平均评分为: {chain_ratings[highest_rated_chain]}")
    print(f"评分最低的连锁店是: {lowest_rated_chain}，平均评分为: {chain_ratings[lowest_rated_chain]}")

    print('***********************************问题4***********************************')
    highest_rated_restaurant = Rating_Canteen(df, 'Sandwiches')
    print(
        f"评分最高的‘ Sandwiches’类别餐厅是：{highest_rated_restaurant['name']}，地址：{highest_rated_restaurant['address']}，评分：{highest_rated_restaurant['rating']}")

    print('***********************************额外问题***********************************')
    # 额外问题：评分是否与评论数量相关？
    correlation = analyze_rating_review_correlation(df)
    print(f"评分与评论数量的相关系数为: {correlation}")

    # 根据相关系数判断关系
    if correlation > 0.5:
        print("评分与评论数量有较强的正相关关系。")
    elif correlation < -0.5:
        print("评分与评论数量有较强的负相关关系。")
    else:
        print("评分与评论数量的相关性不明显。")
