number_list = [10, 20, 50, 10, 25, 30, 40]


def print_numbers(nums):
    sum = 0
    cnt = 0
    for i in nums:
        sum += i
        cnt += 1
        if sum + i > 100:
            print(number_list[:cnt + 1])
            break


print_numbers(number_list)
