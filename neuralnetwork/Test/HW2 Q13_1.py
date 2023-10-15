def main():
    def get_score():
        x = int(input("Enter the score: "))
        return x

    x = get_score()

    def give_grade(x):
        if 100 >= x >= 90:
            return "A"
        elif 90 > x >= 85:
            return "B+"
        elif 85 > x >= 80:
            return "B"
        elif 80 > x >= 75:
            return "C+"
        elif 75 > x >= 70:
            return "C"
        elif 70 > x >= 60:
            return "D"
        elif 60 > x >= 0:
            return "F"

    grade = give_grade(x)
    print(f"Score: {x}, Letter grade: {grade}")


main()

