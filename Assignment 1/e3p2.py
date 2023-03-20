import random

def main(): # exercise 3, part 2, a
    n_simulations = 1000
    sum_people_all_simulations = 0
    for _ in range(1000):
        n_people = 0
        birthdays_left = 365
        birthdays_occupied = [False] * 365
        while birthdays_left > 0:
            n_people += 1
            birthday = random.randrange(365)
            if not birthdays_occupied[birthday]:
                birthdays_occupied[birthday] = True
                birthdays_left -= 1
        sum_people_all_simulations += n_people
    average = sum_people_all_simulations/n_simulations
    print("Peter should expect on average " + str(average) + " people to join his group.")

if __name__ == "__main__":
    main()
