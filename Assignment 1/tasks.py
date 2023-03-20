import random
import statistics

def task2():
    def reward(): # Simulates one play, returns reward
        random_number = random.randrange(1, n_atomic_events+1)
        for tuple in ranges_return_val:
            if random_number in tuple[0]:
                return tuple[1]
        return 0
    # Idea: Using the probabilities of each return directly, without actually simulating the slot machine
    # chooses random number from 1 to 64, if the number is in the ranges specified in 'ranges_return_val', this is the returned number of coins from one play.

    n_atomic_events = 64
    ranges_return_val = [
        (range(1, 2), 20), # BAR/BAR/BAR
        (range(2, 3), 15), # BELL/BELL/BELL
        (range(3, 4), 5), # LEMON/LEMON/LEMON
        (range(4, 5), 3), # CHERRY/CHERRY/CHERRY
        (range(5, 8), 2), # CHERRY/CHERRY/?
        (range(8, 20), 1), # CHERRY/?/?
    ]

    times_before_broke = []
    for n in range(10000):
        coins = 10
        iterations = 0
        while coins > 0:
            iterations += 1
            coins -= 1
            coins += reward()
        times_before_broke.append(iterations)
    mean = statistics.mean(times_before_broke)
    median = statistics.median(times_before_broke)
    print('Mean: ' + str(mean))
    print('Median: ' + str(median))

def task3part1():
    def p_of_two_people_having_the_same_birthday(N): # task 3, part 1, a
        simulations = 1000
        favorable_simulations = 0
        for _ in range(simulations):
            days_occupied = [False] * 365
            for i in range(N):
                birthday = random.randrange(365)
                if days_occupied[birthday]:
                    favorable_simulations += 1
                    break
                days_occupied[birthday] = True
        return favorable_simulations/simulations

    def comparing_probabilites_with_N_in_range(a, b): # task 3, part 1, b
        p_for_n = []
        for n in range(a, b):
            p_for_n.append(p_of_two_people_having_the_same_birthday(N = n))
        return p_for_n

    p_for_n = comparing_probabilites_with_N_in_range(10, 50 + 1)
    favorable_entries = 0
    smallest = -1
    for i in range(len(p_for_n)):
        p = p_for_n[i]
        if p > 0.5:
            favorable_entries += 1
            if smallest < 0:
                smallest = i + 10
    proportion = favorable_entries/len(p_for_n)
    print("Proportion of N where the event happens with the least 50% chance: " + str(proportion))
    print("Smallest n where probability of the event occuring is at least 50%: " + str(smallest))

def task3part2():
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


def main():
    print("exercise 2:")
    task2()
    print("\nexercise 3, part 1:")
    task3part1()
    print("\nexercise 3, part 2:")
    task3part2()


if __name__ == "__main__":
    main()