import random

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

def main():
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

if __name__ == '__main__':
    main()