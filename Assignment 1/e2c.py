import random
import statistics

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

def main():
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

def reward(): # Simulates one play, returns reward
    random_number = random.randrange(1, n_atomic_events+1)
    for tuple in ranges_return_val:
        if random_number in tuple[0]:
            return tuple[1]
    return 0


if __name__ == "__main__":
    main()