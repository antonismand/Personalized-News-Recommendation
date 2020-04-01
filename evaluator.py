import dataset
import random
import time


# class evaluate:
# def __init__(s, split_ratio=0.1):
# s.split_ratio = split_ratio

def evaluate(A, mode, verbose=1, size=100, split_ratio=0.1):
    start = time.time()
    G = 0  # total payoff
    T = 0  # counter of valid events
    if size == 100:
        events = dataset.events
        num_events = dataset.n_events
    else:
        k = int(dataset.n_events * size / 100)
        events = random.sample(dataset.events, k)
        num_events = len(events)

    for t, event in enumerate(events):
        if (t / num_events < split_ratio and mode == "learn") or (
                t / num_events > split_ratio and mode == "deploy"
        ):
            displayed = event[0]
            reward = event[1]
            user = event[2]
            pool_idx = event[3]

            chosen = A.choose_arm(t, user, pool_idx)
            if chosen == event[0]:
                A.update(displayed, reward, user, pool_idx)
                G += event[1]
                T += 1
            if (verbose == 2 and t % 100000 == 0 and t > 0) or (
                    verbose == 3 and t % 50000 == 0 and t > 0
            ):
                print('{:<5}{:<20}{}'.format(str(round(t / num_events * 100)) + '%', A.algorithm, round(G / T, 3)))
    end = time.time()
    if verbose > 0:
        execution_time = round(end - start, 1)
        execution_time = (
            str(round(execution_time / 60,1)) + "m"
            if execution_time > 60
            else str(execution_time) + "s"
        )
        print('{:<10}{:<20}{:<10}{}'.format(mode, A.algorithm, round(G / T, 3), execution_time))
    return G / T, end - start
