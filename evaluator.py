import dataset
import random
import time


def evaluate(A, size=100, learn_ratio = 0.9):
    """
    Policy evaluator as described in the paper

    Parameters
    ----------
    A : class
        algorithm
    size : number
        Run the evaluation only on a portion of the dataset
    learn_ratio : number
        Perform learning(update parameters) only on a small portion of the traffic


    Returns
    -------
    learn : array
        contains the ctr for each trial for the learning bucket
    deploy : array
        contains the ctr for each trial for the deployment bucket
    """
    
    start = time.time()
    G_deploy = 0 # total payoff for the deployment bucket
    G_learn = 0  # total payoff for the learning bucket
    T_deploy = 1 # counter of valid events for the deployment bucket
    T_learn = 0  # counter of valid events for the learning bucket

    learn = []
    deploy = []
    if size == 100:
        events = dataset.events
    else:
        k = int(dataset.n_events * size / 100)
        events = random.sample(dataset.events, k)

    for t, event in enumerate(events):

        displayed = event[0]
        reward = event[1]
        user = event[2]
        pool_idx = event[3]

        chosen = A.choose_arm(G_learn + G_deploy, user, pool_idx)
        if chosen == displayed:
            if random.random() < learn_ratio:
                G_learn += event[1]
                T_learn += 1
                A.update(displayed, reward, user, pool_idx)
                learn.append(G_learn / T_learn)
            else:
                G_deploy += event[1]
                T_deploy += 1
                deploy.append(G_deploy / T_deploy)

    end = time.time()

    execution_time = round(end - start, 1)
    execution_time = (
        str(round(execution_time / 60, 1)) + "m"
        if execution_time > 60
        else str(execution_time) + "s"
    )
    print(
        "{:<20}{:<10}{}".format(
            A.algorithm, round(G_deploy / T_deploy, 4), execution_time
        )
    )

    return learn, deploy
