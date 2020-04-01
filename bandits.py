import numpy as np
import dataset


class linucb:
    def __init__(s, alpha):
        d = len(dataset.features[0]) * 2
        s.A = np.array([np.identity(d)] * dataset.n_arms)
        s.b = np.zeros((dataset.n_arms, d, 1))
        s.alpha = round(alpha,1)
        s.algorithm = "LinUCB (α=" + str(s.alpha) + ")"

    def choose_arm(s, t, user, pool_idx):

        A = s.A[pool_idx]  # (23, 12, 6)
        b = s.b[pool_idx]  # (23, 12, 1)
        user = np.array([user] * len(pool_idx))  # (23, 6)

        A = np.linalg.inv(A)
        x = np.hstack((user, dataset.features[pool_idx]))  # (23, 12)

        x = x.reshape((len(pool_idx), 12, 1))  # (23, 12, 1)

        theta = A @ b  # (23, 12, 1)

        p = np.transpose(theta, (0, 2, 1)) @ x + s.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A @ x
        )
        return np.argmax(p)

    def update(s, displayed, reward, user, pool_idx):
        a = pool_idx[displayed]

        x = np.hstack((user, dataset.features[a]))
        x = x.reshape((12, 1))

        s.A[a] = s.A[a] + x @ np.transpose(x)
        s.b[a] += reward * x


class thompson_sampling:
    def __init__(s):
        s.algorithm = "TS"
        s.alpha = np.ones(dataset.n_arms)
        s.beta = np.ones(dataset.n_arms)

    def choose_arm(s, t, user, pool_idx):
        theta = np.random.beta(s.alpha[pool_idx], s.beta[pool_idx])
        return np.argmax(theta)

    def update(s, displayed, reward, user, pool_idx):
        a = pool_idx[displayed]

        s.alpha[a] += reward
        s.beta[a] += 1 - reward


class ucb1:
    def __init__(s, alpha):
        s.alpha = round(alpha,1)
        s.algorithm = "UCB1 (α=" + str(s.alpha) + ")"
        
        s.q = np.zeros(dataset.n_arms)
        s.n = np.ones(dataset.n_arms)

    def choose_arm(s, t, user, pool_idx):
        ucbs = s.q[pool_idx] + np.sqrt(s.alpha * np.log(t + 1) / s.n[pool_idx])
        return np.argmax(ucbs)

    def update(s, displayed, reward, user, pool_idx):
        a = pool_idx[displayed]

        s.n[a] += 1
        s.q[a] += (reward - s.q[a]) / s.n[a]


class egreedy:
    def __init__(s, epsilon):
        s.e = round(epsilon, 1)
        s.algorithm = "egreedy (ε=" + str(s.e) + ")"
        s.q = np.zeros(dataset.n_arms)
        s.n = np.zeros(dataset.n_arms)
        

    def choose_arm(s, t, user, pool_idx):
        p = np.random.rand()
        if p > s.e:
            return np.argmax(s.q[pool_idx])
        else:
            return np.random.randint(low=0, high=len(pool_idx))

    def update(s, displayed, reward, user, pool_idx):
        a = pool_idx[displayed]

        s.n[a] += 1
        s.q[a] += (reward - s.q[a]) / s.n[a]
