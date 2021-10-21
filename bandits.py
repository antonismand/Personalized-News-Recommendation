import numpy as np
import dataset


class LinUCB:
    """
    LinUCB algorithm implementation
    """

    def __init__(self, alpha, context="user"):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        """
        self.n_features = len(dataset.features[0])
        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2

        self.A = np.array([np.identity(self.n_features)] * dataset.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * dataset.n_arms)
        self.b = np.zeros((dataset.n_arms, self.n_features, 1))
        self.alpha = round(alpha, 1)
        self.algorithm = "LinUCB (α=" + str(self.alpha) + ", context:" + context + ")"

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)

        user = np.array([user] * n_pool)
        if self.context == 1:
            x = user
        else:
            x = np.hstack((user, dataset.features[pool_idx]))

        x = x.reshape(n_pool, self.n_features, 1)

        theta = A_inv @ b

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A_inv @ x
        )
        return np.argmax(p)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]  # displayed article's index
        if self.context == 1:
            x = np.array(user)
        else:
            x = np.hstack((user, dataset.features[a]))

        x = x.reshape((self.n_features, 1))

        self.A[a] += x @ x.T
        self.b[a] += reward * x
        self.A_inv[a] = np.linalg.inv(self.A[a])


class ThompsonSampling:
    """
    Thompson sampling algorithm implementation
    """

    def __init__(self):
        self.algorithm = "TS"
        self.alpha = np.ones(dataset.n_arms)
        self.beta = np.ones(dataset.n_arms)

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        theta = np.random.beta(self.alpha[pool_idx], self.beta[pool_idx])
        return np.argmax(theta)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]

        self.alpha[a] += reward
        self.beta[a] += 1 - reward


class Ucb1:
    """
    UCB 1 algorithm implementation
    """

    def __init__(self, alpha):
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        """

        self.alpha = round(alpha, 1)
        self.algorithm = "UCB1 (α=" + str(self.alpha) + ")"

        self.q = np.zeros(dataset.n_arms)  # average reward for each arm
        self.n = np.ones(dataset.n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        ucbs = self.q[pool_idx] + np.sqrt(self.alpha * np.log(t + 1) / self.n[pool_idx])
        return np.argmax(ucbs)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]


class Egreedy:
    """
    Epsilon greedy algorithm implementation
    """

    def __init__(self, epsilon):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        """

        self.e = round(epsilon, 1)  # epsilon parameter for Egreedy
        self.algorithm = "Egreedy (ε=" + str(self.e) + ")"
        self.q = np.zeros(dataset.n_arms)  # average reward for each arm
        self.n = np.zeros(dataset.n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        p = np.random.rand()
        if p > self.e:
            return np.argmax(self.q[pool_idx])
        else:
            return np.random.randint(low=0, high=len(pool_idx))

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]
