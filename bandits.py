import numpy as np
import dataset


class LinUCB:
    """
    LinUCB algorithm implementation
    """

    def __init__(self, alpha):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        """

        d = len(dataset.features[
                    0]) * 2  # size for A, b matrices: num of features for articles(6) + num of features for users(6) = 12
        self.A = np.array([np.identity(d)] * dataset.n_arms)
        self.b = np.zeros((dataset.n_arms, d, 1))
        self.alpha = round(alpha, 1)
        self.algorithm = "LinUCB (α=" + str(self.alpha) + ")"

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

        A = self.A[pool_idx]  # (23, 12, 6)
        b = self.b[pool_idx]  # (23, 12, 1)
        user = np.array([user] * len(pool_idx))  # (23, 6)

        A = np.linalg.inv(A)
        x = np.hstack((user, dataset.features[
            pool_idx]))  # (23, 12) The vector x summarizes information of both the user u and arm a

        x = x.reshape((len(pool_idx), 12, 1))  # (23, 12, 1)

        theta = A @ b  # (23, 12, 1)

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A @ x
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

        x = np.hstack((user, dataset.features[a]))
        x = x.reshape((12, 1))

        self.A[a] = self.A[a] + x @ np.transpose(x)
        self.b[a] += reward * x


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
