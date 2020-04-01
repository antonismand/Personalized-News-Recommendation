## Bandits implemented
- UCB
- Thompson Sampling
- E-greedy
- LinUCB with disjoint linear models

## References
- ### A Contextual-Bandit Approach to Personalized News Article Recommendation https://arxiv.org/pdf/1003.0146.pdf
- ### Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms  https://arxiv.org/pdf/1003.5956.pdf
    Used algorithm 2 as a policy evaluator (for finite data stream)


## Dataset
### R6A - Yahoo! Front Page Today Module User Click Log Dataset, version 1.0 (1.1 GB)
The dataset contains 45,811,883 user visits to the Today Module. For each visit, both the user and each of the candidate articles are associated with a feature vector of dimension 6 (including a constant feature), constructed using a conjoint analysis with a bilinear model.
The dataset can be found [here](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r).