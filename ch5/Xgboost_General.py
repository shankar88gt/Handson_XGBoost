"""Extreme Graient Boosting
    built in Regularization & Speed
    Extreme means pushing the computation limits to Extreme

Handling Missing Values
Ganing Speed
    Approximate split finding algorithm
        in general split is greedy and does not backtrack
        split finding uses quantiles for candidate splits; global & local
        quantile sketch
    Sparsity aware split-finding
        sparse matrix for memory saves
    parallel computing
        storage in blocks for parallel computing
    cache aware access
        allocates internal buffer , fetch gradient statistics & accumulation with mini batches - 50% improvement
    block compression & sharing
        data sharing multiple disls that alternate when reading the data
    Includes regularization as part of learning objective
"""
