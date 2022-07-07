# QMF - a matrix factorization library

[![Build Status](https://travis-ci.org/quora/qmf.svg?branch=master)](https://travis-ci.org/quora/qmf)
[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)](LICENSE)

## Introduction

QMF is a fast and scalable C++ library for implicit-feedback matrix factorization models. The current implementation supports two main algorithms:

* **Weighted ALS** [1]. This model optimizes a weighted squared loss, and thus allows you to specify different weights on each positive example. The algorithm is based on alternating minimization on user and item factors matrices. QMF uses efficient parallelization to perform these minimizations.
* **BPR** [2]. This model (approximately) optimizes average per-user AUC using stochastic gradient descent (SGD) on randomly sampled (user, positive item, negative item) triplets. Asynchronous, parallel Hogwild! [3] updates are supported in QMF to achieve near-linear speedup in the number of processors (when the dataset is sparse enough).

For evaluation, QMF supports various ranking-based metrics that are computed per-user on test data, in addition to training or test objective values.

For more information, see our blog post about QMF here: https://engineering.quora.com/Open-sourcing-QMF-for-matrix-factorization.

## Building QMF

QMF requires gcc 5.0+, as it uses the C++14 standard, and CMake version 2.8+. It also depends on glog, gflags and lapack libraries.

### Ubuntu

To install libraries dependencies:
```
sudo apt-get install liblapack-dev
```

To build the binaries:
```
cmake .
make
```
To run tests:

```
make test
```

Output binaries will be under the `bin/` folder.

## Usage

Here's a basic example of usage:
```
# to train a WALS model
./wals \
    --train_dataset=<train_dataset> \
    --test_dataset=<test_dataset> \
    --user_factors=<user_factors_file> \
    --item_factors=<item_factors_file> \
    --regularization_lambda=0.05 \
    --confidence_weight=40 \
    --nepochs=10 \
    --nfactors=30 \
    --nthreads=4

# to train a BPR model
./bpr \
    --train_dataset=<train_dataset> \
    --test_dataset=<test_dataset> \
    --user_factors=<user_factors_file> \
    --item_factors=<item_factors_file> \
    --nepochs=10 \
    --nfactors=30 \
    --num_hogwild_threads=4 \
    --nthreads=4
```
The input dataset files should adhere to the following format:
```
<user_id1> <item_id1> <weight1>
<user_id2> <item_id2> <weight2>
...
```
where `weight` is always `1` in BPR, but can be any integer in WALS (`r_ui` in the paper [1]).

The output files will be in the following format:
```
<{user|item}_id> [<bias>] <factor_0> <factor_1> ... <factor_k-1>
...
```
where the bias term will only be present for BPR item factors when the `--use_biases` option is specified.

In order to compute test ranking metrics (averaged per-user), you can add the following parameters to either binary:
* `--test_avg_metrics=<metric1[,metric2,...]>` specifies the metrics, which include `auc` (area under the ROC curve), `ap` (average precision), `p@k` (e.g. `p@10` for precision at 10), `r@k` (recall at k)
* `--num_test_users=<nusers>` specifies the number of users to consider when computing test metrics (by default 0 = all users). Computing these metrics requires computing predicted scores for all items and test users, which can be slow as the number of user gets big. The users are picked uniformely at random with a fixed seed (which can be specified with `--eval_seed`)
* `--test_always` will compute these metrics after each epoch (by default they're computed only after the last epoch)

In the case of BPR, a set of (user, positive item, negative item) triplets is sampled during initialization for both training and test sets (with a fixed seed, or as given by `--eval_seed`), and is used to compute an estimate of the loss after each epoch. This has no effect on training or on the computation of ranking metrics.

Options for WALS:
* `--nepochs` (default 10): number of iterations of alternating least squares
* `--nfactors` (default 30): dimensionality of the learned user and item factors
* `--regularization_lambda`: regularization coefficient
* `--confidence_weight`: weight multiplier for positive items (alpha in the paper [1])
* `--init_distribution_bound` (default 0.01): bound (in absolute value) on weight initialization (with the default, weights are initialized uniformly between -0.01 and 0.01)

Options for BPR:
* `--nepochs` (default 10): number of iterations of SGD
* `--nfactors` (default 30): dimensionality of the learned user and item factors
* `--use_biases` (default false): whether to use additive item biases
* `--user_lambda`: regularization coefficient on user factors
* `--item_lambda`: regularization coefficient on item factors
* `--bias_lambda`: regularization coefficient on biases
* `--init_learning_rate`: initial learning rate
* `--decay_rate` (default 0.9): multiplicative decay applied to the learning rate after each epoch
* `--init_distribution_bound` (default 0.01): bound (in absolute value) on weight initialization (with the default, weights are initialized uniformly between -0.01 and 0.01)
* `--num_negative_samples` (default 3): number of random negatives sampled for each positive item
* `--num_hogwild_threads` (default 1): number of parallel hogwild threads to use for SGD (in contrast, `--nthreads` determines parallelism for deterministic operations, e.g. for evaluation)
* `--eval_num_neg` (default 3): number of random negatives per positive used to generate the fixed evaluation sets mentioned above (used for computing train/test loss, does not affect training or ranking metrics)

For more details on the command-line options, see the definitions in `wals.cpp` and `bpr.cpp`.

## Credits

This library was built at Quora by [Denis Yarats](https://github.com/1nadequacy) and [Alberto Bietti](https://github.com/albietz).

## License
QMF is released under the [Apache 2.0 Licence](https://github.com/quora/qmf/blob/master/LICENSE).

## References

[1] Hu, Koren and Volinsky. Collaborative Filtering for Implicit Feedback Datasets. In *ICDM* 2008.

[2] Rendle, Freudenthaler, Gantner and Schmidt-Thieme. BPR: Bayesian Personalized Ranking from Implicit Feedback. In *UAI* 2009.

[3] Niu, Recht, Ré and Wright. Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent. In *NIPS* 2011.
