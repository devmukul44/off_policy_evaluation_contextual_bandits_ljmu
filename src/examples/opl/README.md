# Example with Off-Policy Policy Learners


## Description

We use synthetic bandit data to evaluate some off-policy learners using their ground-truth policy value calculable with synthetic data.

## Evaluating Off-Policy Learners

In the following, we evaluate the performances of

- Uniform Random Policy (`Random`)
- Inverse Probability Weighting Policy Learner (`IPWLearner`)
- Policy Learner using Neural Networks (`NNPolicyLearner`)

See [our documentation](https://zr-obp.readthedocs.io/en/latest/_autosummary/obp.policy.offline.html) for the details about `IPWLearner` and `NNPolicyLearner`.

`NNPolicyLearner` can use the following OPE estimators as the objective function:
- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Doubly Robust (DR)

See [our documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators.

### Files
- [`./evaluate_off_policy_learners.py`](./evaluate_off_policy_learners.py) implements the evaluation of off-policy learners using synthetic bandit data.
- [`./conf/hyperparams.yaml`](./conf/hyperparams.yaml) defines hyperparameters of some ML methods used to define regression model and IPWLearner.

### Scripts

```bash
# run evaluation of off-policy learners with synthetic_dataset bandit data
python evaluate_off_policy_learners.py\
    --n_rounds $n_rounds\
    --n_actions $n_actions\
    --dim_context $dim_context\
    --beta $beta\
    --base_model_for_evaluation_policy $base_model_for_evaluation_policy\
    --base_model_for_reg_model $base_model_for_reg_model\
    --off_policy_objective $off_policy_objective\
    --n_hidden $n_hidden\
    --n_layers $n_layers\
    --activation $activation\
    --solver $solver\
    --batch_size $batch_size\
    --early_stopping\
    --random_state $random_state
```
- `$n_rounds` and `$n_actions` specify the sample size and the number of actions of the synthetic bandit data, respectively.
- `$dim_context` specifies the dimension of context vectors.
- `$beta` specifies the inverse temperature parameter to control the behavior policy.
- `$base_model_for_ipw_learner` specifies the base ML model for defining evaluation policy and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$off_policy_objective` specifies the OPE estimator for NNPolicyLearner and should be one of "dm", "ipw", or "dr".
- `$n_hidden` specifies the size of hidden layers in NNPolicyLearner.
- `$n_layers` specifies the number of hidden layers in NNPolicyLearner.
- `$activation` specifies the activation function for NNPolicyLearner and should be one of "identity", "tanh", "logistic", or "relu".
- `$solver` specifies the optimizer for NNPolicyLearner and should be one of "adagrad", "sgd", or "adam".
- `$batch_size` specifies the batch size for NNPolicyLearner.
- `$early_stopping` enables early stopping of training of NNPolicyLearner.

For example, the following command compares the performance of the off-policy learners using synthetic bandit data with 100,00 rounds, 10 actions, five dimensional context vectors.

```bash
python evaluate_off_policy_learners.py\
    --n_rounds 10000\
    --n_actions 10\
    --dim_context 5\
    --base_model_for_ipw_learner logistic_regression\
    --off_policy_objective ipw\
    --n_hidden 100\
    --n_layers 1\
    --activation relu\
    --solver adam\
    --batch_size 200\
    --early_stopping

# policy values of off-policy learners (higher means better)
# =============================================
# random_state=12345
# ---------------------------------------------
#                               policy value
# random_policy                     0.499925
# ipw_learner                       0.782430
# nn_policy_learner (with ipw)      0.735947
# =============================================
```

The above result can change with different situations. You can try the evaluation with other experimental settings easily.

