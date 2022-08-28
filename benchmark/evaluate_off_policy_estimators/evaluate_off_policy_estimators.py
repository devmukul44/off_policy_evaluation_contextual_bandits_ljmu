import logging
import os
import mlflow
import time
import yaml
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from joblib import delayed
from joblib import Parallel

from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS, Random
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
    SelfNormalizedDoublyRobust,
    DoublyRobustWithShrinkageTuning,
    SelfNormalizedInverseProbabilityWeighting,
    SwitchDoublyRobustTuning,
    InverseProbabilityWeightingTuning,
    DoublyRobustTuning,
    SubGaussianInverseProbabilityWeightingTuning,
    SubGaussianDoublyRobustTuning
)

from ..ag_fs import (
    init_mlflow,
    download_data,
    get_execution_environment
)


class EvaluateOPE:
    def __init__(self,
                 config_name: str,
                 mlflow_exp_id="mudev_open_bandit_pipeline_v1"):

        # init mlflow
        init_mlflow(mlflow_exp_id)

        # configurations for the job
        self.config_file_name = config_name if get_execution_environment() == 'prod' else 'config_stg.yaml'
        logging.info(f"using config_file_name: {self.config_file_name}")

        with open(f"{os.path.join(os.path.dirname(__file__))}/conf/{self.config_file_name}", "rb") as f:
            self.config: dict = yaml.safe_load(f)
            print(f"Job Config: {self.config}")

        # hyperparameters of the regression model used in model dependent OPE estimators
        with open(f"{os.path.join(os.path.dirname(__file__))}/conf/hyperparams.yaml", "rb") as f:
            self.hyperparams: dict = yaml.safe_load(f)
            print(f"hyperparams: {self.hyperparams}")

        self.base_model_dict = dict(
            logistic_regression=LogisticRegression,
            lightgbm=GradientBoostingClassifier,
            random_forest=RandomForestClassifier,
        )

        # Evaluation Policy Choices
        self.evaluation_policy_dict = dict(bts=BernoulliTS, random=Random)

        # compared OPE estimators
        self.ope_estimators = [
            # standard methods
            DirectMethod(estimator_name='dm'),
            InverseProbabilityWeighting(estimator_name='ipw'),
            DoublyRobust(estimator_name='dr'),

            # self normalized methods
            SelfNormalizedInverseProbabilityWeighting(estimator_name='snipw'),
            SelfNormalizedDoublyRobust(estimator_name='sndr'),

            # tuned methods
            # classic tuned methods
            InverseProbabilityWeightingTuning(lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf],
                                              estimator_name='ipw-t'),
            DoublyRobustTuning(lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf], estimator_name='dr-t'),

            # recent tuned methods
            SwitchDoublyRobustTuning(lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf],
                                     estimator_name='switch-dr-t'),
            DoublyRobustWithShrinkageTuning(lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf],
                                            estimator_name='dr-os-t'),
            SubGaussianDoublyRobustTuning(lambdas=[0.0, 0.25, 0.5, 0.75, 1.0], estimator_name='sg-dr-t'),
            SubGaussianInverseProbabilityWeightingTuning(lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
                                                         estimator_name='sg-ipw-t'),
        ]

        # configurations
        # random state - default=12345
        self.random_state = self.config['random_state']
        np.random.seed(self.random_state)

        # base ML model for regression model, logistic_regression, random_forest or lightgbm - choices=["logistic_regression", "lightgbm", "random_forest"],
        self.base_model = self.config['base_model']
        # evaluation policy, bts or random - choices=["bts", "random"]
        self.evaluation_policy = self.config['evaluation_policy']
        # behavior policy, bts or random - choices=["bts", "random"]
        self.behavior_policy = self.config['behavior_policy']
        # campaign name, men, women, or all - choices=["all", "men", "women"]
        self.campaign = self.config['campaign']

        # number of monte carlo simulation to compute the action distribution of evaluation policies - default=1000000
        self.n_sim_to_compute_action_dist = self.config['n_sim_to_compute_action_dist']
        # number of bootstrap sampling in the experiment - default=1
        self.n_runs = self.config['n_runs']
        # the maximum number of concurrently running jobs. - default=1
        self.n_jobs = self.config['n_jobs']
        # bootstrapped sample size
        self.bootstrap_sample_size = self.config['bootstrap_sample_size']

        # save results of the evaluation of off-policy estimators in './logs' directory.
        self.log_path = Path("./logs") / self.behavior_policy / self.campaign
        self.log_path.mkdir(exist_ok=True, parents=True)

    def execute(self):
        start_time = time.time()
        # Model Training and MLFLOW Tracking
        with mlflow.start_run(run_name=self.config_file_name) as run:
            mlflow.log_param("config_file_name", self.config_file_name)
            mlflow.log_params(self.config)
            mlflow.log_dict(self.hyperparams, "hyperparams.yaml")

            # load dataset
            base_data_path = download_data(self.behavior_policy, self.campaign)
            obd_dataset = OpenBanditDataset(
                behavior_policy=self.behavior_policy,
                campaign=self.campaign,
                data_path=base_data_path
            )

            # compute action distribution by evaluation policy
            kwargs = dict(
                n_actions=obd_dataset.n_actions,
                len_list=obd_dataset.len_list,
                random_state=self.random_state
            )
            if self.evaluation_policy == "bts":
                kwargs["is_zozotown_prior"] = True
                kwargs["campaign"] = self.campaign

            # save action distribution of evaluation policy
            # evaluation_policy_action_dist_path = str(log_path / f"{evaluation_policy}_action_dist.npy")
            # np.save(evaluation_policy_action_dist_path, action_dist_evaluation_policy)
            # mlflow.log_artifact(evaluation_policy_action_dist_path)

            # ground-truth policy value of an evaluation policy
            # which is estimated with factual (observed) rewards (on-policy estimation)
            ground_truth_policy_value = OpenBanditDataset.calc_on_policy_policy_value_estimate(
                behavior_policy=self.evaluation_policy,
                campaign=self.campaign,
            )

            def process(b: int):
                # sample bootstrap from batch logged bandit feedback
                bandit_feedback = obd_dataset.sample_bootstrap_bandit_feedback(
                    sample_size=min(self.bootstrap_sample_size, obd_dataset.n_rounds),
                    random_state=b
                )

                # action distribution for evaluation policy
                evaluation_policy_obj = self.evaluation_policy_dict[self.evaluation_policy](**kwargs)

                batch_action_kwargs = dict(n_rounds=bandit_feedback["n_rounds"])
                if self.evaluation_policy == "bts":
                    kwargs["n_sim"] = self.n_sim_to_compute_action_dist
                action_dist_evaluation_policy = evaluation_policy_obj.compute_batch_action_dist(**batch_action_kwargs)

                # estimate the reward function with an ML model
                regression_model = RegressionModel(
                    n_actions=obd_dataset.n_actions,
                    len_list=obd_dataset.len_list,
                    action_context=obd_dataset.action_context,
                    base_model=self.base_model_dict[self.base_model](**self.hyperparams[self.base_model]),
                )

                estimated_rewards_by_reg_model = regression_model.fit_predict(
                    context=bandit_feedback["context"],
                    action=bandit_feedback["action"],
                    reward=bandit_feedback["reward"],
                    position=bandit_feedback["position"],
                    pscore=bandit_feedback["pscore"],
                    n_folds=3,  # 3-fold cross-fitting
                    random_state=self.random_state,
                )

                # evaluate estimators' performances using relative estimation error (relative-ee)
                ope = OffPolicyEvaluation(
                    bandit_feedback=bandit_feedback,
                    ope_estimators=self.ope_estimators,
                )
                relative_ee_b = ope.evaluate_performance_of_estimators(
                    ground_truth_policy_value=ground_truth_policy_value,
                    action_dist=action_dist_evaluation_policy,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    metric="relative-ee",
                )
                se_b = ope.evaluate_performance_of_estimators(
                    ground_truth_policy_value=ground_truth_policy_value,
                    action_dist=action_dist_evaluation_policy,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    metric="se",
                )
                return (relative_ee_b, se_b)

            processed = Parallel(n_jobs=self.n_jobs, verbose=50, )([delayed(process)(i) for i in np.arange(self.n_runs)])

            # metric dict for each estimator
            metric_dict_ee = {est.estimator_name: dict() for est in self.ope_estimators}
            metric_dict_se = {est.estimator_name: dict() for est in self.ope_estimators}

            for b, (relative_ee_b, se_b) in enumerate(processed):
                for (estimator_name, relative_ee_) in relative_ee_b.items():
                    metric_dict_ee[estimator_name][b] = relative_ee_
                for (estimator_name, se_) in se_b.items():
                    metric_dict_se[estimator_name][b] = se_

            results_df_ee = DataFrame(metric_dict_ee).describe().T.round(6)
            results_df_se = DataFrame(metric_dict_se).describe().T.round(6)

            print("=" * 30)
            print(f"random_state={self.random_state}")
            print("-" * 30)
            print(results_df_ee[["mean", "std"]])
            print("=" * 30)

            evaluation_of_ope_results_ee_path = self.log_path / "evaluation_of_ope_results_ee.csv"
            results_df_ee.to_csv(evaluation_of_ope_results_ee_path)
            mlflow.log_artifact(str(evaluation_of_ope_results_ee_path))

            evaluation_of_ope_results_se_path = self.log_path / "evaluation_of_ope_results_se.csv"
            results_df_se.to_csv(evaluation_of_ope_results_se_path)
            mlflow.log_artifact(str(evaluation_of_ope_results_se_path))

            mlflow.log_metric("execution_time_minutes", (time.time() - start_time) / 60)
