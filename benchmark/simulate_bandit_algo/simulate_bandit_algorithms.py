import logging
import os
import mlflow
import time
import yaml
import argparse
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# plotting deps
from matplotlib import pyplot as plt
import seaborn as sns

from joblib import delayed
from joblib import Parallel

from obp.dataset import OpenBanditDataset
from obp.policy.policy_type import PolicyType
from obp.simulator import run_bandit_simulation

from obp.policy import (
    # base classes
    BaseContextFreePolicy,
    BaseContextualPolicy,

    # ContextFree
    Random,
    EpsilonGreedy,
    BernoulliTS,

    # Contextual - Linear
    LinEpsilonGreedy,
    LinUCB,
    LinTS,

    # Contextual - Logistic
    LogisticEpsilonGreedy,
    LogisticUCB,
    LogisticTS,
)

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


class SimulateBandit:
    def __init__(self,
                 config_name: str,
                 mlflow_exp_id="mudev_open_bandit_pipeline_v2"):

        # init mlflow
        self.mlflow_exp_id = init_mlflow(mlflow_exp_id)

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
            logistic=LogisticRegression,
            lightgbm=GradientBoostingClassifier,
            random_forest=RandomForestClassifier,
        )

        # Evaluation Policy Choices
        self.simulation_policy_dict = dict(bts=BernoulliTS, random=Random)

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
            InverseProbabilityWeightingTuning(
                lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf],
                estimator_name='ipw-t'
            ),
            DoublyRobustTuning(
                lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf],
                estimator_name='dr-t'
            ),

            # recent tuned methods
            SwitchDoublyRobustTuning(
                lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf],
                estimator_name='switch-dr-t'
            ),
            DoublyRobustWithShrinkageTuning(
                lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf],
                estimator_name='dr-os-t'
            ),
            SubGaussianDoublyRobustTuning(
                lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
                estimator_name='sg-dr-t'
            ),
            SubGaussianInverseProbabilityWeightingTuning(
                lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
                estimator_name='sg-ipw-t'
            ),
        ]

        """
        configurations:
            - behavior_policy - bts or random - choices=["bts", "random"]
            - campaign - campaign name, men, women, or all - choices=["all", "men", "women"]
            - base_model - base ML model for regression model, logistic, random_forest or lightgbm - choices=["logistic_", "lightgbm", "random_forest"]
            - random state - default=12345
        """
        self.behavior_policy = self.config['behavior_policy']
        self.campaign = self.config['campaign']
        self.base_model = self.config['base_model']
        self.random_state = self.config['random_state']
        np.random.seed(self.random_state)

        """
        Params:
            - batch_update_size - Number of samples used in a batch parameter update.
            - n_sim_to_compute_action_dist - number of monte carlo simulation to compute the action distribution of evaluation policies - default=1000000
            - n_jobs - the maximum number of concurrently running jobs. - default=1
            - bootstrap_sample_size_bandit_feedback - bootstrapped sample size for bandit feedback
            - n_bootstrap_samples_for_ope - Number of resampling performed in bootstrap sampling for OPE
        """
        self.batch_update_size = self.config['batch_update_size']
        self.n_sim_to_compute_action_dist = self.config['n_sim_to_compute_action_dist']
        self.n_jobs = self.config['n_jobs']
        self.bootstrap_sample_size_bandit_feedback = self.config['bootstrap_sample_size_bandit_feedback']
        self.n_bootstrap_samples_for_ope = self.config['n_bootstrap_samples_for_ope']

        # save results of the evaluation of off-policy estimators in './logs' directory.
        self.log_path = Path("./logs") / self.behavior_policy / self.campaign
        self.log_path.mkdir(exist_ok=True, parents=True)

    def get_simulation_bandit_policy_list(
            self,
            obd_dataset: OpenBanditDataset):
        # general kwargs
        kwargs = dict(
            n_actions=obd_dataset.n_actions,
            len_list=obd_dataset.len_list,
            batch_size=self.batch_update_size,
            random_state=self.random_state
        )
        simulation_policy_list = [
            # - ContextFree -
            # - EpsilonGreedy -epsilon [0,1]
            EpsilonGreedy(epsilon=0.05, **kwargs),  # 5% exploration
            EpsilonGreedy(epsilon=0.1, **kwargs),  # 10% exploration
            Random(**kwargs),  # 100% exploration

            # - BernoulliTS -
            # prior weights
            BernoulliTS(
                policy_name="bts-obp",
                is_zozotown_prior=True,
                campaign=self.campaign,
                **kwargs
            ),
            # no-priors
            BernoulliTS(policy_name="bts", **kwargs),

            # # - Contextual - Linear -
            #
            # # - LinEpsilonGreedy - epsilon [0,1]
            # # 5% exploration
            # LinEpsilonGreedy(
            #     dim=obd_dataset.dim_context,
            #     epsilon=0.05,
            #     **kwargs
            # ),
            # # 10% exploration
            # LinEpsilonGreedy(
            #     dim=obd_dataset.dim_context,
            #     epsilon=0.1,
            #     **kwargs
            # ),
            #
            # # - LinUCB - epsilon [0, inf]
            # # epsilon 0
            # LinUCB(
            #     dim=obd_dataset.dim_context,
            #     **kwargs
            # ),
            # # epsilon 1
            # LinUCB(
            #     dim=obd_dataset.dim_context,
            #     epsilon=1.0,
            #     **kwargs
            # ),
            # # - LinTS -
            # LinTS(
            #     dim=obd_dataset.dim_context,
            #     **kwargs
            # ),
            #
            # # - Contextual - Logistic -
            #
            # # - LogisticEpsilonGreedy - epsilon [0,1]
            # # 5% exploration
            # LogisticEpsilonGreedy(
            #     dim=obd_dataset.dim_context,
            #     epsilon=0.05,
            #     **kwargs
            # ),
            # # 10% exploration
            # LogisticEpsilonGreedy(
            #     dim=obd_dataset.dim_context,
            #     epsilon=0.1,
            #     **kwargs
            # ),
            #
            # # - LogisticUCB -
            # # epsilon [0,inf]
            # LogisticUCB(
            #     dim=obd_dataset.dim_context,
            #     **kwargs
            # ),
            # # epsilon 1
            # LogisticUCB(
            #     dim=obd_dataset.dim_context,
            #     epsilon=1.0,
            #     **kwargs
            # ),
            #
            # # - LogisticTS -
            # LogisticTS(
            #     dim=obd_dataset.dim_context,
            #     **kwargs
            # ),
        ]
        return simulation_policy_list

    def execute(self):
        start_time = time.time()
        # Model Training and MLFLOW Tracking

        def plot_action_dist(action_dist: np.ndarray, title):
            plt.style.use("ggplot")
            fig, ax = plt.subplots(figsize=(25, 15))
            pd.Series(
                action_dist.mean(axis=0).mean(axis=1)
            ).plot(kind='bar', figsize=(20, 10), ax=ax)

            plt.title("Action Choice Probability : " + str(title), fontsize=25)
            plt.xlabel("Action Index", fontsize=20)
            plt.ylabel("Probability", fontsize=20)
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=15)
            # plt.savefig(str(self.log_path / f"{title}.png"))
            return fig

        def plot_ope_summary(estimated_interval: pd.DataFrame, title: str, relative=False):
            plt.style.use("ggplot")
            fig, ax = plt.subplots(figsize=(15, 10))
            if relative:
                sns.barplot(estimated_interval.T / bandit_feedback["reward"].mean(), ax=ax)
            else:
                sns.barplot(data=estimated_interval.T, ax=ax)

            plt.title(f"Estimated Policy Value for " + str(title), fontsize=25)
            plt.xlabel("OPE Estimators", fontsize=20)
            plt.ylabel(f"Estimated Policy Value (± 95% CI)", fontsize=20)
            plt.yticks(fontsize=22.5)
            plt.xticks(fontsize=32.5 - len(self.ope_estimators), rotation=45)

            return fig

        # mlflow run
        with mlflow.start_run(run_name=self.config_file_name) as run:
            mlflow.log_param("config_file_name", self.config_file_name)
            mlflow.log_params(self.config)
            mlflow.log_dict(self.hyperparams, "hyperparams.yaml")

            # load dataset
            base_data_path = download_data(self.behavior_policy, self.campaign)
            obd_dataset: OpenBanditDataset = OpenBanditDataset(
                behavior_policy=self.behavior_policy,
                campaign=self.campaign,
                data_path=base_data_path
            )

            # sample bootstrap from batch logged bandit feedback
            if self.bootstrap_sample_size_bandit_feedback < 1:
                bandit_feedback = obd_dataset.obtain_batch_bandit_feedback()
            else:
                bandit_feedback = obd_dataset.sample_bootstrap_bandit_feedback(
                    sample_size=min(self.bootstrap_sample_size_bandit_feedback, obd_dataset.n_rounds),
                    random_state=self.random_state
                )

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
            action_dist_reg_model_df = pd.DataFrame(estimated_rewards_by_reg_model.mean(axis=0))
            logging.info(str(action_dist_reg_model_df.head()))
            action_dist_path = self.log_path / f"{self.base_model}_reg_action_dist.csv"
            action_dist_reg_model_df.to_csv(action_dist_path)
            mlflow.log_artifact(str(action_dist_path))
            mlflow.log_figure(
                plot_action_dist(estimated_rewards_by_reg_model,
                                 f"{str(self.base_model).capitalize()} Regression"),
                f"{self.base_model}_action_distribution.png"
            )

            # compute action distribution by evaluation policy
            bandit_sim_policy_list = self.get_simulation_bandit_policy_list(obd_dataset=obd_dataset)
            for i in bandit_sim_policy_list:
                mlflow.log_param(i.policy_name, str(i))

            # parallel process for bandit policy simulation and ope estimation
            def process(b: int, bandit_sim_policy: Union[BaseContextFreePolicy, BaseContextualPolicy]):
                process_start_time = time.time()
                # initialize mlflow for each process
                init_mlflow(self.mlflow_exp_id)
                bandit_sim_policy_class_name = str(bandit_sim_policy.__class__).split(".")[-1]
                try:
                    bandit_sim_policy_name = bandit_sim_policy.policy_name
                except:
                    bandit_sim_policy_name = bandit_sim_policy_class_name
                try:
                    bandit_hyper_parameter = bandit_sim_policy.epsilon
                except:
                    bandit_hyper_parameter = None
                with mlflow.start_run(run_name=f"{self.config_file_name}_{bandit_sim_policy_name}") as run:
                    mlflow.log_param("config_file_name", self.config_file_name)
                    mlflow.log_params(self.config)
                    mlflow.log_dict(self.hyperparams, "hyperparams.yaml")
                    mlflow.log_param(f"bandit_sim_policy_name", bandit_sim_policy_name)
                    mlflow.log_param(f"hyper_parameter", bandit_hyper_parameter)

                    if bandit_sim_policy_name == 'bts-obp':
                        mlflow.log_param("action_dist_method", "bts-obp found - compute_batch_action_dist")
                        action_dist_evaluation_policy = bandit_sim_policy.compute_batch_action_dist(
                            n_sim=self.n_sim_to_compute_action_dist, n_rounds=bandit_feedback["n_rounds"]
                        )
                    else:
                        mlflow.log_param("action_dist_method", "run_bandit_simulation")
                        action_dist_evaluation_policy = run_bandit_simulation(
                            bandit_feedback=bandit_feedback,
                            policy=bandit_sim_policy
                        )

                    action_dist_evaluation_policy_df = pd.DataFrame(action_dist_evaluation_policy.mean(axis=0))
                    action_dist_path = self.log_path / f"{bandit_sim_policy_name}_action_dist.csv"
                    action_dist_evaluation_policy_df.to_csv(action_dist_path)
                    mlflow.log_artifact(str(action_dist_path))

                    mlflow.log_figure(
                        plot_action_dist(action_dist_evaluation_policy, f"{str(bandit_sim_policy_name).capitalize()} Policy"),
                        f"{bandit_sim_policy_name}_action_distribution.png"
                    )

                    # evaluate estimators' performances using relative estimation error (relative-ee)
                    ope = OffPolicyEvaluation(
                        bandit_feedback=bandit_feedback,
                        ope_estimators=self.ope_estimators,
                    )

                    # `summarize_off_policy_estimates` returns pandas dataframes including the OPE results
                    # the estimated policy value of the evaluation policy
                    # relative_estimated_policy_value is the policy value of the evaluation policy relative to the ground-truth policy value of the behavior policy
                    estimated_policy_value, estimated_interval = ope.summarize_off_policy_estimates(
                        action_dist=action_dist_evaluation_policy,
                        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                        n_bootstrap_samples=self.n_bootstrap_samples_for_ope,  # number of resampling performed in bootstrap sampling.
                        random_state=self.random_state,
                    )
                    # log summary
                    ope_summary_df = pd.concat([estimated_policy_value, estimated_interval],axis=1)
                    ope_summary_path = self.log_path / f"{bandit_sim_policy_name}_ope_summary.csv"
                    ope_summary_df.to_csv(ope_summary_path)
                    mlflow.log_artifact(str(ope_summary_path))

                    mlflow.log_figure(
                        plot_ope_summary(estimated_interval,
                                         f"{str(bandit_sim_policy_name).capitalize()} Policy",
                                         relative=False),
                        f"{bandit_sim_policy_name}_ope_summary.png"
                    )
                    mlflow.log_figure(
                        plot_ope_summary(estimated_interval,
                                         f"{str(bandit_sim_policy_name).capitalize()} Policy",
                                         relative=True),
                        f"{bandit_sim_policy_name}_ope_summary_relative.png"
                    )
                    mlflow.log_metric("execution_time_minutes", (time.time() - process_start_time) / 60)

            Parallel(n_jobs=self.n_jobs, verbose=50, ) \
                ([delayed(process)(i, bandit_policy) for i, bandit_policy in enumerate(bandit_sim_policy_list)])

            mlflow.log_metric("execution_time_minutes", (time.time() - start_time) / 60)
