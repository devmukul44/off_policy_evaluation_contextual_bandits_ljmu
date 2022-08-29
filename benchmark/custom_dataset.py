import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from obp.dataset import OpenBanditDataset
from sklearn.preprocessing import LabelEncoder


@dataclass
class OpenBanditDatasetWithInteractionFeatures(OpenBanditDataset):
    """
    -- param --
    context_set -> feature engineering - acceptable values [1,2,3,4]
        context_set=1 -> default feature engineering from OpenBanditDataset in OBP (context => `user features`)
        context_set=2 -> context => `user features` + `user item affinity features`
        context_set=3 -> context => `user features` + `item features`
        context_set=4 -> context => `user features` + `item features` + `user item affinity features`
    """
    context_set: int = 1

    def pre_process(self) -> None:
        if self.context_set == 1:
            super().pre_process()
        elif self.context_set == 2:
            self._pre_process_with_user_item_affinity_features()
        elif self.context_set == 3:
            self._pre_process_with_item_context()
        elif self.context_set == 4:
            self._pre_process_with_item_context_and_user_item_affinity_features()
        else:
            raise NotImplementedError

    def _pre_process_with_user_item_affinity_features(self) -> None:
        """
        context => `user features` + `user item affinity features`
        :return:
        """
        logging.info("executing: _pre_process_with_user_item_affinity_features")
        super().pre_process()
        affinity_cols = self.data.columns.str.contains("affinity")
        Xaffinity = self.data.loc[:, affinity_cols].values

        self.context = PCA(n_components=30).fit_transform(
            np.c_[self.context, Xaffinity]
        )

    def _pre_process_with_item_context(self) -> None:
        """
        context => `user features` + `item features`
        :return:
        """
        logging.info("executing: _pre_process_with_item_context")
        super().pre_process()

        fact_data = self.data \
            .join(self.item_context, on='item_id', lsuffix='_1', rsuffix='_2') \
            .drop('item_id_2', axis=1) \
            .rename(columns={'item_id_1': 'item_id'})

        user_cols = fact_data.columns.str.contains("user_feature")
        user_cols_df = pd.get_dummies(
            fact_data.loc[:, user_cols], drop_first=True
        )
        item_feature_0 = fact_data["item_feature_0"].to_frame()
        item_feature_cat = fact_data[['item_feature_1','item_feature_2','item_feature_3']]\
            .apply(LabelEncoder().fit_transform)

        self.context = pd.concat(
            objs=[user_cols_df, item_feature_cat, item_feature_0], axis=1
        ).values

    def _pre_process_with_item_context_and_user_item_affinity_features(self) -> None:
        """
        context => `user features` + `item features` + `user item affinity features`
        :return:
        """
        logging.info("executing _pre_process_with_item_context_and_user_item_affinity_features")
        self._pre_process_with_item_context()
        # add user-item affinity features
        affinity_cols = self.data.columns.str.contains("affinity")
        Xaffinity = self.data.loc[:, affinity_cols].values

        self.context = PCA(n_components=35).fit_transform(
            np.c_[self.context, Xaffinity]
        )
