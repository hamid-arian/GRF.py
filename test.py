#Import necessary libraries and project files
import numpy as np
import pandas as pd
from data_ import utility
from forest.ForestOptions import ForestOptions
from data_.Data import Data
from splitting.RegressionSplittingRule import RegressionSplittingRule
from relabelling.LLRegressionRelabelingStrategy import LLRegressionRelabelingStrategy
from prediction.LocalLinearPredictionStrategy import LocalLinearPredictionStrategy
from forest.ForestTrainer import ForestTrainer

# Load data from CSV
storages,rc = utility.load_data("crime1.txt")
num_rows = rc[0]
num_cols = rc[1]
data1 = Data(storages, num_rows, num_cols)

# Define forest options
forest_options = ForestOptions(
    num_trees=100,
    ci_group_size=1,
    sample_fraction=0.5,
    mtry=2,
    min_node_size=1,
    honesty=True,
    honesty_fraction=0.5,
    honesty_prune_leaves=False,
    alpha=0.05,
    imbalance_penalty=0.0,
    num_threads=4,
    random_seed=42,
    sample_clusters=None,
    samples_per_cluster=10
)

# Define relabeling_strategy
relabeling_strategy = LLRegressionRelabelingStrategy(
    split_lambda=0.1,  # example value
    weight_penalty = True,  # example flag
    overall_beta = [0] * (len(storages) - 1),  # assuming a beta coefficient for each feature except the target
    ll_split_cutoff = 10,  # example cutoff value
    ll_split_variables = storages
)

# Define splitting_rule
feature_columns = ['rownames','county','year','crmrte','prbarr','prbconv','prbpris','avgsen','polpc','density','taxpc','smsa','pctmin','wcon','wtuc','wtrd','wfir','wser','wmfg','wfed','wsta','wloc','mix','pctymle']
storages_df = pd.DataFrame(storages, columns=feature_columns)
# For max_num_unique_values, we can set this to the maximum number of unique values in any numeric column of the dataset.
max_num_unique_values = storages_df.select_dtypes(include=[np.number]).apply(lambda x: len(x.unique())).max()

splitting_rule = RegressionSplittingRule(
    max_num_unique_values=max_num_unique_values,
    alpha=0.05,
    imbalance_penalty=0.01
)

# Define prediction_strategy
# Convert column names to indices assuming that the linear_correction_variables expects indices
# Here we use df.columns.get_loc to get the index of each column name
linear_correction = [storages_df.columns.get_loc(c) for c in feature_columns]
prediction_strategy = LocalLinearPredictionStrategy(
    lambdas = np.logspace(-4, 2, 10),
    weight_penalty = True,
    linear_correction_variables = linear_correction
)

# Train the forest
forest_trainer = ForestTrainer(
    relabeling_strategy=relabeling_strategy,
    splitting_rule_factory=splitting_rule,
    prediction_strategy=prediction_strategy
)

forest = forest_trainer.train(data1, forest_options)


# Make predictions
# predictor = ForestPredictor(num_threads=4, strategy=Y)
# predictions = predictor.predict(forest, Data,Data)