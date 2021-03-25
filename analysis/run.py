from lenskit.datasets import ML1M

import analysis.catalog_coverage
import analysis.gini_index
import analysis.delta_gaps
import analysis.pop_recs_correlation
import analysis.pop_ratio_profile_vs_recs
import analysis.recs_long_tail_distr
import analysis.novelty
import analysis.serendipity
import analysis.bins

import pandas as pd

################## EDIT HERE TO CHANGE CONFIGS ############################
ALGORITHM_NAME = 'word-2-vec'
PLOT_FILE_NAME = 'word-2-vec'
RECS_PATH = '../recs/cb-word-embedding/word-2-vec.csv' # input recs file
###########################################################################

recs = pd.read_csv(RECS_PATH)

ratings = pd.read_csv('../datasets/goodbooks-10k-master/ratings.csv')

analysis.catalog_coverage.run(recs, ALGORITHM_NAME, ratings)
analysis.gini_index.run(recs, ALGORITHM_NAME, ratings)
analysis.delta_gaps.run(recs, ALGORITHM_NAME, ratings)
analysis.pop_recs_correlation.run(recs, ALGORITHM_NAME, ratings, PLOT_FILE_NAME)
analysis.pop_ratio_profile_vs_recs.run(recs, ALGORITHM_NAME, PLOT_FILE_NAME)
analysis.recs_long_tail_distr.run(recs, ALGORITHM_NAME, PLOT_FILE_NAME)
analysis.novelty.run(recs, ALGORITHM_NAME, ratings)
analysis.serendipity.run(recs, ALGORITHM_NAME, ratings)
analysis.bins.run(recs, ALGORITHM_NAME, ratings)

