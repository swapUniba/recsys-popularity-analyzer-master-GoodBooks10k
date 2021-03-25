from lenskit.datasets import ML1M
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import FunctionTransformer

import pandas as pd
import numpy as np
import random, math


class Technique(Enum):
    LOG_REGR = 1
    RANDOM_FOREST = 2
    SVM = 3
    KNN = 4
    DECISION_TREE = 5
    GAUSSIAN_PROCESS = 6


################## EDIT HERE TO CHANGE CONFIGS ########################################
MIN_POSITIVE_RATING = 4  # minimum rating to consider an item as liked
NUM_OF_RECS = 10  # num of recs for each user
OUTPUT_FOLDER = '../recs/cb-classification/'  # output folder' # output folder
OUTPUT_FILE_NAME = 'INSERT_ALGORITHM_NAME_HERE'  # output filename (NO NEED TO ADD .csv)
DESCR = False  # set this to true to use descr
TAGS_AND_GENRES = True  # set this to true to use genres and tags
MODE = Technique.LOG_REGR # edit here to change technique
#######################################################################################


NO_DESCR_TAG = '-no-descr'
DESCR_ONLY_TAG = '-descr-only'
NO_DESCR_TAG = ''
DESCR_ONLY_TAG = ''
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

if MODE == Technique.LOG_REGR:
    CLF = LogisticRegression(random_state=42)
elif MODE == Technique.RANDOM_FOREST:
    CLF = RandomForestClassifier(n_estimators=400, random_state=42)
elif MODE == Technique.SVM:
    CLF = CalibratedClassifierCV(LinearSVC(random_state=42))
elif MODE == Technique.KNN:
    CLF = neighbors.KNeighborsClassifier()
elif MODE == Technique.DECISION_TREE:
    CLF = DecisionTreeClassifier(random_state=42)
elif MODE == Technique.GAUSSIAN_PROCESS:
    CLF = GaussianProcessClassifier(random_state=42)


def pick_tag():
    if not DESCR:
        return NO_DESCR_TAG
    elif not TAGS_AND_GENRES:
        return DESCR_ONLY_TAG
    return ''


def get_item_field(book, field, data):
    maybe_field = data.query('item  == @book')
    if not maybe_field.empty:
        return maybe_field[[field]].values.flatten()[0]

    return ''


def build_items_content():
    items_content = {}
    for item in all_items:
        content = ''
        if TAGS_AND_GENRES:
            content += get_item_field(item, 'tags', items_tags)
           # content += get_movie_field(movie, 'genres', movies_genres)
        #if DESCR:
            #content += get_movie_field(movie, 'description', movies_descriptions)
        items_content[item] = content
    return items_content


def get_items_and_labels(ratings):
    items = list()
    labels = list()
    for ignored, item in ratings.iterrows():
        movie_id = item[['item']].values.flatten()[0]
        content = items_content.get(movie_id)

        if content != '':
            items.append(content)

            rating = item[['rating']].values.flatten()[0]
            labels.append(get_label(rating))

    return items, labels


def get_contents(items):
    items_ids = list()
    contents = list()
    for item in items:
        content = items_content.get(item)
        if content != '':
            items_ids.append(item)
            contents.append(content)
    return items_ids, contents


def get_label(rating):
    if rating >= MIN_POSITIVE_RATING:
        return POSITIVE_LABEL
    return NEGATIVE_LABEL


if not (TAGS_AND_GENRES or DESCR):
    raise Exception('At least one between TAGS_AND_GENRES and DESCRIPTIONS should be True')


ratings = pd.read_csv('../datasets/goodbooks-10k-master/ratings.csv')
users = list(set(ratings[['user']].values.flatten()))

all_items = set(pd.read_csv('../datasets/goodbooks-10k-master/books.csv').index.values.flatten())

items_tags = pd.read_csv('../datasets/books-tags.csv', encoding='latin-1')


items_content = build_items_content()

output_path = OUTPUT_FOLDER + OUTPUT_FILE_NAME + pick_tag() + '.csv'

# write csv doc header
f = open(output_path, 'w')
f.write('user,item,score\n')
f.close()

# accuracies = list()
for user in users:
    print(user)
    user_ratings = ratings.query('user == @user')
    rated_items = set(user_ratings[['item']].values.flatten())
    items, labels = get_items_and_labels(user_ratings)

    np_labels = np.array(labels)
    # skip users with positive or negative ratings only
    if np.all(np_labels == 0) or np.all(np_labels == 1):
        print('skipping user: ', user)
        continue

    pattern = "(?u)\\b[\\w-]+\\b"  # pattern for group-of-words-like-this

    if MODE == Technique.KNN:
        CLF.set_params(n_neighbors=round(math.sqrt(len(items))))

    pipeline = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english', token_pattern=pattern, sublinear_tf=True)),
        ('dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
        ('clf', CLF),
    ])

    try:
        pipeline.fit(items, labels)
    except Exception as err:
        print('Skipping user {} for: {}'.format(user, err))
        continue

    # fetching movies not rated by the user
    not_rated_items = list(all_items.difference(rated_items))
    random.shuffle(not_rated_items)
    new_items, new_items_contents = get_contents(not_rated_items)

    predictions = pipeline.predict(new_items_contents)
    predictions_proba = pipeline.predict_proba(new_items_contents)

    if predictions_proba.shape[1] == 2:
        # select positive class proba only
        positive_proba = predictions_proba[:, 1]
    else:
        positive_proba = predictions_proba.flatten()

    sorted_idx = np.argsort(positive_proba)[::-1][:NUM_OF_RECS]

    f = open(output_path, 'a')
    for i in sorted_idx:
        print(new_items[i], ':', predictions_proba[i])
        f.write('{},{},{}\n'.format(user, new_items[i], positive_proba[i]))
    f.close()
    print('------------------------------------------')
