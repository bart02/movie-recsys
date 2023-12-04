import pickle
from pathlib import Path

from rectools import Columns
from rectools.metrics import MAP, calc_metrics, MeanInvUserFreq, Serendipity
import utils

from rectools.dataset import Dataset

p = Path(__file__).parent
root = p.parent

# Load data
df_train, user_features_train, item_features_train, df_test = utils.read_split_rating_dataset_with_features(
    root / 'data/interim/rating.csv', root / 'data/interim/user.csv', root / 'data/interim/movie.csv')

dataset = Dataset.construct(
    df_train,
    user_features_df=user_features_train,  # our flatten dataframe
    item_features_df=item_features_train,  # our flatten dataframe
    cat_user_features=["gender", "occupation"],  # these will be one-hot-encoded
    make_dense_user_features=False,  # for `sparse` format
    make_dense_item_features=False,  # for `sparse` format
)

# Load model
with open(root / 'models' / 'als_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define metrics
metrics_name = {
    'MAP': MAP,
    'MIUF': MeanInvUserFreq,
    'Serendipity': Serendipity
}
metrics = {}
for metric_name, metric in metrics_name.items():
    for k in (1, 5, 10):
        metrics[f'{metric_name}@{k}'] = metric(k=k)

# Generate recommendations
recos = model.recommend(
    users=df_train[Columns.User].unique(),
    dataset=dataset,
    k=10,
    filter_viewed=True,
)

# Calculate metrics
catalog = df_train[Columns.Item].unique()
print(calc_metrics(
    metrics,
    reco=recos,
    interactions=df_test,
    prev_interactions=df_train,
    catalog=catalog
))
