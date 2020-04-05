from config.config_logs import *
from config.config_dataset import *
import pandas as pd
from dataset.dataset_builder import DatasetBuilder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from eli5.sklearn import PermutationImportance
import xgboost as xgb
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

from model.optimize_parameter import XGBClassifierOptimize

sys.path.append('../')

RANDOM_STATE = 42


class PISelectorColumns(BaseEstimator, TransformerMixin):
    threshold = 0.0
    features = set()
    random_state = 42

    def __init__(self, threshold=0.0, random_state=42):
        self.threshold = threshold
        self.features = set()
        self.random_state = random_state

    def fit(self, X, y=None):
        self.features = set()
        baseline_model = xgb.XGBClassifier(max_depth=3,
                                           n_estimators=100,
                                           learning_rate=0.1,
                                           nthread=5,
                                           subsample=1.,
                                           colsample_bytree=0.5,
                                           min_child_weight=3,
                                           reg_alpha=0.,
                                           reg_lambda=0.,
                                           seed=self.random_state,
                                           missing=1e10)
        baseline_model.fit(X, y)
        PI = PermutationImportance(baseline_model, random_state=self.random_state).fit(X, y)
        for i in range(len(PI.feature_importances_)):
            if self.threshold < PI.feature_importances_[i]:
                self.features.add(i)
        print(f'Отобрано {len(self.features)} признаков')
        return self

    def transform(self, X):
        return X[:, list(self.features)]


@Log(NAME_LOGGER)
class MainModel:
    df: pd.DataFrame = None
    mode: str = ''
    random_state = 42
    target = 'is_churned'
    user_column = 'user_id'
    cat_feats = []
    numeric_features = []
    default_path = ''

    logger = logging.getLogger(NAME_LOGGER)
    x_train = None
    x_valid = None
    y_train = None
    y_valid = None

    best_model = None
    best_params = {
        'random_state': 42,
        'max_depth': 7
    }

    preprocessor = None
    select_feats = None
    gpu_use = 'auto'

    def __init__(self, random_state=RANDOM_STATE, default_path=DATASET_PATH, optimization_params=False, gpu_use='auto'):
        self.random_state = random_state
        self.default_path = default_path
        self.gpu_use = gpu_use
        self.best_params['seed'] = self.random_state
        self.best_params['tree_method'] = gpu_use
        self.optimization_params = optimization_params

    def read_csv(self):
        db = DatasetBuilder(mode=self.mode, dataset_path=self.default_path)
        self.df = pd.read_csv(db.get_path_dataset(), sep=';')
        logger.info(f'Read file {db.get_path_dataset()}')
        del db

    def train_split(self):
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.df.drop(columns=[self.user_column, self.target], axis=1), self.df[self.target],
            stratify=self.df[self.target], test_size=0.3, random_state=self.random_state)

    def find_best_params(self, x_train_balanced, y_train_balanced):
        if self.optimization_params:
            xgbopt = XGBClassifierOptimize(self.random_state)
            return xgbopt.optimize(x_train_balanced, y_train_balanced)
        else:
            return {'alpha': 0.026306405881918238,
                    'colsample_bytree': 0.6121004480140988,
                    'eta': 0.5333195656960023,
                    'lambda': 0.7358009709327051,
                    'max_depth': 3,
                    'min_child_weight': 8.090985286139347,
                    'num_boost_round': 143.0,
                    'scale_pos_weight': 10.192518853683392,
                    'subsample': 0.8721509146242798,
                    'seed': self.random_state,
                    'tree_method': self.gpu_use}

    def column_processor(self):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scale', MinMaxScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='M')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        return ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.cat_feats),
                ('num', numeric_transformer, self.numeric_features)
            ])

    def report(self):
        if self.df is None or self.x_train is None:
            return
        dfsets = [{'set': 'train', 'dmat': self.x_train, 'target': self.y_train},
                  {'set': 'valid', 'dmat': self.x_valid, 'target': self.y_valid}]
        self.logger.info(f'train size={self.x_train.shape} valid size={self.x_valid.shape}')
        for dfset in dfsets:
            self.logger.info(f'Metric values {dfset["set"]}:')
            y_pred = self.best_model.predict(dfset['dmat'])
            y_prob = self.best_model.predict_proba(dfset['dmat'])[:, 1]
            precision = precision_score(y_true=dfset['target'], y_pred=y_pred)
            recall = recall_score(y_true=dfset['target'], y_pred=y_pred)
            f1 = f1_score(y_true=dfset['target'], y_pred=y_pred)
            ll = log_loss(y_true=dfset['target'], y_pred=y_prob)
            roc_auc = roc_auc_score(y_true=dfset['target'], y_score=y_prob)
            self.logger.info('Precision: {}'.format(precision))
            self.logger.info('Recall: {}'.format(recall))
            self.logger.info('F1: {}'.format(f1))
            self.logger.info('Log Loss: {}'.format(ll))
            self.logger.info('ROC AUC: {}'.format(roc_auc))
            TN, FP, FN, TP = confusion_matrix(dfset['target'], y_pred).ravel()
            self.logger.debug(f'TN={TN} FN={FP}')
            self.logger.debug(f'TN={FN} FN={FP}')
            self.logger.info(confusion_matrix(dfset['target'], y_pred))
            self.logger.info(classification_report(dfset['target'], y_pred))

    def clear_temp(self):
        self.x_train = None
        self.x_valid = None
        self.y_train = None
        self.y_valid = None

    def train(self, load_model=False):
        self.mode = TRAIN_MODE
        if load_model:
            self.load_pipeline()
        else:
            self.logger.info(f'Start train')
            self.read_csv()
            self.init_numeric_features()
            self.logger.info(self.numeric_features)
            self.train_split()

            self.preprocessor = Pipeline(steps=[('column_processor', self.column_processor())])
            x_train_balanced = self.preprocessor.fit_transform(self.x_train)

            x_train_balanced, y_train_balanced = SMOTE(random_state=self.random_state,
                                                       sampling_strategy=0.3).fit_sample(x_train_balanced,
                                                                                         self.y_train)
            self.logger.info(f'Start fit SelectorColumns')
            self.select_feats = PISelectorColumns(threshold=0.0)
            x_train_balanced = self.select_feats.fit_transform(x_train_balanced, y_train_balanced)
            self.logger.info(f'End fit SelectorColumns')

            self.logger.info(f'Start fit final_classifier')
            self.best_params = self.find_best_params(x_train_balanced, y_train_balanced)
            final_classifier = xgb.XGBClassifier(**self.best_params)
            final_classifier.fit(x_train_balanced, y_train_balanced)
            self.logger.info(f'End fit final_classifier')

            self.best_model = Pipeline(steps=[('preprocessor', self.preprocessor),
                                              ('select_feats', self.select_feats),
                                              ('classifier', final_classifier)])
            self.logger.debug(self.x_train.shape)
            self.report()
            self.clear_temp()
            self.save_pipeline()
            self.logger.info(f'End train')
        return self

    def init_numeric_features(self):
        self.numeric_features = list(
            set(self.df) - set([self.user_column, self.target] + self.cat_feats))

    def predict(self, mode=TRAIN_MODE):
        self.mode = mode
        self.read_csv()
        if self.best_model is None:
            self.load_pipeline()
        self.init_numeric_features()
        return self.best_model.predict(self.df[self.numeric_features+self.cat_feats])


    def save_pipeline(self):
        pickle.dump(self.best_model, open(os.path.join(self.default_path, 'best_model.p'), 'wb'))

    def load_pipeline(self):
        self.best_model = pickle.load(open(os.path.join(self.default_path, 'best_model.p'), 'rb'))

    def predict_save(self, mode=TEST_MODE, path_to_save=DEFAULT_ANSWER_PATH):
        self.df[self.target] = self.predict(mode)
        self.logger.info(self.df.info())
        self.df[[self.user_column, self.target]].to_csv(os.path.join(path_to_save, f'answers_{mode}.csv'), index=False)


if __name__ == "__main__":
    model = MainModel()
    model.train()
