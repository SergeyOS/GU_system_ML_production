from hyperopt import hp, tpe, space_eval
from hyperopt.fmin import fmin
from config.config_logs import *


@Log()
class XGBClassifierOptimize:
    space = {
        'num_boost_round': hp.quniform('num_boost_round', 50, 500, 1),
        'max_depth': hp.choice('max_depth', [5, 8, 10, 12, 15]),
        'min_child_weight': hp.uniform('min_child_weight', 0, 50),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'alpha': hp.uniform('alpha', 0, 1),
        'lambda': hp.uniform('lambda', 0, 1),
        'eta': hp.uniform('eta', 0.01, 1),
        'scale_pos_weight': hp.uniform('scale_pos_weight', 3, 15),
        'tree_method': 'gpu_hist',
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'seed': 169
    }

    def __init__(self, random_state=169, gpu_use='auto', propobility=0.5):
        self.space['tree_method'] = gpu_use
        self.random_state = random_state
        self.space['seed'] = random_state
        self.propobility = propobility
        self.dtrain = None

    def bin_class_from_proba(self, y_valid_pred_probs):
        result = y_valid_pred_probs.copy()
        result[result >= self.propobility] = 1
        result[result < self.propobility] = 0
        return result.astype(int)

    def f1_eval(self, y_pred, dtrain):
        y_true = dtrain.get_label()
        err = 1 - f1_score(y_true, self.bin_class_from_proba(y_pred), average='macro')
        return 'f1_err', err

    def objective(self, params):
        parameters = {
            'objective': params['objective'],
            'max_depth': int(params['max_depth']),
            'min_child_weight': params['min_child_weight'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'eta': params['eta'],
            'alpha': params['alpha'],
            'lambda': params['lambda'],
            'tree_method': params['tree_method'],
            'scale_pos_weight': params['scale_pos_weight'],
            'eval_metric': params['eval_metric'],
            'seed': params['seed']
        }

        cv_result = xgb.cv(parameters, num_boost_round=int(params['num_boost_round']), dtrain=self.dtrain, nfold=5,
                           seed=self.random_state, maximize=False, early_stopping_rounds=10, feval=self.f1_eval,
                           verbose_eval=True)
        score = cv_result['test-f1_err-mean'][-1:].values[0]
        return score

    def optimize(self, x_train, y_train):
        self.dtrain = xgb.DMatrix(data=x_train, label=np.ravel(y_train))
        best = fmin(fn=self.objective, space=self.space, algo=tpe.suggest, max_evals=10)
        best_params = space_eval(self.space, best)
        return best_params
