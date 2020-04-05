from config.config_dataset import *
from config.config_logs import *
from datetime import datetime
import timeit
import pandas as pd
import os
import sys

sys.path.append('../')


@Log(NAME_LOGGER)
class DatasetBuilder:
    churned_start_date: str = None
    churned_end_date: str = None
    dataset_path: str = None
    raw_data_path: str = None
    inter_list: list = None
    logger = logging.getLogger(NAME_LOGGER)

    def __init__(self, churned_start_date: str = CHURNED_START_DATE,
                 churned_end_date: str = CHURNED_END_DATE,
                 inter_list: list = INTER_LIST,
                 raw_data_path: str = TRAIN_RAW_DATA_PATH,
                 dataset_path: str = DATASET_PATH,
                 mode: str = TRAIN_MODE,
                 argv=None) -> None:
        """

        :param argv:
        :param churned_start_date:
        :param churned_end_date:
        :param inter_list:
        :param raw_data_path:
        :param dataset_path:
        :param mode:
        """

        if argv is not None and len(argv) != 0:
            self.mode, self.dataset_path, self.raw_data_path = parse_args_console(argv)
            if self.dataset_path == '':
                self.dataset_path = dataset_path
            if self.raw_data_path == '':
                self.raw_data_path = raw_data_path
        else:
            self.mode = mode
            self.dataset_path = dataset_path
            self.raw_data_path = raw_data_path

        self.churned_start_date = churned_start_date
        self.churned_end_date = churned_end_date
        self.inter_list = inter_list

    def get_dataset(self) -> pd.DataFrame:
        start = timeit.default_timer()
        self.logger.info('Start reading csv files')

        sample = pd.read_csv('{}sample.csv'.format(self.raw_data_path), sep=';', na_values=['\\N', 'None'],
                             encoding='utf-8')
        profiles = pd.read_csv('{}profiles.csv'.format(self.raw_data_path), sep=';', na_values=['\\N', 'None'],
                               encoding='utf-8')
        payments = pd.read_csv('{}payments.csv'.format(self.raw_data_path), sep=';', na_values=['\\N', 'None'],
                               encoding='utf-8')
        reports = pd.read_csv('{}reports.csv'.format(self.raw_data_path), sep=';', na_values=['\\N', 'None'],
                              encoding='utf-8')
        abusers = pd.read_csv('{}abusers.csv'.format(self.raw_data_path), sep=';', na_values=['\\N', 'None'],
                              encoding='utf-8')
        logins = pd.read_csv('{}logins.csv'.format(self.raw_data_path), sep=';', na_values=['\\N', 'None'],
                             encoding='utf-8')
        pings = pd.read_csv('{}pings.csv'.format(self.raw_data_path), sep=';', na_values=['\\N', 'None'],
                            encoding='utf-8')
        sessions = pd.read_csv('{}sessions.csv'.format(self.raw_data_path), sep=';', na_values=['\\N', 'None'],
                               encoding='utf-8')
        shop = pd.read_csv('{}shop.csv'.format(self.raw_data_path), sep=';', na_values=['\\N', 'None'],
                           encoding='utf-8')

        self.logger.info('Run time (reading csv files)')
        # -----------------------------------------------------------------------------------------------------
        self.logger.info('NO dealing with outliers, missing values and categorical features...')
        # -----------------------------------------------------------------------------------------------------
        # На основании дня отвала (last_login_dt) строим признаки, которые описывают активность игрока перед уходом

        self.logger.info('Creating dataset...')
        # Создадим пустой датасет - в зависимости от режима построения датасета - train или test
        if self.mode == TRAIN_MODE:
            dataset = sample.copy()[['user_id', 'is_churned', 'level', 'donate_total']]
        elif self.mode == TEST_MODE:
            dataset = sample.copy()[['user_id', 'level', 'donate_total']]

        # Пройдемся по всем источникам, содержащим "динамичекие" данные
        for df in [payments, reports, abusers, logins, pings, sessions, shop]:

            # Получим 'day_num_before_churn' для каждого из значений в источнике для определения недели
            data = pd.merge(sample[['user_id', 'login_last_dt']], df, on='user_id')
            data['day_num_before_churn'] = 1 + (
                    data['login_last_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) -
                    data['log_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))).apply(lambda x: x.days)
            df_features = data[['user_id']].drop_duplicates().reset_index(drop=True)

            # Для каждого признака создадим признаки для каждого из времененно интервала (в нашем примере 4 интервала
            # по 7 дней)
            features = list(set(data.columns) - {'user_id', 'login_last_dt', 'log_dt', 'day_num_before_churn'})
            self.logger.info(f'Processing with features:{features}')
            for feature in features:
                for i, inter in enumerate(self.inter_list):
                    inter_df = data.loc[data['day_num_before_churn'].between(inter[0], inter[1], inclusive=True)]. \
                        groupby('user_id')[feature].mean().reset_index(). \
                        rename(index=str, columns={feature: feature + '_{}'.format(i + 1)})
                    df_features = pd.merge(df_features, inter_df, how='left', on='user_id')

            # Добавляем построенные признаки в датасет
            dataset = pd.merge(dataset, df_features, how='left', on='user_id')
            del df_features
            self.logger.info(f'Run time (calculating features):{timeit.default_timer() - start}')

        del payments, reports, abusers, logins, pings, sessions, shop
        # Добавляем "статические" признаки
        dataset = pd.merge(dataset, profiles, on='user_id')
        del profiles
        return dataset

    def prepare_dataset(self, dataset):
        start = timeit.default_timer()
        self.logger.info('Dealing with missing values, outliers, categorical features...')

        # Профили
        dataset['age'] = dataset['age'].fillna(dataset['age'].median())
        dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
        dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
        dataset['gender'] = dataset['gender'].map({'M': 1., 'F': 0.})
        dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
        dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1
        # Пинги
        for period in range(1, len(INTER_LIST) + 1):
            col = 'avg_min_ping_{}'.format(period)
            dataset.loc[(dataset[col] < 0) |
                        (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()
        # Сессии и прочее
        dataset.fillna(0, inplace=True)
        return dataset

    def save_dataset(self) -> str:
        dataset = self.prepare_dataset(self.get_dataset())
        dataset.to_csv(self.get_path_dataset(), sep=';', index=False)
        self.logger.info(f'Dataset is successfully built and saved to {self.get_path_dataset()}.')
        return self.get_path_dataset()

    def get_path_dataset(self):
        return os.path.join(self.dataset_path, f'dataset_raw_{self.mode}.csv')


if __name__ == "__main__":
    builder = DatasetBuilder(raw_data_path=TEST_RAW_DATA_PATH, mode=TEST_MODE, argv=sys.argv[1:])
    #builder = DatasetBuilder(argv=sys.argv[1:])
    builder.save_dataset()
