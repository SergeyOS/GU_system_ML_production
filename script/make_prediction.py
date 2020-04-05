from model.main_model import MainModel, PISelectorColumns
from config.config_dataset import *

if __name__ == "__main__":
    mm = MainModel(random_state=142,  optimization_params=False)
    mm.predict_save(TEST_MODE)
