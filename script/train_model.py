from model.main_model import MainModel

if __name__ == "__main__":
    mm = MainModel(random_state=142,  optimization_params=True)
    mm.train(load_model=False)
