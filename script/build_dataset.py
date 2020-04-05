from dataset.dataset_builder import *
from argparse import ArgumentParser


if __name__ == "__main__":
    if len(sys.argv[1:]) == 0:
        builder = DatasetBuilder(raw_data_path=TEST_RAW_DATA_PATH, mode=TEST_MODE)
        builder.save_dataset()
    else:
        parser = ArgumentParser()
        parser.add_argument(
            '-m', '--mode', type=str, required=False, default=TRAIN_MODE
        )
        args = parser.parse_args(argv)
        if args.mode == TRAIN_MODE:
            builder = DatasetBuilder()
            builder.save_dataset()
        else:
            builder = DatasetBuilder(raw_data_path=TEST_RAW_DATA_PATH, mode=TEST_MODE)
            builder.save_dataset()
