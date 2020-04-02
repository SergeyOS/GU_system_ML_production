import sys
import os
sys.path.append('../')
from argparse import ArgumentParser

CHURNED_START_DATE = '2019-09-01'
CHURNED_END_DATE = '2019-10-01'

INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]

DATASET_PATH = '../source/dataset/'

TRAIN_RAW_DATA_PATH = '../source/train/'
TRAIN_MODE = 'train'

TEST_RAW_DATA_PATH = '../source/test/'
TEST_MODE = 'test'


# Анализ параметров коммандной строки
def parse_args_console(argv):
    """
    Обработка параметров командой строки
    :param argv: параметры командной строки или None
    :return: моde и путь для сохранения
    """
    if argv is None:
        return None

    parser = ArgumentParser()
    parser.add_argument(
        '-m', '--mode', type=str, required=True, default=None
    )
    parser.add_argument(
        '-dp', '--dest_path', type=str, required=True, default=''
    )
    parser.add_argument(
        '-sp', '--source_path', type=str, required=True, default=''
    )
    args = parser.parse_args(argv)
    if not(args.mode is None or args.mode == TRAIN_MODE or args.mode == TEST_MODE):
        raise ValueError
    if (args.source_path is None or not(os.path.exists(args.source_path))):
        raise ValueError
    return args.mode, args.dest_path, args.source_path
