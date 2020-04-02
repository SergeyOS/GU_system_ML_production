import sys
import logging.handlers
import os

sys.path.append('../')


LOGGING_LEVEL = logging.INFO
NAME_LOGGER = 'GU_CHURNED'
LOGS_PATH = '../logs/'

# создаём формировщик логов (formatter):
server_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s %(message)s')

# Подготовка имени файла для логирования
path = os.path.dirname(LOGS_PATH)
if not os.path.exists(path):
    os.mkdir(path)
path = os.path.join(path, f'{NAME_LOGGER}.log')

# создаём потоки вывода логов
steam = logging.StreamHandler(sys.stderr)
steam.setFormatter(server_formatter)
steam2 = logging.StreamHandler(sys.stdout)
steam.setFormatter(server_formatter)
steam.setLevel(logging.ERROR)
log_file = logging.handlers.TimedRotatingFileHandler(path, encoding='utf8', interval=1, when='D')
log_file.setFormatter(server_formatter)

# создаём регистратор и настраиваем его
logger = logging.getLogger(NAME_LOGGER)
logger.addHandler(steam)
logger.addHandler(steam2)
logger.addHandler(log_file)
logger.setLevel(LOGGING_LEVEL)


class Log:
    __logger: logger = None

    def __init__(self, namelogger: str = NAME_LOGGER):
        self.__logger = logging.getLogger(namelogger)

    def __call__(self, cls):
        callable_attributes = {k: v for k, v in cls.__dict__.items()
                               if callable(v)}
        # Decorate each callable attribute of to the input class
        for name, func in callable_attributes.items():
            decorated = self.decorate_method(func)
            setattr(cls, name, decorated)
        return cls

    def decorate_method(self, func_to_log):
        def decorated(*args, **kwargs):
            self.__logger.debug(
                f"Была вызвана функция {func_to_log.__name__} c параметрами {args} , {kwargs}. Вызов из модуля {func_to_log.__module__}")
            ret = func_to_log(*args, **kwargs)
            return ret

        return decorated


# отладка
if __name__ == '__main__':
    logger.critical('Test critical event')
    logger.error('Test error event')
    logger.debug('Test debug event')
    logger.info('Test info event')
