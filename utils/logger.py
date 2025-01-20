# -*- coding: utf-8 -*-
# @Time    : 2023年8月9日10:18:28
# @Author  : cy
import logging
import utils.config as C
import os


class LoggerFactory:
    def __init__(self, log_type):
        self.log_type = log_type
        assert self.log_type in {"data_process", }, "log_type must be in ['data_process', ]"

    def get_logger(self):
        assert self.log_type in {"data_process", }, "log_type must be in ['data_process', ]"
        if self.log_type == "data_process":
            return ProcessLogger("data_process")
        elif self.log_type == "console":
            return ConsoleLogger()
        else:
            raise ValueError("Unsupported log type")


class Logger:
    def log(self, message):
        pass


class ProcessLogger:
    _instance = None

    def __new__(cls, log_type):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logging.basicConfig(level=logging.DEBUG)
            cls._instance.logger = logging.getLogger(f"{log_type}_logger")

            # 创建一个文件处理程序，将日志写入文件
            path = os.path.join(C.log, f'{log_type}.log')
            cls._instance.file_handler = logging.FileHandler(path, mode='w')
            # cls._instance.logger.addHandler(cls._instance.file_handler)

            # 创建一个控制台处理程序，将日志打印到控制台
            cls._instance.stream_handler = logging.StreamHandler()
            # cls._instance.logger.addHandler(cls._instance.stream_handler)

        return cls._instance

    def log(self, message):
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stream_handler)
        self.logger.info(message)

    def log_to_console(self, message):
        self.logger.removeHandler(self.file_handler)
        self.logger.addHandler(self.stream_handler)
        self.logger.info(message)
        # self.logger.addHandler(self.file_handler)

    def log_to_file(self, message):
        self.logger.removeHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)
        self.logger.info(message)
        # self.logger.addHandler(self.stream_handler)


class ConsoleLogger(Logger):
    def log(self, message):
        logging.basicConfig(level=logging.INFO)
        logging.info(message)
