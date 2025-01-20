#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023年8月9日10:50:48
# @Author  : cy
import os

import numpy as np

np.random.seed(0)
# 资源路径
abs_path_pro = os.path.abspath(__file__).replace('\\', '/').split('/')[:-2]
abs_path_pro = "/".join(abs_path_pro) + "/"  # 项目绝对路径

log = abs_path_pro + "log/"  # 日志保存路径

