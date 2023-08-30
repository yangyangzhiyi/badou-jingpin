#!/usr/bin/env python3
# encoding: utf-8
"""
加载数据
"""

from config import Config
import csv


with open(Config['train_data_path'], "r", encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # 跳过第一行标题
    data = tuple()
    for row in csv_reader:
        print(row)
        data[row(0)] = row[1]

# class DataGenerator:
#     def __init__(self, config):
#         self.config = config
#         self.train_data = self.load_data(config.train_data_path)
#
#     def load_data(self, data_path):
#         data = []
