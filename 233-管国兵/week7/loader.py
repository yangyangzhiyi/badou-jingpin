#!/usr/bin/env python3
# encoding: utf-8
"""
加载数据
"""
import random

from config import Config
import csv


def load_data(train_data_path):
    data = []
    with open(train_data_path, "r", encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # 跳过第一行标题
        for row in csv_reader:
            data.append(row)
    return data


def random_choice_data(data, num):
    random_data = []
    for i in range(num):
        random_data.append(random.choice(data))
    return random_data


data = load_data(Config['train_data_path'])
random_data = random_choice_data(data, 1000)
print(random_data)
