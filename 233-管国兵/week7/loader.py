#!/usr/bin/env python3
# encoding: utf-8
"""
加载数据
"""
import csv
import random


class DataLoader(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = load_data(data_path)

    def get_data(self):
        return self.data

    def get_data_size(self):
        return len(self.data)

    def get_random_data(self, n):
        return random_choice_data(self.data, n)


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
