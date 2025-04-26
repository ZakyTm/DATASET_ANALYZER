import pandas as pd
import numpy as np

class DataFilter:
    def __init__(self, data):
        self.data = data
        self.filters = []

    def add_filter(self, column, operator, value):
        self.filters.append((column, operator, value))
        return self

    def apply_filters(self):
        filtered_data = self.data.copy()
        for column, operator, value in self.filters:
            if operator == '>':
                filtered_data = filtered_data[filtered_data[column] > value]
            elif operator == '<':
                filtered_data = filtered_data[filtered_data[column] < value]
            elif operator == '==':
                filtered_data = filtered_data[filtered_data[column] == value]
            elif operator == 'contains':
                filtered_data = filtered_data[filtered_data[column].str.contains(value, na=False)]
            elif operator == 'between':
                filtered_data = filtered_data[filtered_data[column].between(*value)]
        return filtered_data.reset_index(drop=True)

    def clear_filters(self):
        self.filters = []
        return self.data.copy()