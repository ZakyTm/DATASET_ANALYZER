import pandas as pd
import numpy as np

class DataFilter:
    def __init__(self, data=None):
        self.data = data
        self.filters = []

    def add_filter(self, column, operator, value):
        self.filters.append((column, operator, value))
        return self

    def apply_filters(self):
        if self.data is None:
            return None
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

    def apply_filter(self, data, column, operator, value):
        """Apply a single filter to data and return filtered data"""
        filtered_data = data.copy()
        if operator == '>':
            filtered_data = filtered_data[filtered_data[column] > float(value)]
        elif operator == '<':
            filtered_data = filtered_data[filtered_data[column] < float(value)]
        elif operator == '==':
            filtered_data = filtered_data[filtered_data[column] == value]
        elif operator == '!=':
            filtered_data = filtered_data[filtered_data[column] != value]
        elif operator == 'contains':
            filtered_data = filtered_data[filtered_data[column].astype(str).str.contains(value, na=False)]
        elif operator == 'starts with':
            filtered_data = filtered_data[filtered_data[column].astype(str).str.startswith(value, na=False)]
        elif operator == 'ends with':
            filtered_data = filtered_data[filtered_data[column].astype(str).str.endswith(value, na=False)]
        elif operator == 'between':
            try:
                lower, upper = [float(x.strip()) for x in value.split(',')]
                filtered_data = filtered_data[filtered_data[column].between(lower, upper)]
            except ValueError:
                # If parsing fails, return unfiltered data
                return filtered_data
        return filtered_data

    def clear_filters(self):
        self.filters = []
        return self.data.copy() if self.data is not None else None