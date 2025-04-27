"""
Constants for the dataset analyzer application
"""

# File extensions
SUPPORTED_EXTENSIONS = ('.csv', '.xlsx', '.xls')

# Analysis settings
DEFAULT_CORRELATION_METHOD = 'pearson'
CORRELATION_METHODS = ['pearson', 'spearman', 'kendall']
MISSING_DATA_THRESHOLD = 0.5
LOW_VARIANCE_THRESHOLD = 0.01

# Plot types
PLOT_TYPES = {
    'histogram': 'Histogram',
    'box': 'Box Plot',
    'scatter': 'Scatter Plot',
    'bar': 'Bar Chart',
    'line': 'Line Chart',
    'heatmap': 'Heatmap',
    'pie': 'Pie Chart',
    'violin': 'Violin Plot',
    'kde': 'KDE Plot'
}

# Data Types
NUMERIC_DTYPES = ['int64', 'float64', 'int32', 'float32']
CATEGORICAL_DTYPES = ['object', 'category', 'bool']
DATETIME_DTYPES = ['datetime64', 'datetime64[ns]']

# UI settings
UI_PADDING = 10
UI_BUTTON_WIDTH = 15
DEFAULT_FIGURE_SIZE = (10, 6)

# Export formats
EXPORT_FORMATS = {
    'pdf': 'PDF Report',
    'html': 'HTML Report',
    'csv': 'CSV Export',
    'excel': 'Excel Export',
    'json': 'JSON Export'
}

# Default paths
DEFAULT_EXPORT_PATH = 'reports'
DEFAULT_LOGS_PATH = 'logs'
