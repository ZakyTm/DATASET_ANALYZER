import pytest
import pandas as pd
import numpy as np
from core.analyzer import DataAnalyzer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'numeric': [1, 2, 3, 4, None, 6],
        'categorical': ['A', 'B', 'A', None, 'C', 'B'],
        'constant': [1, 1, 1, 1, 1, 1]
    })

def test_initialization():
    analyzer = DataAnalyzer()
    assert analyzer.data is None

def test_identify_issues(sample_data):
    analyzer = DataAnalyzer()
    analyzer.set_data(sample_data)
    issues = analyzer.identify_issues()
    
    assert 'constant' in issues['constant_features']
    assert 'numeric' in issues['high_missing']
    assert issues['duplicate_rows'] >= 0

def test_correlation_calculation(sample_data):
    analyzer = DataAnalyzer()
    analyzer.set_data(sample_data.fillna(0))
    corr = analyzer.calculate_correlations()
    
    assert not corr.empty
    assert 'numeric' in corr.columns