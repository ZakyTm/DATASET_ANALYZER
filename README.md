# Dataset Analyzer Pro

Advanced data analysis tool with interactive visualization capabilities and machine learning integration.

## Features

### Data Handling
- Support for multiple file formats: CSV, Excel, JSON, Parquet, Feather, HDF5, Pickle, SQLite, and zipped files
- Automatic encoding detection
- Advanced error handling and logging
- Interactive drag-and-drop file loading

### Data Analysis
- Automated Exploratory Data Analysis (EDA) reports
- Advanced statistical analysis (correlation analysis, normality tests, outlier detection)
- Interactive data filtering and transformation
- Data quality assessment

### Visualization
- Interactive Plotly visualizations
- Comprehensive correlation matrix analysis
- Box plots, histograms, scatter plots, and more
- Customizable visualization options

### Machine Learning
- Automated machine learning model training
- Support for regression and classification tasks
- Hyperparameter optimization
- Model evaluation and cross-validation
- Feature importance analysis
- Model saving and loading

### Export & Reporting
- PDF report generation with customizable templates
- Interactive HTML reports
- Excel/CSV/JSON export options
- Comprehensive visualization export

### Development
- Well-structured MVC architecture
- Comprehensive logging system
- Modular design for easy extension
- CI/CD integration with GitHub Actions
- Comprehensive testing suite

## Installation

```bash
git clone https://github.com/yourusername/dataset-analyzer.git
cd dataset-analyzer
pip install -r requirements.txt
```

## Quick Start

```bash
python main.py
```

## Usage Examples

### Basic Data Analysis
1. Load your dataset using the "Browse Files" button
2. Click "Analyze Data" to generate summary statistics
3. Explore the various tabs to see data insights

### Machine Learning
1. Load your dataset
2. Go to the "Machine Learning" tab
3. Select a target column
4. Choose a model type (or use auto-detection)
5. Train and evaluate your model

### Custom Reports
1. Load and analyze your dataset
2. Click "Generate Report" in the Export menu
3. Select your preferred format (PDF, HTML)
4. View or share the generated report

## Dependencies

Key dependencies include:
- pandas (data manipulation)
- matplotlib/seaborn/plotly (visualization)
- scikit-learn (machine learning)
- ydata-profiling (EDA)
- reportlab/jinja2 (reporting)

## Project Structure

```
dataset-analyzer/
├── core/                   # Core functionality modules
│   ├── analyzer.py         # Data analysis engine
│   ├── file_handler.py     # File operations
│   ├── filter.py           # Data filtering
│   ├── ml_modeling.py      # Machine learning capabilities
│   ├── preprocessor.py     # Data preprocessing
│   ├── reporter.py         # Reporting engine
│   └── statistical.py      # Statistical analysis
├── utils/                  # Utility modules
│   ├── constants.py        # Application constants
│   ├── helpers.py          # Helper functions
│   └── templates/          # Report templates
├── tests/                  # Test suite
├── logs/                   # Application logs
├── reports/                # Generated reports
├── models/                 # Saved ML models
├── main.py                 # Application entry point
└── requirements.txt        # Dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.