from matplotlib.backend_bases import NavigationToolbar2
import pandas as pd
import seaborn as sns
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox
from tkinter import scrolledtext
from core.analyzer import DataAnalyzer
from core.file_handler import FileHandler
from core.filter import DataFilter
from core.reporter import ReportGenerator
from core.statistical import StatisticalAnalyzer
from core.ml_modeling import MLModel, ModelingTask
from utils.helpers import validate_file, setup_logger, get_logger
from utils.constants import PLOT_TYPES, CORRELATION_METHODS, EXPORT_FORMATS
from tkinterdnd2 import DND_FILES
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import warnings
import traceback
from pathlib import Path
warnings.filterwarnings("ignore", message="Upgrade to ydata-sdk")

# Set up logging
logger = setup_logger()

class DataModel:
    """Model component that handles data operations"""
    
    def __init__(self):
        self.df = None
        self.file_path = None
        self.file_handler = FileHandler()
        self.analyzer = None
        self.stat_analyzer = None
        self.ml_model = None
        self.reporter = None
        self.filter = DataFilter()
    
    def load_data(self, file_path):
        """Load data from file path using the enhanced file handler"""
        try:
            logger.info(f"Loading data from {file_path}")
            self.file_path = file_path
            self.df = self.file_handler.load_data(file_path)
            
            # Initialize analyzers with the loaded data
            self.analyzer = DataAnalyzer()
            self.analyzer.set_data(self.df)
            
            self.stat_analyzer = StatisticalAnalyzer()
            self.stat_analyzer.set_data(self.df)
            
            self.ml_model = MLModel()
            self.ml_model.set_data(self.df)
            
            self.reporter = ReportGenerator(self.analyzer)
            
            # Reset filter
            self.filter = DataFilter()
            
            logger.info(f"Successfully loaded data with shape {self.df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def get_data_info(self):
        """Return information about the dataset"""
        if self.df is not None:
            info = {
                'shape': self.df.shape,
                'columns': self.df.columns.tolist(),
                'dtypes': self.df.dtypes.to_dict(),
                'missing_values': self.df.isnull().sum().to_dict(),
                'memory_usage': self.df.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
                'file_path': self.file_path
            }
            return info
        return None
    
    def get_data_preview(self, rows=5):
        """Return preview of the data"""
        if self.df is not None:
            return self.df.head(rows)
        return None
    
    def get_correlations(self, method='pearson'):
        """Return correlation matrix using specified method"""
        if self.df is not None and self.analyzer is not None:
            return self.analyzer.calculate_correlations(method=method)
        return pd.DataFrame()
    
    def get_metrics(self):
        """Return basic statistics"""
        if self.df is not None and self.stat_analyzer is not None:
            return self.stat_analyzer.get_summary_statistics()
        return pd.DataFrame()
    
    def apply_filter(self, column, operator, value):
        """Apply data filter"""
        if self.df is not None:
            try:
                logger.info(f"Applying filter: {column} {operator} {value}")
                self.df = self.filter.apply_filter(self.df, column, operator, value)
                
                # Update analyzers with filtered data
                self.analyzer.set_data(self.df)
                self.stat_analyzer.set_data(self.df)
                self.ml_model.set_data(self.df)
                self.reporter = ReportGenerator(self.analyzer)
                
                logger.info(f"Data filtered, new shape: {self.df.shape}")
                return True
            except Exception as e:
                logger.error(f"Error applying filter: {str(e)}")
                return False
        return False
    
    def reset_filters(self):
        """Reset all filters and reload the original data"""
        if self.file_path:
            return self.load_data(self.file_path)
        return False
    
    def train_predictive_model(self, target_column, model_type='auto', test_size=0.25):
        """Train a predictive model on the data"""
        if self.df is not None and self.ml_model is not None:
            try:
                # Prepare data for modeling
                logger.info(f"Preparing data for modeling with target: {target_column}")
                success = self.ml_model.prepare_data(target_column, test_size=test_size)
                if not success:
                    return False
                
                # Train model
                logger.info(f"Training model of type: {model_type}")
                success = self.ml_model.train_model(model_name=model_type)
                if not success:
                    return False
                
                # Return model results
                return self.ml_model.results
            except Exception as e:
                logger.error(f"Error training model: {str(e)}")
                logger.error(traceback.format_exc())
                return False
        return False
    
    def optimize_model(self):
        """Optimize hyperparameters for the current model"""
        if self.ml_model and self.ml_model.pipeline:
            try:
                logger.info("Optimizing model hyperparameters")
                return self.ml_model.optimize_hyperparameters()
            except Exception as e:
                logger.error(f"Error optimizing model: {str(e)}")
                return None
        return None
    
    def cross_validate_model(self, cv=5):
        """Cross-validate the current model"""
        if self.ml_model and self.ml_model.pipeline:
            try:
                logger.info(f"Cross-validating model with {cv} folds")
                return self.ml_model.cross_validate(cv=cv)
            except Exception as e:
                logger.error(f"Error cross-validating model: {str(e)}")
                return None
        return None
    
    def save_model(self, filename=None):
        """Save the current model to disk"""
        if self.ml_model and self.ml_model.pipeline:
            try:
                logger.info(f"Saving model with filename: {filename}")
                return self.ml_model.save_model(filename)
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                return None
        return None
    
    def export_data(self, format_type='csv', include_index=False):
        """Export data to specified format"""
        if self.df is not None:
            try:
                logger.info(f"Exporting data to {format_type} format")
                if format_type == 'csv':
                    return self.file_handler.export_csv(self.df, include_index)
                elif format_type == 'excel':
                    return self.file_handler.export_excel(self.df, include_index)
                elif format_type == 'json':
                    return self.file_handler.export_json(self.df)
                elif format_type == 'pdf':
                    # Create a report dict for PDF generation
                    report_data = {
                        'title': 'Data Analysis Report',
                        'info': self.get_data_info(),
                        'statistics': self.get_metrics()
                    }
                    return self.file_handler.export_pdf(report_data)
                elif format_type == 'html':
                    # Create a report dict for HTML generation
                    report_data = {
                        'title': 'Data Analysis Report',
                        'rows': self.df.shape[0],
                        'columns': self.df.shape[1],
                        'stats': self.get_metrics(),
                        'description': f"Analysis of {os.path.basename(self.file_path)}"
                    }
                    return self.file_handler.export_html(report_data)
                else:
                    logger.error(f"Unsupported export format: {format_type}")
                    return None
            except Exception as e:
                logger.error(f"Error exporting data: {str(e)}")
                logger.error(traceback.format_exc())
                return None
        return None

    def generate_report(self):
        """Generate a comprehensive report"""
        if self.df is not None and self.analyzer is not None:
            try:
                logger.info("Generating comprehensive report")
                # Generate profile report
                report = self.analyzer.generate_profile()
                report_path = self.file_handler.save_report(report)
                
                # Generate plots
                try:
                    plots = self.reporter.generate_interactive_plots()
                except Exception as plot_e:
                    logger.error(f"Error generating plots: {str(plot_e)}")
                    plots = []
                
                # Generate correlation matrix
                try:
                    corr_plot = self.reporter.generate_correlation_matrix()
                except Exception as corr_e:
                    logger.error(f"Error generating correlation matrix: {str(corr_e)}")
                    corr_plot = None
                
                return {
                    'report_path': report_path,
                    'plots': plots,
                    'correlation_matrix': corr_plot
                }
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                logger.error(traceback.format_exc())
                return None
        return None

class DataVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Analyzer Pro")
        self.root.geometry("1200x800")
        
        # Initialize data
        self.df = None
        self.file_path = None
        
        # Create model first
        self.model = DataModel()
        
        # Create controller with reference to self and model
        self.controller = DataController(self, self.model)
        
        # Only now set up UI components
        self._setup_ui()
    
    def browse_files(self):
        """Delegate to controller's browse_files method"""
        return self.controller.browse_files()
    
    def analyze_data(self):
        """Delegate to controller's analyze_data method"""
        if not self.controller.analyze_data():
            messagebox.showerror("Error", "Please select a valid data file first.")
    
    def display_correlations(self, corr_matrix):
        def _update_plot():
            self.ax.clear()
            sns.heatmap(corr_matrix, annot=True, ax=self.ax)
            self.canvas.draw()
        
        # Schedule the update on the main thread
        self.root.after(0, _update_plot)
    
    def _setup_bindings(self):
        self.root.bind('<Control-o>', lambda e: self.controller.browse_files())
        self.root.bind('<Control-q>', lambda e: self.root.destroy())
        
        # Drag-and-drop if available
        if hasattr(self, 'drop_frame'):
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)
    
    def _setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X)
        
        # Use self.browse_files instead of self.controller.browse_files
        ttk.Button(btn_frame, text="Browse Files", command=self.browse_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Analyze Data", command=self.analyze_data).pack(side=tk.LEFT, padx=5)
        
        self.file_path_var = tk.StringVar()
        ttk.Label(file_frame, textvariable=self.file_path_var).pack(fill=tk.X, pady=5)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Data info tab
        info_frame = ttk.Frame(notebook, padding="10")
        notebook.add(info_frame, text="Data Info")
        self.data_info_text = tk.Text(info_frame, wrap=tk.WORD, height=10)
        self.data_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Data preview tab
        preview_frame = ttk.Frame(notebook, padding="10")
        notebook.add(preview_frame, text="Data Preview")
        self.data_preview_text = tk.Text(preview_frame, wrap=tk.WORD, height=10)
        self.data_preview_text.pack(fill=tk.BOTH, expand=True)
        
        # Correlations tab
        corr_frame = ttk.Frame(notebook, padding="10")
        notebook.add(corr_frame, text="Correlations")
        self.correlations_text = tk.Text(corr_frame, wrap=tk.WORD, height=10)
        self.correlations_text.pack(fill=tk.BOTH, expand=True)
        
        # Metrics tab
        metrics_frame = ttk.Frame(notebook, padding="10")
        notebook.add(metrics_frame, text="Metrics")
        self.metrics_text = tk.Text(metrics_frame, wrap=tk.WORD, height=10)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization tab
        viz_frame = ttk.Frame(notebook, padding="10")
        notebook.add(viz_frame, text="Visualization")
        
        # Visualization controls
        viz_controls = ttk.Frame(viz_frame)
        viz_controls.pack(fill=tk.X, pady=5)
        
        ttk.Label(viz_controls, text="Plot Type:").pack(side=tk.LEFT, padx=5)
        self.plot_type = tk.StringVar(value="histogram")
        plot_dropdown = ttk.Combobox(viz_controls, textvariable=self.plot_type)
        plot_dropdown['values'] = tuple(PLOT_TYPES.keys())
        plot_dropdown.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(viz_controls, text="Generate Plot", command=self.visualize_data).pack(side=tk.LEFT, padx=5)
        
        # Matplotlib figure and canvas
        self.figure = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add Machine Learning tab
        ml_frame = ttk.Frame(notebook, padding="10")
        notebook.add(ml_frame, text="Machine Learning")
        self._setup_ml_tab(ml_frame)
        
        # Add Export options to menu bar
        self._setup_menu_bar()
        
        # Add additional filter frame
        self._setup_filter_frame(main_frame)
    
    def _setup_ml_tab(self, parent):
        """Set up the machine learning tab UI"""
        # Main container
        ml_content = ttk.Frame(parent)
        ml_content.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for settings
        left_panel = ttk.Frame(ml_content, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Model configuration
        model_frame = ttk.LabelFrame(left_panel, text="Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        # Target column selection
        ttk.Label(model_frame, text="Target Column:").pack(anchor='w', pady=2)
        self.target_column = ttk.Combobox(model_frame, state='readonly')
        self.target_column.pack(fill=tk.X, pady=2)
        
        # Model type selection
        ttk.Label(model_frame, text="Model Type:").pack(anchor='w', pady=2)
        self.model_type = tk.StringVar(value='auto')
        model_types = ['auto', 'linear', 'decision_tree', 'random_forest', 'logistic', 'svc']
        model_type_combo = ttk.Combobox(model_frame, textvariable=self.model_type, values=model_types, state='readonly')
        model_type_combo.pack(fill=tk.X, pady=2)
        
        # Test size slider
        ttk.Label(model_frame, text="Test Size:").pack(anchor='w', pady=2)
        self.test_size = tk.DoubleVar(value=0.25)
        test_size_slider = ttk.Scale(model_frame, from_=0.1, to=0.5, variable=self.test_size, orient=tk.HORIZONTAL)
        test_size_slider.pack(fill=tk.X, pady=2)
        ttk.Label(model_frame, textvariable=self.test_size).pack(anchor='e')
        
        # Model actions
        actions_frame = ttk.LabelFrame(left_panel, text="Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(actions_frame, text="Train Model", command=self.train_model).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Cross-Validate", command=self.cross_validate_model).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Optimize Model", command=self.optimize_model).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Save Model", command=self.save_model).pack(fill=tk.X, pady=2)
        
        # Right panel for results
        right_panel = ttk.Frame(ml_content)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Notebook for results
        results_notebook = ttk.Notebook(right_panel)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Performance tab
        performance_frame = ttk.Frame(results_notebook, padding=10)
        results_notebook.add(performance_frame, text="Performance")
        
        self.ml_results_text = scrolledtext.ScrolledText(performance_frame, wrap=tk.WORD)
        self.ml_results_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualizations tab
        visualizations_frame = ttk.Frame(results_notebook, padding=10)
        results_notebook.add(visualizations_frame, text="Visualizations")
        
        self.ml_viz_figure = plt.Figure(figsize=(8, 6))
        self.ml_viz_canvas = FigureCanvasTkAgg(self.ml_viz_figure, visualizations_frame)
        self.ml_viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _setup_filter_frame(self, parent):
        """Add data filtering controls"""
        filter_frame = ttk.LabelFrame(parent, text="Data Filtering", padding=10)
        filter_frame.pack(fill=tk.X, pady=5)
        
        filter_controls = ttk.Frame(filter_frame)
        filter_controls.pack(fill=tk.X)
        
        # Column selection
        ttk.Label(filter_controls, text="Column:").pack(side=tk.LEFT, padx=2)
        self.filter_column = ttk.Combobox(filter_controls, state='readonly')
        self.filter_column.pack(side=tk.LEFT, padx=2)
        
        # Operator selection
        ttk.Label(filter_controls, text="Operator:").pack(side=tk.LEFT, padx=2)
        operators = ['>', '<', '==', '!=', 'contains', 'starts with', 'ends with', 'between']
        self.filter_operator = ttk.Combobox(filter_controls, values=operators, state='readonly')
        self.filter_operator.pack(side=tk.LEFT, padx=2)
        
        # Value entry
        ttk.Label(filter_controls, text="Value:").pack(side=tk.LEFT, padx=2)
        self.filter_value = ttk.Entry(filter_controls, width=20)
        self.filter_value.pack(side=tk.LEFT, padx=2)
        
        # Action buttons
        ttk.Button(filter_controls, text="Apply Filter", command=self.apply_filter).pack(side=tk.LEFT, padx=2)
        ttk.Button(filter_controls, text="Reset Filters", command=self.reset_filters).pack(side=tk.LEFT, padx=2)
    
    def _setup_menu_bar(self):
        """Set up the application menu bar"""
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open...", command=self.browse_files, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy, accelerator="Ctrl+Q")
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Analysis menu
        analysis_menu = tk.Menu(menu_bar, tearoff=0)
        analysis_menu.add_command(label="Generate Report", command=self.generate_report)
        analysis_menu.add_command(label="Data Profiling", command=self.profile_data)
        menu_bar.add_cascade(label="Analysis", menu=analysis_menu)
        
        # Export menu
        export_menu = tk.Menu(menu_bar, tearoff=0)
        export_menu.add_command(label="Export to CSV", command=lambda: self.export_data('csv'))
        export_menu.add_command(label="Export to Excel", command=lambda: self.export_data('excel'))
        export_menu.add_command(label="Export to PDF Report", command=lambda: self.export_data('pdf'))
        export_menu.add_command(label="Export to HTML Report", command=lambda: self.export_data('html'))
        export_menu.add_command(label="Export to JSON", command=lambda: self.export_data('json'))
        menu_bar.add_cascade(label="Export", menu=export_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def update_file_path_display(self):
        """Update the file path display with the current file path"""
        if self.file_path:
            self.file_path_var.set(f"File: {self.file_path}")
            # Update the column dropdowns with available columns
            if self.model.df is not None:
                columns = list(self.model.df.columns)
                self.filter_column['values'] = columns
                self.target_column['values'] = columns
    
    def visualize_data(self):
        """Visualize the data based on the selected plot type"""
        if self.model.df is None:
            messagebox.showerror("Error", "No data loaded. Please load a file first.")
            return
        
        try:
            plot_type = self.plot_type.get()
            
            # Clear the current figure
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            
            # Depending on the plot type
            if plot_type == "histogram":
                # For histogram, select a numeric column
                numeric_cols = self.model.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) == 0:
                    messagebox.showerror("Error", "No numeric columns available for histogram.")
                    return
                
                # Use the first numeric column
                col = numeric_cols[0]
                self.model.df[col].hist(ax=self.ax)
                self.ax.set_title(f"Histogram of {col}")
                
            elif plot_type == "box":
                # For box plot, use numeric columns
                numeric_cols = self.model.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) == 0:
                    messagebox.showerror("Error", "No numeric columns available for box plot.")
                    return
                
                self.model.df[numeric_cols[:5]].boxplot(ax=self.ax)  # Limit to 5 columns
                self.ax.set_title("Box Plot")
                
            elif plot_type == "scatter":
                # For scatter, need two numeric columns
                numeric_cols = self.model.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) < 2:
                    messagebox.showerror("Error", "Need at least 2 numeric columns for scatter plot.")
                    return
                
                # Use the first two numeric columns
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                self.ax.scatter(self.model.df[x_col], self.model.df[y_col])
                self.ax.set_xlabel(x_col)
                self.ax.set_ylabel(y_col)
                self.ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                
            elif plot_type == "bar":
                # For bar chart, use a categorical column
                cat_cols = self.model.df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) == 0:
                    messagebox.showerror("Error", "No categorical columns available for bar chart.")
                    return
                
                # Use the first categorical column
                col = cat_cols[0]
                self.model.df[col].value_counts().plot(kind='bar', ax=self.ax)
                self.ax.set_title(f"Bar Chart of {col}")
                
            # Refresh the canvas
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")
    
    def train_model(self):
        """Train a machine learning model"""
        if self.model.df is None:
            messagebox.showerror("Error", "No data loaded. Please load a file first.")
            return
        
        try:
            target_column = self.target_column.get()
            if not target_column:
                messagebox.showerror("Error", "Please select a target column.")
                return
            
            model_type = self.model_type.get()
            test_size = self.test_size.get()
            
            # Train the model
            results = self.model.train_predictive_model(target_column, model_type, test_size)
            
            if not results:
                messagebox.showerror("Error", "Failed to train model. See logs for details.")
                return
            
            # Display results
            self.ml_results_text.delete(1.0, tk.END)
            self.ml_results_text.insert(tk.END, "Model Training Results\n")
            self.ml_results_text.insert(tk.END, "=====================\n\n")
            
            for metric, value in results.items():
                if metric != 'confusion_matrix':  # Display confusion matrix separately
                    self.ml_results_text.insert(tk.END, f"{metric}: {value}\n")
            
            # Plot feature importance or confusion matrix
            self.ml_viz_figure.clear()
            
            if hasattr(self.model.ml_model, 'feature_importance') and self.model.ml_model.feature_importance is not None:
                ax = self.ml_viz_figure.add_subplot(111)
                # Sort feature importance
                sorted_importance = dict(sorted(
                    self.model.ml_model.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
                # Limit to top 10
                if len(sorted_importance) > 10:
                    sorted_importance = dict(list(sorted_importance.items())[:10])
                
                # Plot
                ax.barh(list(sorted_importance.keys()), list(sorted_importance.values()))
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                ax.set_title('Feature Importance')
            
            elif 'confusion_matrix' in results:
                ax = self.ml_viz_figure.add_subplot(111)
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
            
            self.ml_viz_figure.tight_layout()
            self.ml_viz_canvas.draw()
            
            messagebox.showinfo("Success", "Model trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
    
    def cross_validate_model(self):
        """Cross-validate the current model"""
        if not hasattr(self.model, 'ml_model') or self.model.ml_model is None:
            messagebox.showerror("Error", "No model available. Please train a model first.")
            return
        
        try:
            cv_results = self.model.cross_validate_model(cv=5)
            
            if not cv_results:
                messagebox.showerror("Error", "Cross-validation failed. See logs for details.")
                return
            
            # Display results
            self.ml_results_text.delete(1.0, tk.END)
            self.ml_results_text.insert(tk.END, "Cross-Validation Results\n")
            self.ml_results_text.insert(tk.END, "======================\n\n")
            self.ml_results_text.insert(tk.END, f"Mean {cv_results['metric']}: {cv_results['cv_mean']:.4f}\n")
            self.ml_results_text.insert(tk.END, f"Standard Deviation: {cv_results['cv_std']:.4f}\n\n")
            
            self.ml_results_text.insert(tk.END, "Individual Scores:\n")
            for i, score in enumerate(cv_results['cv_scores']):
                self.ml_results_text.insert(tk.END, f"Fold {i+1}: {score:.4f}\n")
            
            # Plot cross-validation results
            self.ml_viz_figure.clear()
            ax = self.ml_viz_figure.add_subplot(111)
            ax.bar(range(1, len(cv_results['cv_scores'])+1), cv_results['cv_scores'])
            ax.axhline(y=cv_results['cv_mean'], color='r', linestyle='-', label=f"Mean: {cv_results['cv_mean']:.4f}")
            ax.set_xlabel('Fold')
            ax.set_ylabel(cv_results['metric'])
            ax.set_title('Cross-Validation Results')
            ax.legend()
            
            self.ml_viz_figure.tight_layout()
            self.ml_viz_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during cross-validation: {str(e)}")
    
    def optimize_model(self):
        """Optimize hyperparameters for the current model"""
        if not hasattr(self.model, 'ml_model') or self.model.ml_model is None:
            messagebox.showerror("Error", "No model available. Please train a model first.")
            return
        
        try:
            messagebox.showinfo("Info", "Hyperparameter optimization started. This may take a while...")
            
            # Start optimization in a separate thread to prevent UI freezing
            def optimize_thread():
                opt_results = self.model.optimize_model()
                
                # Update UI in the main thread
                self.root.after(0, lambda: self._show_optimization_results(opt_results))
            
            threading.Thread(target=optimize_thread).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during hyperparameter optimization: {str(e)}")
    
    def _show_optimization_results(self, results):
        """Show the results of hyperparameter optimization"""
        if not results:
            messagebox.showerror("Error", "Hyperparameter optimization failed. See logs for details.")
            return
        
        # Display results
        self.ml_results_text.delete(1.0, tk.END)
        self.ml_results_text.insert(tk.END, "Hyperparameter Optimization Results\n")
        self.ml_results_text.insert(tk.END, "===============================\n\n")
        self.ml_results_text.insert(tk.END, f"Best {results['scoring']} Score: {results['best_score']:.4f}\n\n")
        
        self.ml_results_text.insert(tk.END, "Best Parameters:\n")
        for param, value in results['best_params'].items():
            self.ml_results_text.insert(tk.END, f"{param}: {value}\n")
        
        # Re-evaluate model with best parameters
        self.train_model()
        
        messagebox.showinfo("Success", "Hyperparameter optimization completed successfully!")
    
    def save_model(self):
        """Save the current model to disk"""
        if not hasattr(self.model, 'ml_model') or self.model.ml_model is None:
            messagebox.showerror("Error", "No model available. Please train a model first.")
            return
        
        try:
            # Ask for filename
            filetypes = [("Joblib files", "*.joblib"), ("All files", "*.*")]
            filename = filedialog.asksaveasfilename(
                defaultextension=".joblib",
                filetypes=filetypes,
                title="Save Model"
            )
            
            if not filename:  # User cancelled
                return
            
            # Save model
            model_path = self.model.save_model(filename)
            
            if model_path:
                messagebox.showinfo("Success", f"Model saved successfully to {model_path}")
            else:
                messagebox.showerror("Error", "Failed to save model. See logs for details.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving model: {str(e)}")
    
    def apply_filter(self):
        """Apply filter to the data"""
        if self.model.df is None:
            messagebox.showerror("Error", "No data loaded. Please load a file first.")
            return
        
        try:
            column = self.filter_column.get()
            operator = self.filter_operator.get()
            value = self.filter_value.get()
            
            if not column or not operator:
                messagebox.showerror("Error", "Please select a column and operator.")
                return
            
            if not value and operator not in ['is null', 'is not null']:
                messagebox.showerror("Error", "Please enter a filter value.")
                return
            
            # Apply filter
            success = self.model.apply_filter(column, operator, value)
            
            if success:
                # Update display with filtered data
                self.show_data_preview()
                self.show_data_info()
                messagebox.showinfo("Success", "Filter applied successfully.")
            else:
                messagebox.showerror("Error", "Failed to apply filter. See logs for details.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error applying filter: {str(e)}")
    
    def reset_filters(self):
        """Reset all filters"""
        if self.model.df is None:
            messagebox.showerror("Error", "No data loaded. Please load a file first.")
            return
        
        try:
            success = self.model.reset_filters()
            
            if success:
                # Update display with original data
                self.show_data_preview()
                self.show_data_info()
                messagebox.showinfo("Success", "Filters reset successfully.")
            else:
                messagebox.showerror("Error", "Failed to reset filters. See logs for details.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error resetting filters: {str(e)}")
    
    def generate_report(self):
        """Generate a comprehensive report"""
        if self.model.df is None:
            messagebox.showerror("Error", "No data loaded. Please load a file first.")
            return
        
        try:
            messagebox.showinfo("Info", "Generating comprehensive report. This may take a while...")
            
            # Start report generation in a separate thread to prevent UI freezing
            def report_thread():
                report_results = self.model.generate_report()
                
                # Update UI in the main thread
                self.root.after(0, lambda: self._show_report_results(report_results))
            
            threading.Thread(target=report_thread).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating report: {str(e)}")
    
    def _show_report_results(self, results):
        """Show the results of report generation"""
        if not results:
            messagebox.showerror("Error", "Report generation failed. See logs for details.")
            return
        
        # Ask if user wants to open the report
        answer = messagebox.askyesno("Report Generated", 
                                    f"Report generated successfully at {results['report_path']}.\n\nDo you want to open it now?")
        
        if answer:
            # Open the report in the default browser
            import webbrowser
            webbrowser.open(results['report_path'])
    
    def export_data(self, format_type):
        """Export data to specified format"""
        if self.model.df is None:
            messagebox.showerror("Error", "No data loaded. Please load a file first.")
            return
        
        try:
            # Export data
            file_path = self.model.export_data(format_type)
            
            if file_path:
                messagebox.showinfo("Success", f"Data exported successfully to {file_path}")
            else:
                messagebox.showerror("Error", "Failed to export data. See logs for details.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {str(e)}")
    
    def profile_data(self):
        """Generate a data profile report"""
        if self.model.df is None:
            messagebox.showerror("Error", "No data loaded. Please load a file first.")
            return
        
        try:
            messagebox.showinfo("Info", "Generating data profile. This may take a while for large datasets...")
            
            # Start profiling in a separate thread to prevent UI freezing
            def profile_thread():
                if self.model.analyzer:
                    report = self.model.analyzer.generate_profile()
                    
                    if report:
                        # Save report to file
                        file_handler = FileHandler()
                        report_path = file_handler.save_report(report)
                        
                        # Open in default browser
                        self.root.after(0, lambda: self._open_profile_report(report_path))
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Failed to generate profile report."))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Analyzer not initialized."))
            
            threading.Thread(target=profile_thread).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating profile: {str(e)}")
    
    def _open_profile_report(self, report_path):
        """Open the profile report in the default browser"""
        try:
            import webbrowser
            webbrowser.open(report_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error opening report: {str(e)}")
    
    def show_documentation(self):
        """Show the application documentation"""
        docs_content = """
        # Dataset Analyzer Pro - Documentation
        
        ## Basic Usage
        1. Load your data using "Browse Files"
        2. Analyze the data to see statistics and visualizations
        3. Use the filtering options to refine your dataset
        4. Generate reports and export results
        
        ## Machine Learning
        1. Select a target column
        2. Choose a model type
        3. Train and evaluate your model
        4. Optimize hyperparameters
        5. Save your model for later use
        
        ## For more information
        Visit the GitHub repository: https://github.com/yourusername/dataset-analyzer
        """
        
        # Create documentation window
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentation")
        doc_window.geometry("700x500")
        
        doc_text = scrolledtext.ScrolledText(doc_window, wrap=tk.WORD)
        doc_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        doc_text.insert(tk.END, docs_content)
        doc_text.config(state=tk.DISABLED)
    
    def show_about(self):
        """Show about information"""
        about_text = """
        Dataset Analyzer Pro
        Version 1.0.0
        
        A comprehensive data analysis tool with machine learning capabilities.
        
        Created for advanced data analysis and visualization.
        """
        
        messagebox.showinfo("About", about_text)

class DataController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
    
    def display_correlations(self, corr_matrix):
        self.view.canvas.figure.clear()
        ax = self.view.canvas.figure.add_subplot(111)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        self.view.canvas.draw()
    
    
    def add_data_filter(self, column, operator, value):
        try:
            self.data_filter.add_filter(column, operator, self._parse_value(value))
            self.view.update_preview(self.data_filter.apply_filters())
        except Exception as e:
            self.view.show_error(f"Filter error: {str(e)}")
            
    def save_current_plot(self):
        if self.current_plot:
            path = self.file_handler.save_plot(self.current_plot, "current_plot")
            self.view.show_info(f"Plot saved to {path}")
            
    
    def export_pdf(self):
        try:
            report_data = {
                'stats': self.analyzer.get_basic_stats(),
                'plots': [self.file_handler.save_plot(fig, name) 
                        for name, fig in self.reporter.generate_interactive_plots()]
            }
            path = self.file_handler.export_pdf(report_data)
            self.view.show_info(f"PDF exported to {path}")
        except Exception as e:
            self.view.show_error(str(e))
    
    def export_html(self):
        try:
            path = self.file_handler.export_html(self.analyzer.data)
            self.view.show_info(f"Data exported to {path}")
            path = self.file_handler.export_html(self.reporter.report_data)
            self.view.show_info(f"Report exported to {path}")
            path = self.file_handler.export_html(self.reporter.correlation_data)
            self.view.show_info(f"Correlations exported to {path}")
            path = self.file_handler.export_html(self.reporter.profile_data)
            self.view.show_info(f"Profile exported to {path}")
            path = self.file_handler.export_html(self.reporter.distribution_data)
            self.view.show_info(f"Distributions exported to {path}")
        except Exception as e:
            self.view.show_error(str(e))
            
    
    def zoom_plot(self):
        if self.current_plot:
            self.current_plot.zoom()
            
    def pan_plot(self):
        if self.current_plot:
            self.current_plot.pan()
            
    def export_csv(self):
        try:
            path = self.file_handler.export_csv(self.analyzer.data)
            self.view.show_info(f"Data exported to {path}")
        except Exception as e:
            self.view.show_error(str(e))
    
    
    def _parse_value(self, value):
        try:
            return float(value) if '.' in value else int(value)
        except:
            return value
        
    def browse_files(self):
        """Open file dialog and load selected file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")]
        )
        
        if file_path:
            success = self.model.load_data(file_path)
            if success:
                self.view.file_path = file_path
                self.view.update_file_path_display()
                return True
        return False
    
    def analyze_data(self):
        """Trigger analysis of the data"""
        if self.model.df is not None:
            self.view.show_data_info()
            self.view.show_data_preview()
            self.view.show_correlations()
            self.view.show_metrics()
            return True
        return False
    
         
    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.process_file(folder_path)
            
    def process_file(self, file_path):
        if not file_path:
            return
        
        try:
            self.view.update_status(f"Loading {os.path.basename(file_path)}...")
            
            # Explicit CSV check
            if file_path.lower().endswith('.csv'):
                data = self.file_handler.load_csv_file(file_path)
            else:
                data = self.file_handler.load_data(file_path)
                
            self.analyzer.set_data(data)
            self.update_preview(data)
            self.view.update_status(f"Successfully loaded {os.path.basename(file_path)}")
            
        except pd.errors.ParserError as e:
            error_msg = f"File format error: {str(e)}\nPlease check if it's a valid CSV/Excel file"
            self.view.show_error(error_msg)
        except UnicodeDecodeError as e:
            error_msg = f"Encoding error: {str(e)}\nTry saving the file as UTF-8 encoded CSV"
            self.view.show_error(error_msg)
        except Exception as e:
            error_msg = f"Failed to load file: {str(e)}"
            self.view.show_error(error_msg)
            
        finally:
            self.view.update_status("")
    
    def update_preview(self, data):
        try:
            # Clear existing preview
            self.view.preview_table.delete(*self.view.preview_table.get_children())
            
            # Set columns
            self.view.preview_table["columns"] = list(data.columns)
            for col in data.columns:
                self.view.preview_table.heading(col, text=col)
                self.view.preview_table.column(col, width=100, anchor='center')
                
            # Add sample rows (limit to 10)
            for _, row in data.head(10).iterrows():
                self.view.preview_table.insert("", tk.END, values=list(row.astype(str)))
                
        except Exception as e:
            self.view.show_error(f"Preview update failed: {str(e)}")

    def preview_table(self, event):
        try:
            selected_item = self.view.preview_table.selection()[0]
            row_data = self.view.preview_table.item(selected_item, "values")
            self.view.show_info("\n".join(row_data))
        except Exception as e:
            self.view.show_error(f"Preview error: {str(e)}")
            
    
    def start_analysis(self):
        if not hasattr(self.controller.analyzer, 'data') or self.controller.analyzer.data is None:
            self.show_error("Please load a dataset first!")
            return
            
        # Get analysis options from view
        options = {
            'profile': self.var_profile.get(),
            'correlations': self.var_corr.get()
        }
        
        threading.Thread(
            target=self.controller._run_analysis,
            args=(options,),
            daemon=True
        ).start()

    
    def _run_analysis(self, options):
        try:
            self.view.update_status("Analyzing...")
            
            if options['profile']:
                # Generate and save the profile report
                report = self.analyzer.generate_profile()
                report_path = self.file_handler.save_report(report)
                self.view.update_status(f"Report saved to: {report_path}")
                
            if options['correlations']:
                corr_matrix = self.analyzer.calculate_correlations()
                self.view.display_correlations(corr_matrix)
                
            self.view.update_status("Analysis complete")
        except Exception as e:
            self.view.show_error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = DataVisualizerApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Application Error", f"An error occurred: {str(e)}")
        print(f"Error: {e}")