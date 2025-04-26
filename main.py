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
from utils.helpers import validate_file
from tkinterdnd2 import DND_FILES
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import warnings
warnings.filterwarnings("ignore", message="Upgrade to ydata-sdk")

class DataModel:
    """Model component that handles data operations"""
    
    def __init__(self):
        self.df = None
        self.file_path = None
    
    def load_data(self, file_path):
        """Load data from file path"""
        try:
            self.file_path = file_path
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(file_path)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_data_info(self):
        """Return information about the dataset"""
        if self.df is not None:
            info = {
                'shape': self.df.shape,
                'columns': self.df.columns.tolist(),
                'dtypes': self.df.dtypes.to_dict(),
                'missing_values': self.df.isnull().sum().to_dict()
            }
            return info
        return None
    
    def get_data_preview(self, rows=5):
        """Return preview of the data"""
        if self.df is not None:
            return self.df.head(rows)
        return None
    
    def get_correlations(self):
        """Return correlation matrix"""
        if self.df is not None:
            # Filter only numeric columns
            numeric_df = self.df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                return numeric_df.corr()
        return pd.DataFrame()
    
    def get_metrics(self):
        """Return basic statistics"""
        if self.df is not None:
            # Filter only numeric columns
            numeric_df = self.df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                return numeric_df.describe()
        return pd.DataFrame()


class DataVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Analyzer")
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
            
    def _setup_controller(self):
        """Initialize the controller component"""
        self.controller = DataController(self)
       
    
    def _add_advanced_controls(self):
    # Plot controls
        self.plot_controls = ttk.Frame(self.root)
        self.plot_controls.pack(fill=tk.X)
        
        ttk.Button(self.plot_controls, text="Zoom", command=self.controller.zoom_plot).pack(side=tk.LEFT)
        ttk.Button(self.plot_controls, text="Pan", command=self.controller.pan_plot).pack(side=tk.LEFT)
        ttk.Button(self.plot_controls, text="Save Plot", 
                command=self.controller.save_current_plot).pack(side=tk.LEFT)
    
    def _setup_export(self):
        # Export menu
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export PDF", command=self.controller.export_pdf)
        file_menu.add_command(label="Export CSV", command=self.controller.export_csv)
        file_menu.add_command(label="Export HTML", command=self.controller.export_html)
        menubar.add_cascade(label="Export", menu=file_menu)
        self.root.config(menu=menubar)
    
    def _setup_filtering(self):
        # Filter controls
        filter_frame = ttk.LabelFrame(self.root, text="Data Filtering")
        filter_frame.pack(fill=tk.X)
        
        self.filter_column = ttk.Combobox(filter_frame)
        self.filter_column.pack(side=tk.LEFT)
        
        self.filter_operator = ttk.Combobox(filter_frame, 
                                          values=['>', '<', '==', 'contains', 'between'])
        self.filter_operator.pack(side=tk.LEFT)
        
        self.filter_value = ttk.Entry(filter_frame)
        self.filter_value.pack(side=tk.LEFT)
        
        ttk.Button(filter_frame, text="Add Filter", 
                 command=self.add_filter).pack(side=tk.LEFT)
        ttk.Button(filter_frame, text="Clear Filters", 
                 command=self.clear_filters).pack(side=tk.LEFT)
        

    def add_filter(self):
        col = self.filter_column.get()
        op = self.filter_operator.get()
        val = self.filter_value.get()
        self.controller.add_data_filter(col, op, val)
        
    def clear_filters(self):
        self.controller.clear_filters()
    
    
    
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
        plot_dropdown['values'] = ('histogram', 'box', 'scatter', 'bar')
        plot_dropdown.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(viz_controls, text="Generate Plot", command=self.visualize_data).pack(side=tk.LEFT, padx=5)
        
        # Matplotlib figure and canvas
        self.figure = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, viz_frame)
        self.toolbar.update()
        
    def update_file_path_display(self):
        """Update the file path display"""
        if self.file_path:
            self.file_path_var.set(f"Selected file: {self.file_path}")
        else:
            self.file_path_var.set("No file selected")
    
    def visualize_data(self):
        """Generate visualization based on selected plot type"""
        if self.model.df is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        # Clear previous figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        plot_type = self.plot_type.get()
        
        try:
            numeric_df = self.model.df.select_dtypes(include=['number'])
            
            if plot_type == 'histogram':
                numeric_df.hist(ax=ax, figsize=(10, 6))
                ax.set_title("Histograms of Numeric Features")
            
            elif plot_type == 'box':
                numeric_df.boxplot(ax=ax)
                ax.set_title("Box Plots of Numeric Features")
            
            elif plot_type == 'scatter':
                # Take first two numeric columns for scatter plot
                if len(numeric_df.columns) >= 2:
                    x_col = numeric_df.columns[0]
                    y_col = numeric_df.columns[1]
                    ax.scatter(numeric_df[x_col], numeric_df[y_col])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            
            elif plot_type == 'bar':
                # For bar charts, use the first numeric column
                if len(numeric_df.columns) >= 1:
                    y_col = numeric_df.columns[0]
                    numeric_df[y_col].value_counts().plot(kind='bar', ax=ax)
                    ax.set_title(f"Bar Chart of {y_col}")
            
            self.canvas.draw()
        
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Error generating plot: {str(e)}")
        
    def _create_file_input_section(self, parent):
        file_frame = ttk.LabelFrame(parent, text="Data Input")
        file_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # File controls
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Browse Files", command=self.controller.browse_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Folder", command=self.controller.browse_folder).pack(side=tk.LEFT)

        # Initialize preview table
        self.preview_table = ttk.Treeview(file_frame, height=5, show='headings')
        
        # Add scrollbars
        scroll_y = ttk.Scrollbar(file_frame, orient='vertical', command=self.preview_table.yview)
        scroll_x = ttk.Scrollbar(file_frame, orient='horizontal', command=self.preview_table.xview)
        self.preview_table.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        # Layout components
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_table.pack(fill=tk.BOTH, expand=True)

    def _create_analysis_controls(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Analysis options
        self.var_profile = tk.BooleanVar(value=True)
        self.var_corr = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Generate Profile", 
                      variable=self.var_profile).pack(side=tk.LEFT)
        ttk.Checkbutton(control_frame, text="Show Correlations", 
                      variable=self.var_corr).pack(side=tk.LEFT)
        
        # Action buttons
        ttk.Button(control_frame, text="Analyze", 
                 command=self.controller.start_analysis).pack(side=tk.RIGHT)
        
    
    # Add these to the DataVisualizerApp class
    def _create_preprocessing_controls(self, parent):
        preprocess_frame = ttk.LabelFrame(parent, text="Data Preprocessing")
        preprocess_frame.pack(fill=tk.X, pady=5)
        
        self.var_clean_names = tk.BooleanVar(value=True)
        self.var_handle_missing = tk.BooleanVar(value=True)
        self.var_normalize = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(preprocess_frame, text="Clean Column Names",
                    variable=self.var_clean_names).pack(side=tk.LEFT)
        ttk.Checkbutton(preprocess_frame, text="Handle Missing Values",
                    variable=self.var_handle_missing).pack(side=tk.LEFT)
        ttk.Checkbutton(preprocess_frame, text="Normalize Data",
                    variable=self.var_normalize).pack(side=tk.LEFT)

    def _create_export_controls(self, parent):
        export_frame = ttk.Frame(parent)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Export Report as PDF",
                command=self.controller.export_pdf).pack(side=tk.LEFT)
        ttk.Button(export_frame, text="Export Data as CSV",
                command=self.controller.export_csv).pack(side=tk.LEFT)

    def _create_results_section(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Console tab
        console_frame = ttk.Frame(notebook)
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True)
        notebook.add(console_frame, text="Console")
        
        # Visualization tab
        vis_frame = ttk.Frame(notebook)
        self.canvas = FigureCanvasTkAgg(plt.figure(), vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        notebook.add(vis_frame, text="Visualizations")
    
    def analyze_data(self):
        """Delegate to controller's analyze_data method"""
        if not self.controller.analyze_data():
            messagebox.showerror("Error", "Please select a valid data file first.")
    

    def show_data_info(self):
        """Display data information"""
        info = self.model.get_data_info()
        if info:
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, f"Dataset Shape: {info['shape']}\n\n")
            self.data_info_text.insert(tk.END, f"Columns: {', '.join(info['columns'])}\n\n")
            
            self.data_info_text.insert(tk.END, "Data Types:\n")
            for col, dtype in info['dtypes'].items():
                self.data_info_text.insert(tk.END, f"{col}: {dtype}\n")
            
            self.data_info_text.insert(tk.END, "\nMissing Values:\n")
            for col, count in info['missing_values'].items():
                self.data_info_text.insert(tk.END, f"{col}: {count}\n")
                
    def show_data_preview(self):
        """Display data preview"""
        preview = self.model.get_data_preview()
        if preview is not None:
            self.data_preview_text.delete(1.0, tk.END)
            self.data_preview_text.insert(tk.END, preview.to_string())

    def show_correlations(self):
        """Display correlation matrix"""
        corr = self.model.get_correlations()
        if not corr.empty:
            self.correlations_text.delete(1.0, tk.END)
            self.correlations_text.insert(tk.END, corr.to_string())
        else:
            self.correlations_text.delete(1.0, tk.END)
            self.correlations_text.insert(tk.END, "No numeric columns available for correlation analysis.")
    
    
    def show_metrics(self):
        """Display metrics"""
        metrics = self.model.get_metrics()
        if not metrics.empty:
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, metrics.to_string())
        else:
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "No numeric columns available for metrics calculation.")

    def update_status(self, message):
        self.status_var.set(message)
    
    def show_error(self, message):
        messagebox.showerror("Error", message)

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