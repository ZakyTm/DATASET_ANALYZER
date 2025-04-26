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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings("ignore", message="Upgrade to ydata-sdk")

class DataVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Analyzer Pro")
        self.root.geometry("1000x800")
        # Initialize the UI
        self._setup_ui()
        
        # Initialize the controller
        self._setup_controller()
        
        # Initialize the analyzer
        
        #state variables
        self._add_advanced_controls()
        self._setup_export()
        self._setup_filtering()
        # Bindings
        self._setup_bindings()
        
    
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
    

    # def _setup_ui(self):
    #     # Main container
    #     main_frame = ttk.Frame(self.root)
    #     main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    #     # File input section
    #     self._create_file_input_section(main_frame)
        
    #     # Analysis controls
    #     self._create_analysis_controls(main_frame)
        
    #     # Results area
    #     self._create_results_section(main_frame)
        
    #     # Status bar
    #     self.status_var = tk.StringVar()
    #     ttk.Label(main_frame, textvariable=self.status_var, 
    #             relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)
    
    
    def _setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File input section
        file_frame = ttk.LabelFrame(main_frame, text="Data Input", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        # Buttons with better spacing
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Browse Files", command=self.controller.browse_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Folder", command=self.controller.browse_folder).pack(side=tk.LEFT)

        # Analysis controls
        control_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Checkboxes
        self.var_profile = tk.BooleanVar(value=True)
        self.var_corr = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Generate Profile", variable=self.var_profile).pack(anchor=tk.W)
        ttk.Checkbutton(control_frame, text="Show Correlations", variable=self.var_corr).pack(anchor=tk.W)

        # Results area
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Console tab
        console_frame = ttk.Frame(notebook)
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True)
        notebook.add(console_frame, text="Console")

        # Visualization tab
        vis_frame = ttk.Frame(notebook)
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        notebook.add(vis_frame, text="Visualizations")

        # Analyze button at bottom
        ttk.Button(main_frame, text="Analyze", command=self.controller.start_analysis).pack(pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)

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

    def update_status(self, message):
        self.status_var.set(message)
    
    def show_error(self, message):
        messagebox.showerror("Error", message)

class DataController:
    def __init__(self, view):
        self.view = view
        self.analyzer = DataAnalyzer()
        self.file_handler = FileHandler()
        self.data = None
        self.reporter = ReportGenerator(self.analyzer) 
        self.current_filters = []
        self.data_filter = DataFilter(pd.DataFrame())
        self.current_plot = None
    
    
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
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_file(file_path)
            
    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.process_file(folder_path)
            
    # def process_file(self, file_path):
    #     if not validate_file(file_path):
    #         self.view.show_error("Invalid file format")
    #         return
            
    #     try:
    #         data = self.file_handler.load_data(file_path)
    #         self.analyzer.set_data(data)
    #         self.view.update_status(f"Loaded: {file_path}")
    #         self.update_preview(data)
    #     except Exception as e:
    #         self.view.show_error(str(e))
    
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

    # def _run_analysis(self, options):
    #     try:
    #         self.view.update_status("Analyzing...")
            
    #         if options['profile']:
    #             report = self.analyzer.generate_profile()
    #             self.file_handler.save_report(report)
                
    #         if options['correlations']:
    #             corr_matrix = self.analyzer.calculate_correlations()
    #             self.view.display_correlations(corr_matrix)
                
    #         self.view.update_status("Analysis complete")
    #     except Exception as e:
    #         self.view.show_error(f"Analysis failed: {str(e)}")
    
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
    root = tk.Tk()
    app = DataVisualizerApp(root)
    root.mainloop()