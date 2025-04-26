import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

class ReportGenerator:
    def __init__(self, data_analyzer):
        self.analyzer = data_analyzer
        self.data = data_analyzer.data
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

    def generate_interactive_plots(self):
        """Generate interactive Plotly visualizations"""
        figs = []
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                fig = self._create_numeric_plot(col)
            else:
                fig = self._create_categorical_plot(col)
            figs.append((col, fig))
        self._save_plots(figs)
        return figs

    def _create_numeric_plot(self, column):
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f"Distribution of {column}", 
            f"Box Plot of {column}"
        ))
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=self.data[column], name="Distribution"),
            row=1, col=1
        )
        
        # Box Plot
        fig.add_trace(
            go.Box(y=self.data[column], name="Spread"),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"Analysis of {column}",
            height=400,
            showlegend=False
        )
        return fig

    def _create_categorical_plot(self, column):
        counts = self.data[column].value_counts().nlargest(20)
        fig = px.bar(
            counts,
            x=counts.values,
            y=counts.index,
            orientation='h',
            title=f"Top Categories in {column}"
        )
        fig.update_layout(height=400)
        return fig

    def _save_plots(self, figures):
        """Save plots as HTML and PNG"""
        for col, fig in figures:
            safe_name = "".join(c if c.isalnum() else "_" for c in col)
            # Save as interactive HTML
            fig.write_html(self.report_dir / f"{safe_name}_interactive.html")
            # Save static version
            fig.write_image(self.report_dir / f"{safe_name}_static.png")

    def generate_correlation_matrix(self):
        """Create advanced correlation visualization"""
        corr = self.analyzer.calculate_correlations()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )
        fig.write_html(self.report_dir / "correlation_matrix.html")
        return fig