import pandas as pd
import plotly.express as px
from model import ModelTrainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Instantiate the ModelTrainer class
trainer = ModelTrainer()

class GraphBuilder:
    def __init__(self, csv_file='data.csv'):
        # Reading the file
        self.df = pd.read_csv(csv_file)

        # Loading column names into a data dictionary
        self.data_dict = self.df.columns

        # Renaming columns
        for i in range(1, len(self.df.columns)):
            self.df.rename(columns={self.df.columns[i]: f"X{i}"}, inplace=True)

    def update_plots(self, selected_feature, clipped):
        feat = f'X{self.data_dict.get_loc(selected_feature)}'

        # Plotting the boxplot
        fig_boxplot = px.box(self.df, x='Bankrupt?', y=feat, points='all')
        fig_boxplot.update_layout(
            xaxis_title="Bankrupt",
            yaxis_title=f"{selected_feature.strip()}",
            title="Boxplot" if not clipped else "Clipped Boxplot",
            title_x=0.5  # Center the title along the x-axis
        )

        if clipped:
            q1, q9 = self.df[feat].quantile([0.1, 0.9])
            mask = self.df[feat].between(q1, q9)
            fig_clipped = px.box(self.df[mask], x='Bankrupt?', y=feat, points='all')
            fig_clipped.update_layout(
                xaxis_title="Bankrupt",
                yaxis_title=f"{selected_feature.strip()}",
                title="Clipped Boxplot",
                title_x=0.5  # Center the title along the x-axis
            )
            return fig_clipped, self.feat_histogram(selected_feature, clipped)

        return fig_boxplot, self.feat_histogram(selected_feature, clipped)

    def feat_histogram(self, selected_feature, clipped=False):
        feat = f'X{self.data_dict.get_loc(selected_feature)}'

        # Create a histogram using Plotly Express
        fig = px.histogram(self.df, x=feat, color='Bankrupt?', marginal='box', nbins=30,
                           title="Histogram" if not clipped else "Clipped Histogram",
                           labels={'Bankrupt?': 'Bankrupt'},
                           category_orders={'Bankrupt?': ['0', '1']})

        # Update the layout
        fig.update_layout(xaxis_title=f"{selected_feature.strip()}",
                          yaxis_title="Count",
                          title_x=0.5)  # Center the title along the x-axis

        # If clipped, update the data for the clipped histogram
        if clipped:
            q1, q9 = self.df[feat].quantile([0.1, 0.9])
            mask = self.df[feat].between(q1, q9)
            fig_clipped = px.histogram(self.df[mask], x=feat, color='Bankrupt?', marginal='box', nbins=30,
                                       title="Clipped Histogram",
                                       labels={'Bankrupt?': 'Bankrupt'},
                                       category_orders={'Bankrupt?': ['0', '1']})

            # Update the layout for the clipped histogram
            fig_clipped.update_layout(xaxis_title=f"{selected_feature.strip()}",
                                      yaxis_title="Count",
                                      title_x=0.5)  # Center the title along the x-axis

            return fig_clipped

        return fig
    
  