import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from business import GraphBuilder
from model import ModelTrainer
from sklearn import metrics
import plotly.graph_objs as go

# Instantiate the GraphBuilder class
gb = GraphBuilder()

trainer=ModelTrainer()

# Define the Dash app
app = dash.Dash(__name__)
server=app.server

# Define the layout of the app
app.layout = html.Div([
    # Heading
    html.H1("Bankruptcy Prediction"),

    # Label for the dropdown
    html.Label("Select a Feature from the Dropdown"),

    # Dropdown for selecting feature
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in gb.data_dict[1:]],
        value=gb.data_dict[1],  # Set the default value
        style={'width': '50%'}
    ),

    # Radio button for 'clipped'
    dcc.RadioItems(
        id='clipped-radio',
        options=[
            {'label': 'Not Clipped', 'value': False},
            {'label': 'Clipped', 'value': True}
        ],
        value=False,  # Set the default value
        labelStyle={'display': 'block'}
    ),

    # Container for the plots (side by side)
    html.Div([
        # Output for the boxplot
        dcc.Graph(id='boxplot-output', style={'width': '50%', 'display': 'inline-block'}),

        # Output for the histogram
        dcc.Graph(id='histogram-output', style={'width': '50%', 'display': 'inline-block'})
    ]),

    html.H2('Confusion Matrix'),

    html.Div([
        # Input components
        html.Label('Select threshold'),
        dcc.Slider(
            id='threshold-slider',
            min=0,
            max=1,
            value=0.5,
            step=0.05,
            marks={i: str(i) for i in range(0, 11)},
            tooltip={'placement': 'bottom'},
            className='mb-5'
        ),
        html.Label('Correct Prediction Profit'),
        dcc.Input(
            id='tp-profit-input',
            type='number',
            value=100000000,
            placeholder='Correct Prediction Profit',
            className='mb-5'
        ),
        html.Label('False Prediction Loss'),
        dcc.Input(
            id='fp-loss-input',
            type='number',
            value=250000000,
            placeholder='False Prediction Loss',
            className='mb-5'
        ),

        # Output for the confusion matrix
        dcc.Graph(id='confusion-matrix-output', style={'width': '80%', 'display': 'inline-block'}),

        html.Div(id='profit-info', style={'margin-top': '20px'}),

    ])
])

# Define callback to update the boxplot and histogram based on inputs
@app.callback(
    [Output('boxplot-output', 'figure'),
     Output('histogram-output', 'figure')],
    [Input('feature-dropdown', 'value'),
     Input('clipped-radio', 'value')]
)
def display_plots(selected_feature, clipped):
    return gb.update_plots(selected_feature, clipped)


@app.callback(
    Output('confusion-matrix-output', 'figure'),
    Input('threshold-slider', 'value'),
)
def update_cnf_matrix(threshold):
    # Assuming X_test and y_test are defined before calling this function
    X_test = gb.df.drop('Bankrupt?', axis=1)
    y_test = gb.df['Bankrupt?']

    # Get the trained model directly
    model = trainer.model

    y_pred_proba = model.predict_proba(X_test)[:, -1]
    y_pred = y_pred_proba > threshold
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    layout = go.Layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label")
    )

    # Create the heatmap trace
    trace = go.Heatmap(z=conf_matrix, x=['Not Bankrupt', 'Bankrupt'], y=['Not Bankrupt', 'Bankrupt'], 
                       colorscale='Blues')

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)

    return fig


@app.callback(
    Output('profit-info', 'children'),
    [Input('threshold-slider', 'value'),
     Input('tp-profit-input', 'value'),
     Input('fp-loss-input', 'value')]
)
def update_profit_info(threshold, tp_profit, fp_loss):
    # Assuming X_test and y_test are defined before calling this function
    X_test = gb.df.drop('Bankrupt?', axis=1)
    y_test = gb.df['Bankrupt?']

    # Get the trained model directly
    model = trainer.model

    y_pred_proba = model.predict_proba(X_test)[:, -1]
    y_pred = y_pred_proba > threshold
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    tp, fp, _, _ = conf_matrix.ravel()

    profit = tp * tp_profit
    loss = fp * fp_loss

    text = f"Profit: €{profit}\nLoss: €{loss}\nNet Profit: €{profit-loss}"

    return text



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
