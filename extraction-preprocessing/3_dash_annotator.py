from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import os
import json
import numpy as np

json_rootdirectory = r"C:\Users\flori\OneDrive - univ-angers.fr\Documents\Home\Research\SPECTR\ISPECTR\data\2025\lemans\preannotation"

app = Dash()

sideBar={
    "toolPanels": [
        {
            "id": "columns",
            "labelDefault": "Columns",
            "labelKey": "columns",
            "iconKey": "columns",
            "toolPanel": "agColumnsToolPanel",
        },
        {
            "id": "filters",
            "labelDefault": "Filters",
            "labelKey": "filters",
            "iconKey": "filter",
            "toolPanel": "agFiltersToolPanel",
        },
        {
            "id": "filters 2",
            "labelKey": "filters",
            "labelDefault": "More Filters",
            "iconKey": "menu",
            "toolPanel": "agFiltersToolPanel",
        },
    ],
    "position": "left",
    "defaultToolPanel": "filters",
}

app.layout = [
    html.H1(children='iSPECTR annotation tool'),
    dcc.Dropdown(os.listdir(os.path.join(json_rootdirectory, "input_jsons")), '', id='json-dropdown-selection'),
    dcc.Graph(id='elp-graph-content')
]

@callback(
    Output('elp-graph-content', 'figure'),
    Input('json-dropdown-selection', 'value')
)
def update_graph(json_filename):
    if json_filename == '':
        return None

    with open(os.path.join(json_rootdirectory, "input_jsons", json_filename), "r") as f:
        sample_data = json.load(f)

    plot_df = pd.DataFrame({"Point": np.arange(304) + 1,
                            "Relative intensity": np.array(sample_data["traces"]["ELP"]["data"])})

    fig = px.line(plot_df, x="Point", y="Relative intensity", labels={"Trace": "ELP"})
    return fig


if __name__ == '__main__':
    app.run()