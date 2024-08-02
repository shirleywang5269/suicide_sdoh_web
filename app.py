from dash import Dash, html, dcc, Input, Output
import sdoh
import demo

app = Dash(__name__)

app.layout = html.Div([
    dcc.Tabs(id="tabs-example", value='tab-1', children=[
        dcc.Tab(label='SDOH Page', value='tab-1'),
        dcc.Tab(label='Demo Page', value='tab-2'),
    ]),
    html.Div(id='tabs-content-example')
])

@app.callback(Output('tabs-content-example', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return sdoh.layout
    elif tab == 'tab-2':
        return demo.layout

if __name__ == '__main__':
    app.run_server(debug=True)
