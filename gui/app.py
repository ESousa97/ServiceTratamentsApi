import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from flask import Flask, request, jsonify
import pandas as pd
import plotly.express as px
import json

external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

stored_indicators = None

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Intelligent Spreadsheet Analyzer - Dashboard Web",
                        className="text-center text-primary mb-4",
                        style={"fontWeight": "700", "letterSpacing": "1.5px"}),
                width=12),
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(id='dynamic-graphs-container',
                     style={
                         "display": "flex",
                         "flexWrap": "wrap",
                         "justifyContent": "space-between",
                         "gap": "20px",
                         "paddingLeft": "10px",
                         "paddingRight": "10px",
                     }),
            width=12,
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.Footer("© 2025 Intelligent Spreadsheet Analyzer | Desenvolvido por Você",
                        className="text-center text-secondary my-3",
                        style={"fontSize": "12px", "fontStyle": "italic"}),
            width=12
        )
    ])
], fluid=True)


@server.route('/update_data', methods=['POST'])
def update_data():
    global stored_indicators
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    agrupamentos = data.get("agrupamentos", [])
    for grupo in agrupamentos:
        tabela = grupo.get("tabela")
        if tabela and isinstance(tabela, str):
            try:
                grupo["tabela"] = json.loads(tabela)
            except Exception:
                grupo["tabela"] = None

    stored_indicators = data
    return jsonify({"status": "success"})


@app.callback(
    Output('dynamic-graphs-container', 'children'),
    Input('dynamic-graphs-container', 'id')
)
def update_graphs(_):
    if not stored_indicators:
        return html.Div(
            "Nenhum dado recebido ainda para exibir gráficos.",
            style={"color": "#ccc", "fontStyle": "italic", "padding": "20px"}
        )

    agrupamentos = stored_indicators.get("agrupamentos", [])
    cards = []

    for i, grp in enumerate(agrupamentos):
        tabela = grp.get("tabela")
        if tabela:
            try:
                df = pd.DataFrame(tabela)
                if df.empty:
                    continue

                cols = df.columns
                # Só gera gráfico se colunas essenciais existirem
                if 'frequencia' not in cols or ('termo_base' not in cols and 'termo' not in cols):
                    continue

                df_chart = df.head(8)
                x_col = 'frequencia'
                y_col = 'termo_base' if 'termo_base' in cols else 'termo'

                fig = px.bar(
                    df_chart, y=y_col, x=x_col, orientation='h',
                    labels={x_col: 'Frequência', y_col: grp['coluna']},
                    title=f"Frequência Top - {grp['coluna']}"
                )
                fig.update_layout(
                    template="plotly_dark",
                    yaxis={'autorange': 'reversed'},
                    margin=dict(l=40, r=20, t=50, b=40),
                    font=dict(size=12, color="#ddd", family="Segoe UI")
                )

                card = dbc.Card([
                    dbc.CardHeader(f"Gráfico {i + 1}: {grp['coluna']}",
                                   className="bg-dark text-primary fw-semibold"),
                    dbc.CardBody(
                        dcc.Graph(figure=fig, config={'displayModeBar': True})
                    )
                ],
                    className="shadow-lg",
                    style={
                        "backgroundColor": "#181818",
                        "width": "48%",
                        "marginBottom": "16px",
                        "boxShadow": "0 4px 12px rgba(0,0,0,0.6)"
                    }
                )
                cards.append(card)

            except Exception as e:
                print(f"Erro ao gerar gráfico para {grp['coluna']}: {e}")
                continue

    if not cards:
        return html.Div(
            "Nenhum agrupamento com dados para gráfico.",
            style={"color": "#ccc", "fontStyle": "italic", "padding": "20px"}
        )

    return cards


if __name__ == '__main__':
    app.run_server(debug=True)
