import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from flask import Flask, request, jsonify
import pandas as pd, plotly.express as px, json, uuid

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Tema + fonte
# ──────────────────────────────────────────────────────────────────────────────
external_stylesheets = [
    dbc.themes.DARKLY,
    "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap"
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
stored_indicators = None

# ──────────────────────────────────────────────────────────────────────────────
# 2. Paleta & estilos
# ──────────────────────────────────────────────────────────────────────────────
COLORWAY   = ['#4E79A7', '#F28E2B', '#E15759', '#59A14F', '#EDC948',
              '#B07AA1', '#FF9DA7', '#9C755F', '#9ccef2', '#bab0ac']
CARD_STYLE = {
    "backgroundColor": "rgba(25,25,30,0.75)",
    "backdropFilter": "blur(6px)",
    "borderRadius": "10px",
    "border": "1px solid rgba(88,101,242,0.35)",
    "boxShadow": "0 6px 20px rgba(0,0,0,0.55)",
    "marginBottom": "18px",
    "transition": "transform .25s",
}
HEADER_STYLE = {
    "background": "linear-gradient(135deg,rgba(88,101,242,0.9),rgba(88,101,242,0.3))",
    "fontFamily": "Poppins, sans-serif",
    "letterSpacing": ".5px"
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. Layout
# ──────────────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2(
        "Intelligent Spreadsheet Analyzer – Dashboard Web",
        className="text-center text-primary mb-4",
        style={"fontFamily":"Poppins","fontWeight":700,"letterSpacing":"1px"}
    ), width=12)),
    dbc.Row(dbc.Col(
        dcc.Dropdown(
            id='chart-type-selector',
            options=[{"label":n,"value":v} for n,v in [
                ("Bar Chart (Frequência)","bar"),
                ("Pie Chart (Distribuição)","pie"),
                ("Line Chart (Tendência)","line"),
                ("Scatter Plot (Dispersão)","scatter"),
                ("Box Plot (Distribuição)","box")
            ]],
            value="bar", clearable=False,
            style={"color":"#000","fontFamily":"Poppins"}
        ),
        width=6, className="mx-auto"), justify="center"
    ),
    dbc.Row(dbc.Col(
        dcc.Loading(id="loading-graphs", type="circle",
                    children=html.Div(id='dynamic-graphs-container'),
                    style={"width":"100%"}), width=12)
    ),
    dbc.Row(dbc.Col(html.Footer(
        "© 2025 Intelligent Spreadsheet Analyzer | Desenvolvido por José Enoque",
        className="text-center text-secondary my-3",
        style={"fontFamily":"Poppins","opacity":0.75,"fontSize":"12px","fontStyle":"italic"}
    ), width=12))
], fluid=True)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Rota de recepção de dados
# ──────────────────────────────────────────────────────────────────────────────
@server.route('/update_data', methods=['POST'])
def update_data():
    global stored_indicators
    data = request.get_json()
    if not data:
        return {"error":"No data received"}, 400
    for g in data.get("agrupamentos", []):
        t = g.get("tabela")
        if isinstance(t,str):
            try:  g["tabela"] = json.loads(t)
            except: g["tabela"] = None
    stored_indicators = data
    return {"status":"success"}

# ──────────────────────────────────────────────────────────────────────────────
# 5. Callback principal
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(Output('dynamic-graphs-container', 'children'),
              Input('chart-type-selector', 'value'))
def render_cards(chart_type):
    if not stored_indicators:
        return html.Div("Nenhum dado recebido ainda.",
                        style={"color":"#ccc","fontStyle":"italic",
                               "padding":"20px","textAlign":"center"})
    rows, cols = [], []
    for grp in stored_indicators['agrupamentos']:
        df = pd.DataFrame(grp.get("tabela") or [])
        if df.empty: continue

        y_col = ('termo_base' if 'termo_base' in df.columns else
                 'termo' if 'termo' in df.columns else None)
        has_f  = 'frequencia' in df.columns
        val_col= next((c for c in ['valor','valor_medio','media'] if c in df.columns), None)
        title  = grp['coluna']; fig=None

        if chart_type=="bar" and has_f and y_col:
            fig = px.bar(df.head(8), y=y_col, x='frequencia', orientation='h',
                         template="plotly_dark", color_discrete_sequence=COLORWAY,
                         labels={'frequencia':'Frequência', y_col:title})
            fig.update_layout(title=f"Barra • {title}", yaxis={'autorange':'reversed'})
        elif chart_type=="pie" and y_col:
            if has_f:
                g=df.groupby(y_col)['frequencia'].sum().reset_index().nlargest(5,'frequencia')
                fig=px.pie(g, names=y_col, values='frequencia',
                           template="plotly_dark", color_discrete_sequence=COLORWAY,
                           title=f"Pizza • {title}")
            elif val_col:
                g=df.groupby(y_col)[val_col].mean().reset_index().nlargest(5,val_col)
                fig=px.pie(g, names=y_col, values=val_col,
                           template="plotly_dark", color_discrete_sequence=COLORWAY,
                           title=f"Pizza • {title} (média)")
            if fig: fig.update_traces(textposition='inside', textinfo='percent+label',
                                      marker=dict(line=dict(color="#202225",width=2)))
        elif chart_type=="line" and val_col and y_col:
            fig=px.line(df.head(8), x=y_col, y=val_col, markers=True,
                        template="plotly_dark", color_discrete_sequence=['#F28E2B'],
                        title=f"Linha • {title}")
        elif chart_type=="scatter":
            nums=df.select_dtypes('number').columns
            if len(nums)>=2:
                fig=px.scatter(df.head(30), x=nums[0], y=nums[1], color=y_col,
                               template="plotly_dark", color_discrete_sequence=COLORWAY,
                               title=f"Scatter • {title}")
        elif chart_type=="box" and val_col and y_col:
            fig=px.box(df, x=y_col, y=val_col, template="plotly_dark",
                       color_discrete_sequence=['#E15759'], title=f"Box • {title}")

        body = dcc.Graph(id=f"g-{chart_type}-{title}", figure=fig or {},
                         config={'responsive':True}) if fig else \
               html.Div("Sem dados.", style={"color":"#FF5E5B","fontStyle":"italic"})

        card = dbc.Card([
            dbc.CardHeader(f"{chart_type.title()} – {title}", style=HEADER_STYLE,
                           className="fw-semibold text-white"),
            dbc.CardBody(body, style={"padding":"10px"})
        ], key=str(uuid.uuid4()), style=CARD_STYLE, className="h-100 hover-zoom")

        cols.append(dbc.Col(card, width=6))
        if len(cols)==2:
            rows.append(dbc.Row(cols, className="gx-2")); cols=[]
    if cols: rows.append(dbc.Row(cols, className="gx-2"))
    return html.Div(rows, key=chart_type)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  HTML base (corrigido: inclui {%renderer%})
# ──────────────────────────────────────────────────────────────────────────────
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Dashboard</title>
    {%favicon%}
    {%css%}
    <style>
        body{font-family:"Poppins",sans-serif;}
        .hover-zoom:hover{transform:translateY(-4px) scale(1.02);}
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}  <!-- DashRenderer obrigatório -->
    </footer>
</body>
</html>
'''

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=True)
