# FILE: your_ml_project/app.py (Integrated and Fully Corrected)

from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

app = Flask(__name__)

# ======================================================================
# 1. GLOBAL REGION SALES MODEL (China, EU27, USA)
# ======================================================================
df_regions = pd.read_csv("ev_sales_yearwise_by_region.csv")
df_regions.columns = df_regions.columns.str.strip()
regions = ['China', 'EU27', 'USA']
region_models = {}
poly_regions = PolynomialFeatures(degree=3)
X_regions = df_regions[['Year']].values
X_regions_poly = poly_regions.fit_transform(X_regions)
for region in regions:
    y_region = df_regions[region].values / 1_000_000
    model = LinearRegression()
    model.fit(X_regions_poly, y_region)
    region_models[region] = model

# ======================================================================
# 2. BATTERY DEMAND + EV STOCK MODEL
# ======================================================================
df_ev_hist = pd.read_csv("global_ev_stock_2013_2023.csv")
df_battery_hist = pd.read_csv("Battery_Demand_by_Mode_2016_2022.csv")
df_ev_hist.rename(columns={'EV Stock (Million)': 'EV_Stock'}, inplace=True)
df_battery_hist.rename(columns={'Total (GWh)': 'Battery_Demand'}, inplace=True)
df_merged_hist = pd.merge(df_battery_hist, df_ev_hist, on="Year")
poly_ev_stock = PolynomialFeatures(degree=2)
X_ev_stock_poly = poly_ev_stock.fit_transform(df_ev_hist[['Year']])
ev_stock_model = LinearRegression()
ev_stock_model.fit(X_ev_stock_poly, df_ev_hist[['EV_Stock']])
poly_batt_demand = PolynomialFeatures(degree=2)
X_batt_demand_poly = poly_batt_demand.fit_transform(df_merged_hist[['Year']])
battery_demand_model = LinearRegression()
battery_demand_model.fit(X_batt_demand_poly, df_merged_hist[['Battery_Demand']])

# ======================================================================
# 3. HELPER FUNCTIONS
# ======================================================================
def create_region_plot(region):
    actual_y = df_regions[region].values / 1_000_000
    future_years = np.arange(df_regions['Year'].min(), 2031)
    future_X_poly = poly_regions.transform(future_years.reshape(-1, 1))
    pred_y = region_models[region].predict(future_X_poly)
    fig = go.Figure(data=[
        go.Scatter(x=df_regions['Year'], y=actual_y, mode='markers', name=f"{region} Actual"),
        go.Scatter(x=future_years, y=pred_y, mode='lines', name=f"{region} Predicted")
    ])
    fig.update_layout(title=f"{region} EV Sales Prediction (in millions)", xaxis_title="Year", yaxis_title="Sales (Millions)")
    return pio.to_html(fig, full_html=False)

def create_battery_plot(end_year):
    future_years = np.arange(2023, end_year + 1).reshape(-1, 1)
    X_future_ev_poly = poly_ev_stock.transform(future_years)
    y_future_ev = ev_stock_model.predict(X_future_ev_poly).flatten()
    X_future_batt_poly = poly_batt_demand.transform(future_years)
    y_future_batt = battery_demand_model.predict(X_future_batt_poly).flatten()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_merged_hist['Year'], y=df_merged_hist['Battery_Demand'], name='Actual Battery Demand', marker_color='steelblue'), secondary_y=False)
    fig.add_trace(go.Bar(x=future_years.flatten(), y=y_future_batt, name='Predicted Battery Demand', marker_color='lightcoral'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_ev_hist['Year'], y=df_ev_hist['EV_Stock'], name='Actual EV Stock', mode='lines+markers', marker_color='black'), secondary_y=True)
    fig.add_trace(go.Scatter(x=future_years.flatten(), y=y_future_ev, name='Predicted EV Stock', mode='lines', line=dict(dash='dash'), marker_color='green'), secondary_y=True)
    fig.update_layout(title_text=f"Battery Demand vs EV Stock Prediction (until {end_year})", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Battery Demand (GWh)", color='steelblue', secondary_y=False)
    fig.update_yaxes(title_text="EV Stock (Millions)", color='green', secondary_y=True)
    return fig.to_html(full_html=False)
# ===================================================================
# 2. INDIA EV SALES (using pre-trained model)
# ===================================================================
df_india = pd.read_csv('IND YEARWISE SALES - Sheet1.csv')
df_india.columns = df_india.columns.str.strip()
df_india.rename(columns={'YEAR': 'Year', 'TOTAL EV SALES': 'EV_Sales'}, inplace=True)
df_india['EV_Sales'] = pd.to_numeric(df_india['EV_Sales'], errors='coerce') / 1_000_000

# Load pre-trained model and transformer
model_india = joblib.load('ev_sales_poly_model.pkl')
poly_india = joblib.load('poly_transformer.pkl')

# ======================================================================
# 4. ROUTES
# ======================================================================
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/p')
def india_prediction_page():
    return render_template('index2.html')

@app.route("/gp", methods=["GET"])
def index():
    return render_template("index3.html")

@app.route("/globalprediction", methods=["POST"])
def global_predict():
    year = int(request.form['year'])
    year_poly = poly_regions.transform([[year]])
    prediction = {region: round(region_models[region].predict(year_poly)[0], 3) for region in regions}
    graph_html = {region: create_region_plot(region) for region in regions}
    return render_template("index3.html", prediction=prediction, year=year, graph_html=graph_html)

@app.route('/battery_Demand_trend/', methods=['GET', 'POST'])
def battery_demand_prediction():
    selected_year = 2040
    if request.method == 'POST':
        selected_year = int(request.form.get('prediction_year'))
    plot_html = create_battery_plot(end_year=selected_year)
    return render_template('index_battery_demand.html', plot_html=plot_html, selected_year=selected_year)

@app.route('/oil_vs_ev', methods=['GET', 'POST'])
def oil_ev_prediction():
    selected_year = 2040
    if request.method == 'POST':
        selected_year = int(request.form.get('prediction_year', 2040))
    return render_template('index_ev_market.html', selected_year=selected_year)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        if not 2010 <= year <= 2050:
            return render_template('index2.html', error="Enter year between 2010 and 2050.")

        start_year = df_india['Year'].min()
        end_year = year

        all_years = np.arange(start_year, end_year + 1).reshape(-1, 1)
        all_years_poly = poly_india.transform(all_years)
        predicted_sales = model_india.predict(all_years_poly) / 10
        predicted_value = round(predicted_sales[-1], 2)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_india['Year'],
            y=df_india['EV_Sales'],
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='royalblue', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=all_years.flatten(),
            y=predicted_sales,
            mode='lines',
            name='Prediction Line',
            line=dict(color='orange', width=3, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=[year],
            y=[predicted_value],
            mode='markers+text',
            name='Predicted',
            marker=dict(size=10, color='red'),
            text=[f'{predicted_value}'],
            textposition='top center'
        ))

        max_y = max(max(predicted_sales), df_india['EV_Sales'].max())
        y_range_top = max_y * 1.1

        fig.update_layout(
            title='EV Sales Prediction (India)',
            xaxis_title='Year',
            yaxis_title='EV Sales (in Millions)',
            plot_bgcolor='white',
            height=600,
            width=1100,
            margin=dict(l=80, r=80, t=80, b=80),
            hovermode='x unified',
            xaxis=dict(dtick=1, tickangle=-45, gridcolor='lightgrey'),
            yaxis=dict(gridcolor='lightgrey', tickformat=".2f", dtick=50, range=[0, y_range_top])
        )

        graph_html = fig.to_html(full_html=False)

        return render_template('index2.html',
                               prediction=predicted_value,
                               year=year,
                               map_div=graph_html)

    except Exception as e:
        print("ERROR:", e)
        return render_template('index2.html', error="Error creating plot")

if __name__ == '__main__':
    app.run(debug=True)
