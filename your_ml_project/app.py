# FILE: your_ml_project/app.py (Complete and Corrected)

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

app = Flask(__name__)



# Load and clean the dataset
df = pd.read_csv("ev_sales_yearwise_by_region.csv")
df.columns = df.columns.str.strip()
print("DEBUG: Columns loaded:", df.columns)

# Prepare models per region
regions = ['China', 'EU27', 'USA']
models = {}
poly = PolynomialFeatures(degree=3)
X = df[['Year']].values
X_poly = poly.fit_transform(X)

# ==============================================================================
#  1. LOAD DATA & TRAIN ALL MODELS AT STARTUP
# ==============================================================================

# --- Models for Global Region Prediction ---
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

# --- Models for Battery Demand & EV Stock Prediction ---
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

# ==============================================================================
#  2. HELPER FUNCTIONS FOR PLOTTING
# ==============================================================================

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

# ==============================================================================
#  3. FLASK ROUTES
# ==============================================================================



for region in regions:
    y = df[region].values / 1_000_000  # convert to millions
    model = LinearRegression()
    model.fit(X_poly, y)
    models[region] = model

def make_prediction(year):
    year_poly = poly.transform([[year]])
    result = {}
    for region, model in models.items():
        pred = model.predict(year_poly)[0]
        result[region] = pred
    return result

def create_plot(region):
    actual_y = df[region].values / 1_000_000
    future_years = np.arange(df['Year'].min(), 2031)
    future_X = poly.transform(future_years.reshape(-1, 1))
    pred_y = models[region].predict(future_X)

    traces = [
        go.Scatter(x=df['Year'], y=actual_y, mode='markers', name=f"{region} Actual"),
        go.Scatter(x=future_years, y=pred_y, mode='lines', name=f"{region} Predicted")
    ]

    layout = go.Layout(
        title=f"{region} EV Sales Prediction (in millions)",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Sales (Millions)"),
        legend=dict(x=0, y=1.2, orientation="h")
    )
    fig = go.Figure(data=traces, layout=layout)
    return pio.to_html(fig, full_html=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/gp", methods=["GET"])
def index():
    return render_template("index3.html")

@app.route("/globalprediction", methods=["POST"])
def global_predict():
    year = int(request.form['year'])
    prediction = make_prediction(year)
    graph_html = {region: create_plot(region) for region in regions}
    return render_template("index3.html", prediction=prediction, year=year, graph_html=graph_html)




@app.route('/p')
def india_prediction_page():  # ✅ New function name
    return render_template('index2.html')





@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        if not 2010 <= year <= 2050:
            return render_template('index2.html', error="Enter year between 2010 and 2050.")

        start_year = df['YEAR'].min()
        end_year = year

        all_years = np.arange(start_year, end_year + 1).reshape(-1, 1)
        all_years_poly = poly.transform(all_years)
        predicted_sales = model.predict(all_years_poly)/10 
        predicted_value = round(predicted_sales[-1], 2)

        fig = go.Figure()

        # Actual sales line
        fig.add_trace(go.Scatter(
            x=df['YEAR'],
            y=df['EV_Sales'],
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='royalblue', width=3)
        ))

        # Full prediction line from start year
        fig.add_trace(go.Scatter(
            x=all_years.flatten(),
            y=predicted_sales,
            mode='lines',
            name='Prediction Line',
            line=dict(color='orange', width=3, dash='dash')
        ))

        # Highlight predicted point
        fig.add_trace(go.Scatter(
            x=[year],
            y=[predicted_value],
            mode='markers+text',
            name='Predicted',
            marker=dict(size=10, color='red'),
            text=[f'{predicted_value}'],
            textposition='top center'
        ))

        max_y = max(max(predicted_sales), df['EV_Sales'].max())
        y_range_top = max_y * 1.1

        fig.update_layout(
            title='EV Sales Prediction',
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



# THIS IS THE CORRECTED FUNCTION. REPLACE YOUR OLD ONE WITH THIS.

@app.route('/battery_Demand_trend/', methods=['GET', 'POST'])
def battery_demand_prediction():
    # Set a default year. This will be used for the initial page load.
    selected_year = 2040
    
    # If the user submitted the form (a POST request), 
    # then update the year with the value from the dropdown.
    if request.method == 'POST':
        selected_year = int(request.form.get('prediction_year'))
        
    # NOW, create the plot using whatever 'selected_year' is.
    # This line is now OUTSIDE the 'if' block, so it always runs.
    plot_html = create_battery_plot(end_year=selected_year)
    
    # Finally, render the page with the plot and the correct selected year for the dropdown.
    return render_template('index_battery_demand.html', plot_html=plot_html, selected_year=selected_year)

# --- ADD THIS NEW ROUTE TO YOUR app.py ---

@app.route('/oil_vs_ev', methods=['GET', 'POST'])
def oil_ev_prediction():
    selected_year = 2040  # Default value
    
    if request.method == 'POST':
        # Get the year from the form if it was submitted
        selected_year = int(request.form.get('prediction_year', 2040))
        
    
    # Render the template with the plot and the selected year
    return render_template('index_ev_market.html', selected_year=selected_year)

# --- END OF NEW ROUTE ---



if __name__ == '__main__':
    app.run(debug=True)