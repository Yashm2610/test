from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# ===================================================================
# 1. GLOBAL EV SALES (China, EU27, USA)
# ===================================================================
df_regions = pd.read_csv("ev_sales_yearwise_by_region.csv")
df_regions.columns = df_regions.columns.str.strip()
print("DEBUG: Global columns loaded:", df_regions.columns)

regions = ['China', 'EU27', 'USA']
region_models = {}
poly_regions = PolynomialFeatures(degree=3)
X_regions = df_regions[['Year']].values
X_regions_poly = poly_regions.fit_transform(X_regions)

for region in regions:
    y = df_regions[region].values / 1_000_000
    model = LinearRegression()
    model.fit(X_regions_poly, y)
    region_models[region] = model

def make_prediction(year):
    year_poly = poly_regions.transform([[year]])
    result = {}
    for region, model in region_models.items():
        pred = model.predict(year_poly)[0]
        result[region] = round(pred, 3)
    return result

def create_plot(region):
    actual_y = df_regions[region].values / 1_000_000
    future_years = np.arange(df_regions['Year'].min(), 2031)
    future_X = poly_regions.transform(future_years.reshape(-1, 1))
    pred_y = region_models[region].predict(future_X)

    traces = [
        go.Scatter(x=df_regions['Year'], y=actual_y, mode='markers', name=f"{region} Actual"),
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

# ===================================================================
# 3. FLASK ROUTES
# ===================================================================

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
    prediction = make_prediction(year)
    graph_html = {region: create_plot(region) for region in regions}
    return render_template("index3.html", prediction=prediction, year=year, graph_html=graph_html)

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

# ===================================================================

if __name__ == '__main__':
    app.run(debug=True)
