# app.py (Flask backend for EV Sales Predictor using pre-trained model)
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go

app = Flask(__name__)

# Load model, transformer, and dataset
model = joblib.load('ev_sales_poly_model.pkl')
poly = joblib.load('poly_transformer.pkl')
df = pd.read_csv('IND YEARWISE SALES - Sheet1.csv')
df.columns = df.columns.str.strip()
df['EV_Sales'] = pd.to_numeric(df['TOTAL EV SALES'], errors='coerce') / 1_000_000

@app.route('/')
def home():
    return render_template('index.html')




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

if __name__ == '__main__':
    app.run(debug=True)