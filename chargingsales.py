from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go

app = Flask(__name__)

# Load model and transformer
model = joblib.load('ev_charging_poly_model.pkl')
poly = joblib.load('poly_transformer.pkl')

# Load and clean dataset
df = pd.read_csv('chargingdata - Sheet1.csv')
df.columns = df.columns.str.strip()
df['CPTS'] = pd.to_numeric(df['CPTS'], errors='coerce') / 1_000_000  # Convert to millions

@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year_input = request.form.get('year')
        if not year_input or not year_input.isdigit():
            return render_template('index3.html', error="Enter a valid year.")

        year = int(year_input)
        if not 2010 <= year <= 2050:
            return render_template('index3.html', error="Year must be between 2010 and 2050.")

        # Prediction
        all_years = np.arange(df['YEAR'].min(), year + 1).reshape(-1, 1)
        all_years_poly = poly.transform(all_years)
        predicted_raw = model.predict(all_years_poly)

        # Convert prediction to millions to match actual data
        predicted_millions = predicted_raw / 1_000_000
        predicted_value = round(predicted_millions[-1], 2)

        # Plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['YEAR'],
            y=df['CPTS'],
            mode='lines+markers',
            name='Actual Data',
            line=dict(color='royalblue', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=all_years.flatten(),
            y=predicted_millions,
            mode='lines',
            name='Prediction',
            line=dict(color='orange', dash='dash', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=[year],
            y=[predicted_value],
            mode='markers+text',
            name='Predicted',
            marker=dict(size=10, color='red'),
            text=[f"{predicted_value:.2f}M"],
            textposition='top center',
            hovertemplate='Year: %{x}<br>Prediction: %{y:.2f}M'
        ))

        max_y = max(max(predicted_millions), df['CPTS'].max())

        fig.update_layout(
            title='EV Charging Points Forecast',
            xaxis_title='Year',
            yaxis_title='Charging Points (Millions)',
            plot_bgcolor='white',
            height=600,
            width=1100,
            margin=dict(t=40, b=40, l=60, r=60),
            hovermode='x unified',
            xaxis=dict(dtick=1, tickangle=-45, gridcolor='lightgrey'),
            yaxis=dict(range=[0, max_y * 1.1], gridcolor='lightgrey', tickformat=".2f")
        )

        graph_html = fig.to_html(full_html=False)

        return render_template("index3.html",
                               prediction=predicted_value,
                               year=year,
                               map_div=graph_html)

    except Exception as e:
        print("ERROR:", e)
        return render_template("index3.html", error="Error creating plot")

if __name__ == '__main__':
    app.run(debug=True ,port=5001)
