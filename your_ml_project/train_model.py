import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import plotly.graph_objs as go

# Load data
df = pd.read_csv("IND YEARWISE SALES - Sheet1.csv")
df.columns = df.columns.str.strip()

# Convert columns to numeric
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['TOTAL EV SALES'] = pd.to_numeric(df['TOTAL EV SALES'], errors='coerce')
df.dropna(subset=['YEAR', 'TOTAL EV SALES'], inplace=True)

# Convert sales to millions
df['Sales_lakhs'] = df['TOTAL EV SALES'] / 1_000_000


# Prepare features and target
x = df[['YEAR']]
y = df['Sales_lakhs']

# Polynomial transformation (degree = 3)
degree = 3
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(x)

# Train model
model = LinearRegression()
model.fit(x_poly, y)

# Save model and transformer
joblib.dump(model, 'ev_sales_poly_model.pkl')
joblib.dump(poly, 'poly_transformer.pkl')

# Predictions on training data
y_pred = model.predict(x_poly)

# Accuracy metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Predict future years
future_years = np.arange(df['YEAR'].min(), 2071).reshape(-1, 1)
# FIX: Wrap in DataFrame to preserve feature name
future_years_df = pd.DataFrame(future_years, columns=['YEAR'])
future_x_poly = poly.transform(future_years_df)
future_preds = model.predict(future_x_poly)

# Confidence intervals
residuals = y - y_pred
std_error = np.std(residuals)
upper = future_preds + 1.96 * std_error
lower = future_preds - 1.96 * std_error

# Interactive Plotly Plot (with "boxed" look)
trace_actual = go.Scatter(x=df['YEAR'], y=y, mode='markers+lines', name='Actual Sales',
                          line=dict(color='blue'), marker=dict(size=6))
trace_predicted = go.Scatter(x=future_years.ravel(), y=future_preds, mode='lines', name='Predicted Sales',
                             line=dict(color='orange', dash='dash'))
trace_ci = go.Scatter(x=np.concatenate([future_years.ravel(), future_years.ravel()[::-1]]),
                      y=np.concatenate([upper, lower[::-1]]),
                      fill='toself', fillcolor='rgba(255,165,0,0.2)',
                      line=dict(color='rgba(255,255,255,0)'),
                      hoverinfo="skip", showlegend=True, name='95% Confidence Interval')

layout = go.Layout(
    title='EV Sales in India with Prediction (Interactive)',
    xaxis=dict(title='Year', showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='lightgrey'),
    yaxis=dict(title='EV Sales (in million)', showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='lightgrey'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='closest',
    template='plotly_white'
)

fig = go.Figure(data=[trace_actual, trace_predicted, trace_ci], layout=layout)
fig.show()
