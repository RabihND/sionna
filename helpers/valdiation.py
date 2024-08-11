# plot amp_vs_dist.json :data = {"distance": distance, "mean_power_db": mean_power_db, "regression_line": regression_line}


import numpy as np
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# import the file json
import json

# Load the JSON file
with open('amp_vs_dist.json', 'r') as file:
    data = json.load(file)

# Extract the data
distance = data['distance']
mean_power_db = data['mean_power_db']
regression_line = data['regression_line']


## Calcualte the equation of the regression line
# y = mx + b
coefficients = np.polyfit(distance, mean_power_db, 1)
polynomial = np.poly1d(coefficients)
regression_line = polynomial(distance)

m1 = coefficients[0]
b1 = coefficients[1]
equation = f"y = {m1:.2f}x + {b1:.2f}"

print(f"Regression line equation: {equation}")



# load the data from sionna josn
with open('amp_vs_dist_s.json', 'r') as file:
    data_s = json.load(file)

# Extract the data
distance_s = data_s['distance']
mean_power_db_s = data_s['mean_power_db']

# the mean_power_db_s is somthing like that: "mean_power_db": [[-46.400718688964844], [-45.20615005493164], i want to be like that: "mean_power_db": [-46.400718688964844, -45.20615005493164]
mean_power_db_s = [item for sublist in mean_power_db_s for item in sublist]


#calculate the regression line
# Calculate linear regression
coefficients = np.polyfit(distance_s, mean_power_db_s, 1)
polynomial = np.poly1d(coefficients)
regression_line_s = polynomial(distance_s)

# shift all the value _s with 100 to be able to plot the 2 data in the same figure
shifted = 61.49+39.85
mean_power_db_s = [x + shifted for x in mean_power_db_s]
regression_line_s = [x + shifted for x in regression_line_s]


## Calcualte the equation of the regression line
# y = mx + b
m2 = coefficients[0]
b2 = coefficients[1]
equation = f"y = {m2:.2f}x + {b2:.2f}"
print(f"Regression line equation: {equation}")




# Some mesurement to show the similarity between the 2 regression lines

# Calculate the correlation coefficient
correlation = np.corrcoef(regression_line, regression_line_s)
print(f"Correlation Coefficient: {correlation[0, 1]*100:.2f}%")

# Caalcualte the difference between the 2 regression line splopes in percentage
slope_diff = (abs(m1 - m2) / m1) * 100
print(f"Slope Difference: {slope_diff:.2f}%")




# # Plotly plotting
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=distance_s, y=mean_power_db_s, mode='markers+lines', name='Puissance Moyenne', line=dict(dash='dash',width=4), marker=dict(size=10)))
# fig.add_trace(go.Scatter(x=distance_s, y=regression_line_s, mode='lines', name='Ligne de Régression',line=dict(width=4)))
# fig.update_layout(title='Puissance Moyenne du Canal en dB',
#                 xaxis_title='Distance [m]',
#                 yaxis_title='Puissance Moyenne [dB]',
#                 width=1920,
#                 height=1080,font=dict(size=30))

# fig.show()
# Create subplots: 1 row, 2 columns
# fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Data", "Sionna Data"))

# # Add traces to the first subplot
# fig.add_trace(go.Scatter(x=distance, y=mean_power_db, mode='markers+lines', name='Puissance Moyenne', line=dict(dash='dash', width=4), marker=dict(size=10)), row=1, col=1)
# fig.add_trace(go.Scatter(x=distance, y=regression_line, mode='lines', name='Ligne de Régression', line=dict(width=4)), row=1, col=1)

# # Add traces to the second subplot
# fig.add_trace(go.Scatter(x=distance_s, y=mean_power_db_s, mode='markers+lines', name='Puissance Moyenne Sionna', line=dict(dash='dash', width=4), marker=dict(size=10)), row=1, col=2)
# fig.add_trace(go.Scatter(x=distance_s, y=regression_line_s, mode='lines', name='Ligne de Régression Sionna', line=dict(width=4)), row=1, col=2)

# # Update layout
# fig.update_layout(title='Puissance Moyenne du Canal en dB',
#                   xaxis_title='Distance [m]',
#                   yaxis_title='Puissance Moyenne [dB]',
#                   width=1920,
#                   height=1080,
#                   font=dict(size=30))


# sam figure with the 2 data

fig = go.Figure()
fig.add_trace(go.Scatter(x=distance, y=mean_power_db, mode='markers+lines', name='Expérimentale', line=dict(dash='dash',width=4), marker=dict(size=10)))
fig.add_trace(go.Scatter(x=distance, y=regression_line, mode='lines', name='Ligne de Régression Exp.',line=dict(width=4)))
fig.add_trace(go.Scatter(x=distance_s, y=mean_power_db_s, mode='markers+lines', name='Simulée', line=dict(dash='dash',width=4), marker=dict(size=10)))
fig.add_trace(go.Scatter(x=distance_s, y=regression_line_s, mode='lines', name='Ligne de Régression Sim. ',line=dict(width=4)))

# Add annotations for the slopes
mid_index = len(distance) // 2
fig.add_annotation(x=distance[mid_index], y=regression_line[mid_index], text=f"Pente = {m1:0.2f}", showarrow=True, arrowhead=2, ax=200, ay=-60, arrowwidth=2)
fig.add_annotation(x=distance_s[mid_index], y=regression_line_s[mid_index], text=f"Pente = {m2:0.2f}", showarrow=True, arrowhead=2, ax=200, ay=-100,arrowwidth=2)

fig.update_layout(title='Comparaison des Puissances Moyennes du Canal en dB',
                xaxis_title='Distance [m]',
                yaxis_title='Puissance Moyenne [dB]',
                width=1920,
                height=1080,font=dict(size=30))


fig.show()