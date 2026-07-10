import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('bracket_training_data.csv')

X = data[['Thickness_mm', 'Fillet_Radius_mm', 'Hole_Diameter_mm']]
y = data['Min_Safety_Factor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training the Multi-Layer Perceptron Regressor...")
model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', max_iter=2000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL PERFORMANCE RESULTS ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.7, label='Predicted vs Actual')

perfect_line = np.linspace(min(y_test)-0.5, max(y_test)+0.5, 100)
plt.plot(perfect_line, perfect_line, color='red', linestyle='--', label='Perfect Accuracy')

plt.title('AI Safety Factor Predictions vs. Actual CAD Simulation Data')
plt.xlabel('Actual Safety Factor (Fusion 360)')
plt.ylabel('AI Predicted Safety Factor (Neural Network)')
plt.legend()
plt.grid(True)

plt.savefig('results_graph.png', dpi=300)
print("\nGraph successfully created and saved as 'results_graph.png'!")
plt.show()
