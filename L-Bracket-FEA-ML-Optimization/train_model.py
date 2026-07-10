import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

bracket_data = pd.read_csv("bracket_training_data.csv")

features = bracket_data[["Thickness_mm", "Fillet_Radius_mm", "Hole_Diameter_mm"]]
target = bracket_data["Min_Safety_Factor"]

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size = 0.2, random_state = 42)

data_scaler = StandardScaler()
scaled_train_x = data_scaler.fit_transform(train_features)
scaled_test_x = data_scaler.transform(test_features)

print("Training the Multi-Layer Perceptron Regressor")
ann_model = MLPRegressor(hidden_layer_sizes = (10, 10), activation = "relu", max_iter = 2000, random_state = 42)
ann_model.fit(scaled_train_x, train_target)

predictions = ann_model.predict(scaled_test_x)
calculated_mae = mean_absolute_error(test_target, predictions)
calculated_r2 = r2_score(test_target, predictions)

print("\nMODEL PERFORMANCE RESULTS:")
print(f" -> Mean Absolute Error (MAE): {calculated_mae:.4f}")
print(f" -> R-squared (R2) Score: {calculated_r2:.4f}")

plt.figure(figsize = (6, 6))
plt.scatter(test_target, predictions, color = "blue", edgecolors = "k", alpha = 0.7, label = "Predicted vs Actual")

baseline_range = np.linspace(min(test_target) - 5, max(test_target) + 5, 100)
plt.plot(baseline_range, baseline_range, color = "red", linestyle = "--", label = "Perfect Accuracy")

plt.title("AI Safety Factor Predictions vs. Actual CAD Data")
plt.xlabel("Actual Safety Factor (Fusion 360)")
plt.ylabel("AI Predicted Safety Factor (Neural Network)")
plt.legend()
plt.grid(True)

plt.savefig("results_graph.png", dpi=300)
print("\nGraph successfully created and saved as 'results_graph.png'")
plt.show()
