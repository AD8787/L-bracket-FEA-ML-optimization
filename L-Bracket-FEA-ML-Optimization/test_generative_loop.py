import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

bracket_data = pd.read_csv("bracket_training_data.csv")

features = bracket_data[["Thickness_mm", "Fillet_Radius_mm", "Hole_Diameter_mm"]]
target = bracket_data["Min_Safety_Factor"]

data_scaler = StandardScaler()
scaled_features = data_scaler.fit_transform(features)

ann_model = MLPRegressor(hidden_layer_sizes = (10, 10), activation = "relu", max_iter = 2000, random_state = 42)
ann_model.fit(scaled_features, target)

abs_min_sf = 2.0
target_sf = 4.0
num_candidates = 10000
np.random.seed(42)

candidate_thickness = np.random.uniform(1.0, 10.0, num_candidates)
candidate_fillet = np.random.uniform(1.0, 5.0, num_candidates)
candidate_hole = np.random.uniform(1.0, 10.0, num_candidates)

candidates_df = pd.DataFrame({
    "Thickness_mm": candidate_thickness,
    "Fillet_Radius_mm": candidate_fillet,
    "Hole_Diameter_mm": candidate_hole
})

scaled_candidates = data_scaler.transform(candidates_df)
predicted_sf = ann_model.predict(scaled_candidates)
candidates_df["Predicted_Safety_Factor"] = predicted_sf

valid_designs = candidates_df[candidates_df["Predicted_Safety_Factor"] >= target_sf].copy()
optimal_design = valid_designs.sort_values(by = "Thickness_mm").iloc[0]

print("\nOptimal Bracket Design for Target Safety Factor of {target_sf}:")
print(f"- Absolute Minimum Safety Factor: {abs_min_sf}")
print(f"- Target Safety Factor: {target_sf}")
print(f"- Optimal Thickness: {optimal_design['Thickness_mm']:.2f} mm")
print(f"- Optimal Fillet Radius: {optimal_design['Fillet_Radius_mm']:.2f} mm")
print(f"- Optimal Hole Diameter: {optimal_design['Hole_Diameter_mm']:.2f} mm")
print(f"- Predicted Safety Factor: {optimal_design['Predicted_Safety_Factor']:.3f}")
