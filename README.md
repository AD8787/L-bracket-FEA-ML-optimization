# L-bracket-FEA-ML-optimization
Independent Project for the UC Santa Cruz AIEA Lab Internship

## Project Structure
  - Week 4:
    * `bracket_training_data.csv`: Custom dataset containing 20 simulation runs from Autodesk Fusion.
    * `train_model.py`: Python script that creates and trains the model. 
    * `results_graph.png`: The image generated accuracy scatter plot.
      
  - Week 6:
    * `test_generative_loop.py`: Creates generative loop in Python to optimize variables (the inputs below and the target output of the safety factor) in the L-bracket using spreadsheet data.

## Core Features
* **Inputs:** Bracket Thickness (mm), Fillet Radius (mm), and Hole Diameter (mm).
* **Target Output:** Minimum Safety Factor.

## How to Run
1. Install essential modules using Terminal: `pip install pandas numpy scikit-learn matplotlib`
2. Run the script: `python train_model.py`
