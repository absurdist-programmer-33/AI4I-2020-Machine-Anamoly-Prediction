# Smart Machine Anomaly Detector — LOF 

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Open the local URL Streamlit prints in the terminal.

## Dataset
**AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## What the demo does
- Uses five numerical sensor columns: air temperature, process temperature, rotational speed, torque, and tool wear.
- Lets you enter a sensor reading and get an anomaly flag and score.

## Important limitations
- An anomaly is not the same thing as a failure. LOF detects local-density deviations; it does not predict time-to-failure.
- Results depend on feature choice, scaling, `n_neighbors`, `contamination`, and the dataset split. Tune parameters on validation data and report the method honestly.
