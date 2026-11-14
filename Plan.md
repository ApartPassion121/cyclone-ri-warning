# Execution Plan

## Settings I will use

- **Area:** West Pacific corridor — latitude 5–40, longitude 110–145
- **Data years:** 2000–2025
- **Training years:** 2000–2020
- **Testing years:** 2021–2025
- **Prediction window:** next 24 hours
- **Outcomes:**
  - Rapid intensification: wind change **≥ +30 kt**
  - Rapid weakening: wind change **≤ −30 kt**
  - No rapid change: wind change between −29 and +29 kt

## Step 1 — Set up the repository

- [x] Create the public repository `cyclone-ri-warning`.
- [x] Add the MIT License.
- [x] Upload this `PLAN.md`.
- [x] Keep the web files (`index.html` and the JSON exports) at the repository root so GitHub Pages can serve them.

## Step 2 — Create the notebook

- [x] Open Kaggle and create a new notebook named `cyclone_ri_warning`.
- [x] In the notebook settings, turn Internet on and leave the accelerator set to None (this project runs on CPU).
- [x] Add the project title and this one-sentence description:

> An AI system that predicts whether an existing tropical cyclone will rapidly intensify, rapidly weaken, or see no rapid change during the next 24 hours, presented through an interactive 3D globe for the Philippines/West Pacific corridor.

- [x] Save the notebook before running any code.

## Step 3 — Download and load the data

- [x] Download the full IBTrACS archive CSV (records since 1842) from NOAA NCEI:

```text
https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv
```

- [x] Load it with pandas.
- [x] Skip the units row directly below the header.
- [x] Use `keep_default_na=False` so the basin code `NA` is not mistaken for missing data.
- [x] Print the shape, column names, first rows, and missing-value counts.
- [x] Do not continue until the file loads without errors.

## Step 4 — Filter and clean the storm records

- [x] Keep only records where `TRACK_TYPE` is `main`.
- [x] Keep only the `WP` basin.
- [x] Keep records from 2000 onwards.
- [x] Keep only points inside latitude 5–40 and longitude 110–145.
- [x] Convert latitude, longitude, wind, pressure, and distance-to-land columns to numbers.
- [x] Convert `ISO_TIME` to a datetime column.
- [x] Sort the records by storm ID and time.
- [x] Print the number of remaining records and storms after every filter.

## Step 5 — Build one wind and one pressure column

- [x] Create a unified `WIND` column by taking the first available value across the agency wind columns, starting with the official WMO value and then the other reporting agencies.
- [x] Create a unified `PRES` column the same way from the agency pressure columns.
- [x] Print how many records still have no wind value.
- [x] Drop records without a valid wind value or time.
- [x] Save this cleaned table before creating the prediction labels.

## Step 6 — Create the features and the answer

For every storm record:

- [x] Find the same storm exactly 6, 12, and 24 hours earlier.
- [x] Find the same storm exactly 24 hours later.
- [x] Use an exact time match; do not assume that every row is always the same number of hours apart.
- [x] Calculate:
  - current wind;
  - current pressure;
  - wind change over the last 6 hours;
  - wind change over the last 12 hours;
  - wind change over the last 24 hours;
  - pressure change over the last 6 hours;
  - pressure change over the last 24 hours;
  - latitude;
  - longitude;
  - distance to land;
  - month;
  - storm movement speed from the position 6 hours earlier.
- [x] Calculate the answer from the wind 24 hours later:

```python
delta_24h = wind_24h_later - wind_now
```

- [x] Assign the outcome:
  - `RI` if `delta_24h >= 30`;
  - `WEAKEN` if `delta_24h <= -30`;
  - `STEADY` otherwise.
- [x] Drop rows that do not have both past and future records.
- [x] Print the number and percentage of rows in each outcome class.
- [x] Save the finished modelling table as `data/processed/corridor_labelled.csv`.

## Step 7 — Split the training and testing data

- [x] Put records from 2000–2020 in the training set.
- [x] Put records from 2021–2025 in the testing set.
- [x] Do not shuffle the rows into a random split.
- [x] Print the number of training and testing rows.
- [x] Print the outcome counts for both sets.

## Step 8 — Run the simple forecasts first

- [x] Run a forecast that always predicts `STEADY`.
- [x] Run a climatology forecast using the outcome frequencies from the training years.
- [x] Run a persistence rule based on how much the storm already changed during the previous 24 hours.
- [x] For each simple forecast, record:
  - the confusion matrix;
  - recall for `RI`;
  - recall for `WEAKEN`;
  - precision for `RI`;
  - precision for `WEAKEN`;
  - the number of missed rapid changes;
  - the number of false alarms.

## Step 9 — Train the main models

- [x] Scale the numeric features using values calculated from the training set only.
- [x] Train multinomial logistic regression with balanced class weights.
- [x] Train a random forest with balanced class weights as a comparison.
- [x] Use both models on the same 2021–2025 testing set.
- [x] Record the same metrics used for the simple forecasts.
- [x] Compare the models by rapid-change recall and false alarms, not by overall accuracy alone.
- [x] Inspect the results for Rai, Doksuri, Mawar, and Yagi when those storms appear in the testing data.
- [x] Write down at least three cases the model got wrong and one possible reason for each.

## Step 10 — Build the from-scratch comparison

- [x] Implement multinomial logistic regression using NumPy.
- [x] Use one weight vector for each of the three outcomes.
- [x] Train it with gradient descent on the same training features.
- [x] Apply the softmax calculation to produce the three probabilities.
- [x] Compare its testing results with scikit-learn's logistic regression.
- [x] Record whether the two implementations produce similar predictions.

## Step 11 — Add sea-surface temperature

- [x] Get sea-surface temperature from NOAA OISST through the CoastWatch ERDDAP griddap service (monthly snapshots, 2000–2025, corridor grid).
- [x] Match each storm record to the nearest SST value in time, latitude, and longitude.
- [x] Fill the remaining gaps (coastal cells the satellite grid masks out) with the training-set median SST, 28.6 °C, and record how many points were filled.
- [x] Add the temperature as a new feature.
- [x] Train the models again.
- [x] Compare the results with and without sea-surface temperature using the same testing years.
- [x] Keep the temperature feature only if the comparison and its limitations are recorded honestly.

## Step 12 — Choose the warning settings

- [x] Examine the predicted probabilities instead of only the final class labels.
- [x] Compare different probability thresholds for issuing an `RI` warning.
- [x] Compare different probability thresholds for issuing a `WEAKEN` warning.
- [x] Choose settings that catch a high proportion of rapid changes without producing an unusable number of false alarms.
- [x] Record the chosen thresholds and the reason for choosing them.
- Chosen: warning threshold **0.25** for both RI and WEAKEN warnings (from the sweep: at 0.25 the model catches 92.9% of RI and 89.9% of WEAKEN; lower thresholds bought almost nothing and cost hundreds more false alarms). A warning means *prepare*, not *evacuate*; the globe still shows the raw probabilities.

## Step 13 — Export the model for the globe

- [x] Export these items to `model_weights.json` at the repository root:
  - feature names in the exact order used by the model;
  - scaler centre and scale for every feature;
  - one coefficient list for each outcome;
  - one intercept for each outcome;
  - the chosen warning thresholds.
- [x] Select three testing records.
- [x] Record the probabilities Python gives for those records.
- [x] Keep those examples to check the JavaScript calculation later.

## Step 14 — Prepare the globe's location data

- [x] Divide the corridor into 2.5° × 2.5° latitude/longitude cells.
- [x] For each cell, calculate:
  - how many storms passed through it since 2000;
  - how many storm records it contains;
  - the percentage that rapidly intensified;
  - the percentage that rapidly weakened.
- [x] Export the cell results to `region_stats.json` at the repository root.
- [x] Also export `region_defaults.json`: for each cell and month, the median value of every model feature — the globe uses these to pre-fill the input form for the clicked location (every value can be overridden by the user).
- [x] Check several cells against the original data before using the files.

## Step 15 — Build the interactive 3D globe

- [x] Create a single `index.html` at the repository root (served by GitHub Pages) with the three JSON files beside it.
- [x] Load Three.js / Globe.gl.
- [x] Display the West Pacific region and the historical storm cells, Nullschool-style: dark globe, a toggleable coloured data layer (historical RI rate, or typical SST for the selected month) with a colour scale bar.
- [x] Let the user click a location on the globe.
- [x] Show the selected latitude and longitude.
- [x] Load that location's historical statistics from `region_stats.json`.
- [x] Add controls for:
  - current wind;
  - current pressure;
  - wind change over the last 6, 12, and 24 hours;
  - pressure change over the last 6 hours;
  - pressure change over the last 24 hours;
  - sea-surface temperature;
  - distance to land;
  - storm movement speed;
  - month.
- [x] Add a separate conditions checklist for:
  - atmospheric instability;
  - humidity;
  - vertical wind shear.
- [x] Label the checklist as information shown beside the prediction; do not use checklist values in the model calculation unless they were also used during training.
- [x] Load `model_weights.json`.
- [x] Standardise the user's values using the exported scaler values.
- [x] Calculate the three outcome scores in JavaScript.
- [x] Apply softmax to convert the scores into probabilities.
- [x] Display:
  - rapid-intensification probability;
  - rapid-weakening probability;
  - no-rapid-change probability;
  - the most likely outcome;
  - the six-condition panel;
  - the location's historical storm context.
- [x] Test the globe with the three saved Python examples.
- [x] Do not continue until the JavaScript probabilities match the Python probabilities.

## Step 16 — Test the complete product

- [ ] Run the notebook from the beginning without errors.
- [ ] Open the globe locally and test location selection.
- [ ] Test one likely intensification example.
- [ ] Test one likely weakening example.
- [ ] Test one likely steady example.
- [ ] Check the layout on a desktop screen and a phone-sized screen.
- [ ] Fix errors before recording the demonstration.

## Step 17 — Finish the repository documentation

- [ ] Update `README.md` with:
  - what the project does;
  - how to run the notebook;
  - how to open the globe;
  - the data source;
  - the main results;
  - the limitations.
- [ ] Add the IBTrACS citation and DOI to the notebook and README.
- [ ] Add citations for every library and dataset used.
- [ ] Commit the notebook, processed-data documentation, model weights, region statistics, and web files.

## Step 18 — Write the project report

- [ ] Write the problem.
- [ ] Write what I built.
- [ ] Write how the model works.
- [ ] Write the data and train/test split.
- [ ] Write the baseline and model results.
- [ ] Include missed cases and false alarms.
- [ ] Write the limitations.
- [ ] Write what I would improve next.
- [ ] List all sources.
- [ ] Add the required AI-assistance disclosure.

## Step 19 — Record and submit

- [ ] Record a demonstration of two minutes or less.
- [ ] Show the globe, location selection, factor inputs, and three probabilities.
- [ ] Show one notebook result table.
- [ ] Make the GitHub repository public.
- [ ] Test the public repository and globe links while signed out.
- [ ] Submit the repository link, write-up, and demonstration video.
- [ ] Save the submission confirmation.
