# Execution Plan

## Settings I will use

- **Area:** West Pacific corridor — latitude 5–40, longitude 110–145
- **Data years:** 2010–2025
- **Training years:** 2010–2019
- **Testing years:** 2020–2025
- **Prediction window:** next 24 hours
- **Outcomes:**
  - Rapid intensification: wind change **≥ +30 kt**
  - Rapid weakening: wind change **≤ −30 kt**
  - No rapid change: wind change between −29 and +29 kt

## Step 1 — Set up the repository

- [ ] Create the public repository `cyclone-ri-warning`.
- [ ] Add the MIT License.
- [ ] Upload this `PLAN.md`.
- [ ] Create these folders as I reach each stage:
  - `notebooks/`
  - `data/processed/`
  - `web/`
  - `writeup/`

## Step 2 — Create the notebook

- [ ] Open Kaggle using my personal Google account.
- [ ] Create a notebook named `cyclone_ri_warning.ipynb`.
- [ ] Add the project title and this one-sentence description:

> An AI system that predicts whether an existing tropical cyclone will rapidly intensify, rapidly weaken, or see no rapid change during the next 24 hours, presented through an interactive 3D globe for the Philippines/West Pacific corridor.

- [ ] Save the notebook before running any code.

## Step 3 — Download and load the data

- [ ] Download the IBTrACS since-1980 CSV from NOAA NCEI:

```text
https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv
```

- [ ] Load it with pandas.
- [ ] Skip the units row directly below the header.
- [ ] Use `keep_default_na=False` so the basin code `NA` is not mistaken for missing data.
- [ ] Print the shape, column names, first rows, and missing-value counts.
- [ ] Do not continue until the file loads without errors.

## Step 4 — Filter and clean the storm records

- [ ] Keep only records where `TRACK_TYPE` is `main`.
- [ ] Keep only the `WP` basin.
- [ ] Keep records from 2010 onwards.
- [ ] Keep only points inside latitude 5–40 and longitude 110–145.
- [ ] Convert latitude, longitude, wind, pressure, and distance-to-land columns to numbers.
- [ ] Convert `ISO_TIME` to a datetime column.
- [ ] Sort the records by storm ID and time.
- [ ] Print the number of remaining records and storms after every filter.

## Step 5 — Build one wind and one pressure column

- [ ] Create a unified `WIND` column by taking the first available value across the agency wind columns, starting with the official WMO value and then the other reporting agencies.
- [ ] Create a unified `PRES` column the same way from the agency pressure columns.
- [ ] Print how many records still have no wind value.
- [ ] Drop records without a valid wind value or time.
- [ ] Save this cleaned table before creating the prediction labels.

## Step 6 — Create the features and the answer

For every storm record:

- [ ] Find the same storm exactly 6, 12, and 24 hours earlier.
- [ ] Find the same storm exactly 24 hours later.
- [ ] Use an exact time match; do not assume that every row is always the same number of hours apart.
- [ ] Calculate:
  - current wind;
  - current pressure;
  - wind change over the last 6 hours;
  - wind change over the last 12 hours;
  - wind change over the last 24 hours;
  - pressure change over the last 6 hours;
  - latitude;
  - longitude;
  - distance to land;
  - month;
  - storm movement speed from the position 6 hours earlier.
- [ ] Calculate the answer from the wind 24 hours later:

```python
delta_24h = wind_24h_later - wind_now
```

- [ ] Assign the outcome:
  - `RI` if `delta_24h >= 30`;
  - `WEAKEN` if `delta_24h <= -30`;
  - `STEADY` otherwise.
- [ ] Drop rows that do not have both past and future records.
- [ ] Print the number and percentage of rows in each outcome class.
- [ ] Save the finished modelling table as `data/processed/corridor_labelled.csv`.

## Step 7 — Split the training and testing data

- [ ] Put records from 2010–2019 in the training set.
- [ ] Put records from 2020–2025 in the testing set.
- [ ] Do not shuffle the rows into a random split.
- [ ] Print the number of training and testing rows.
- [ ] Print the outcome counts for both sets.

## Step 8 — Run the simple forecasts first

- [ ] Run a forecast that always predicts `STEADY`.
- [ ] Run a climatology forecast using the outcome frequencies from the training years.
- [ ] Run a persistence rule based on how much the storm already changed during the previous 24 hours.
- [ ] For each simple forecast, record:
  - the confusion matrix;
  - recall for `RI`;
  - recall for `WEAKEN`;
  - precision for `RI`;
  - precision for `WEAKEN`;
  - the number of missed rapid changes;
  - the number of false alarms.

## Step 9 — Train the main models

- [ ] Scale the numeric features using values calculated from the training set only.
- [ ] Train multinomial logistic regression with balanced class weights.
- [ ] Train a random forest with balanced class weights as a comparison.
- [ ] Use both models on the same 2020–2025 testing set.
- [ ] Record the same metrics used for the simple forecasts.
- [ ] Compare the models by rapid-change recall and false alarms, not by overall accuracy alone.
- [ ] Inspect the results for Goni, Rai, Doksuri, and Yagi when those storms appear in the testing data.
- [ ] Write down at least three cases the model got wrong and one possible reason for each.

## Step 10 — Build the from-scratch comparison

- [ ] Implement multinomial logistic regression using NumPy.
- [ ] Use one weight vector for each of the three outcomes.
- [ ] Train it with gradient descent on the same training features.
- [ ] Apply the softmax calculation to produce the three probabilities.
- [ ] Compare its testing results with scikit-learn's logistic regression.
- [ ] Record whether the two implementations produce similar predictions.

## Step 11 — Add sea-surface temperature

- [ ] Get sea-surface temperature for each storm record's location and date from a public NOAA SST source.
- [ ] Match the temperature to the nearest available location and time.
- [ ] If the full date-by-date match is not practical, use the NOAA climatological temperature for that location and month and label it as climatology in the notebook.
- [ ] Add the temperature as a new feature.
- [ ] Train the models again.
- [ ] Compare the results with and without sea-surface temperature using the same testing years.
- [ ] Keep the temperature feature only if the comparison and its limitations are recorded honestly.

## Step 12 — Choose the warning settings

- [ ] Examine the predicted probabilities instead of only the final class labels.
- [ ] Compare different probability thresholds for issuing an `RI` warning.
- [ ] Compare different probability thresholds for issuing a `WEAKEN` warning.
- [ ] Choose settings that catch a high proportion of rapid changes without producing an unusable number of false alarms.
- [ ] Record the chosen thresholds and the reason for choosing them.

## Step 13 — Export the model for the globe

- [ ] Export these items to `web/model_weights.json`:
  - feature names in the exact order used by the model;
  - scaler centre and scale for every feature;
  - one coefficient list for each outcome;
  - one intercept for each outcome;
  - the chosen warning thresholds.
- [ ] Select three testing records.
- [ ] Record the probabilities Python gives for those records.
- [ ] Keep those examples to check the JavaScript calculation later.

## Step 14 — Prepare the globe's location data

- [ ] Divide the corridor into 5° × 5° latitude/longitude cells.
- [ ] For each cell, calculate:
  - how many storms passed through it since 2010;
  - how many storm records it contains;
  - the percentage that rapidly intensified;
  - the percentage that rapidly weakened.
- [ ] Export the cell results to `web/region_stats.json`.
- [ ] Check several cells against the original data before using the file.

## Step 15 — Build the interactive 3D globe

- [ ] Create `web/index.html` and `web/app.js`.
- [ ] Load Three.js / Globe.gl.
- [ ] Display the West Pacific region and the historical storm cells.
- [ ] Let the user click a location on the globe.
- [ ] Show the selected latitude and longitude.
- [ ] Load that location's historical statistics from `region_stats.json`.
- [ ] Add controls for:
  - current wind;
  - current pressure;
  - wind change over the last 6, 12, and 24 hours;
  - pressure change over the last 6 hours;
  - sea-surface temperature;
  - distance to land;
  - storm movement speed;
  - month.
- [ ] Add a separate conditions checklist for:
  - atmospheric instability;
  - humidity;
  - vertical wind shear.
- [ ] Label the checklist as information shown beside the prediction; do not use checklist values in the model calculation unless they were also used during training.
- [ ] Load `model_weights.json`.
- [ ] Standardise the user's values using the exported scaler values.
- [ ] Calculate the three outcome scores in JavaScript.
- [ ] Apply softmax to convert the scores into probabilities.
- [ ] Display:
  - rapid-intensification probability;
  - rapid-weakening probability;
  - no-rapid-change probability;
  - the most likely outcome;
  - the six-condition panel;
  - the location's historical storm context.
- [ ] Test the globe with the three saved Python examples.
- [ ] Do not continue until the JavaScript probabilities match the Python probabilities.

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
