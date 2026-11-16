# Execution Plan

## Settings I will use

- **Area:** North Pacific — latitude 5–40, longitude 110 °E eastward to 70 °W, split into two models at the dateline: west 110–180 °E, east 180 °E–70 °W. The steps below were written for the original 110–145 °E corridor they were carried out in; Step 23 records the widening and the readings that changed with it.
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
- Chosen: warning threshold **0.25** for both RI and WEAKEN warnings (from the sweep: at 0.25 the corridor model catches 92.9% of RI and 89.9% of WEAKEN — the two widened fits were swept separately and land on the same 0.25, west 85.4%/85.2% and east 90.8%/88.2%, see Step 23; lower thresholds bought almost nothing and cost hundreds more false alarms). A warning means *prepare*, not *evacuate*; the globe still shows the raw probabilities.

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

## Step 16 — Test the ERA5 environment upgrade (added Sat Nov 14)

- [x] Create a Copernicus CDS account and accept the CC BY licences for both ERA5 monthly datasets.
- [x] Download ERA5 monthly means 2000–2020 for the corridor: pressure levels (u, v, humidity at 200/700/850 hPa) and single levels (CAPE).
- [x] Build a climatology: one typical SHEAR, RH700 and CAPE value per 2.5° cell × month, training years only (no leakage into the 2021–2025 test).
- [x] Retrain with the extra features and compare at the warning threshold t = 0.25: 15 features RI 91.6% / WEAKEN 90.4%; 14 (+CAPE) RI 90.9% / 89.9%; 16 (+all three) RI 90.9% / 90.4% — none holds the shipped model's RI 92.9%.
- [x] Verdict: NOT adopted. A monthly average gives every storm in a cell and month the same values, so it cannot separate the one storm that explodes. Recorded in the notebook as Step 15.
- [x] Keep the climatology values as displayed context on the globe's conditions panel (typical shear / humidity / CAPE for the place and month).

## Step 17 — Add the 24-hour track forecast (added Sat Nov 14)

- [x] Build the track target: each storm point's actual movement over the next 24 hours, plus its motion over the last 6 hours.
- [x] Compare four methods on the unseen years 2021–2025: persistence 411 km mean error; climatology (typical move per cell × month) 371 km; linear regression 377 km; blend of persistence + climatology 358 km.
- [x] Ship the blend for live storms (uses their real motion) and climatology for clicked cells; print the honest mean error next to the forecast.
- [x] Export `track_climatology.json` (1,594 cell-months) from the notebook (notebook Step 16) and stage it for the repo.
- [x] Wire the track arrow into the globe: line from the storm's position to the predicted position 24 hours later (draws for the selected cell and for live storms; 2026-11-15 it also falls back across months and neighbouring cells when a cell-month is thin).

## Step 18 — Add live storms (added Sat Nov 14)

- [x] Verify a live source: IBTrACS ACTIVE (NOAA NCEI) — the same dataset family as the training archive, with provisional rows for storms in progress.
- [x] Build `live_storms.json`: each active West Pacific storm's latest position, its real 13 features from its own last 24 hours of reports, the model's probabilities, and its blend track. (Current entry: Cempaka/Samuel's last tracked point.)
- [x] Write the refresher: `update_live.py` plus a GitHub Action (`live.yml`, every 6 hours) that rebuilds the file and commits it — no server needed.
- [x] Upload `live_storms.json`, `track_climatology.json`, `update_live.py` and `.github/workflows/live.yml` to the repo.
- [x] Show live storms on the globe with a LIVE marker; clicking one loads its real values into the panel.

## Step 19 — Test the complete product

- [ ] Run the notebook from the beginning without errors.
- [x] Open the globe and test location selection (live site, Sat Nov 14).
- [x] Test one likely intensification example (east of Mindanao defaults → RI WATCH).
- [ ] Test one likely weakening example.
- [ ] Test one likely steady example.
- [ ] Check the layout on a desktop screen and a phone-sized screen.
- [ ] Fix errors before recording the demonstration.

## Step 20 — Finish the repository documentation

- [x] Update `README.md` with:
  - what the project does;
  - how to run the notebook;
  - how to open the globe;
  - the data source;
  - the main results;
  - the limitations.
- [x] Add the IBTrACS citation and DOI to the notebook and README.
- [ ] Add citations for every library and dataset used.
- [ ] Commit the notebook (.ipynb from Kaggle Version 6), processed-data documentation, model weights, region statistics, and web files. (Weights, statistics and web files are in; the .ipynb upload is still outstanding.)

## Step 21 — Write the project report

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

## Step 22 — Record and submit

- [ ] Record a demonstration of two minutes or less.
- [ ] Show the globe, location selection, factor inputs, and three probabilities.
- [ ] Show one notebook result table.
- [ ] Make the GitHub repository public.
- [ ] Test the public repository and globe links while signed out.
- [ ] Submit the repository link, write-up, and demonstration video.
- [ ] Save the submission confirmation.

## Step 23 — Widen the domain past 145 °E (added Sun Nov 15)

- [x] Measure what sits outside the corridor before training anything: 224 storms cross 145 °E and 88 exist only east of it. Observed 24 h RI rates: 2.65 % in the WP core (635 of 28,503 records), 3.22 % in the WP east of 145 °E (312/8,302), 2.99 % in the central Pacific (138/4,619), 6.25 % in the eastern Pacific (734/31,448). Risk is roughly twice as high on the far side of the dateline, so one threshold cannot serve both. (`tools/domain_census.py`)
- [x] Split at the dateline instead of fitting one model across 190° of longitude: west 5–40 °N, 110–180 °E; east 5–40 °N, 180 °E–70 °W, which covers the central and eastern Pacific and the Gulf/Caribbean part of the Atlantic.
- [x] Replace the linear longitude feature with a cyclic pair (`LON_SIN`, `LON_COS`) in both widened fits, so longitude is a position on a circle rather than a distance from 110 °E.
- [x] Retrain outside the notebook (`tools/ri_domain_model.py`): west 23,917 labelled points (3.31 % RI, 364 cells), east 20,234 points (5.56 %, 437 cells). At t = 0.25 the west fit catches 85.4 % of RI and 85.2 % of weakening, the east fit 90.8 % and 88.2 %.
- [x] Ship weights, cell statistics, cell × month defaults and track climatology for both as the `_wp` and `_east` twins. The page drops the original 110–145 °E file at load time because the west box contains it, so nothing is scored twice.
- [x] Draw the union of every loaded region (801 cells, 749 drawn) instead of only the region the last click landed in, so panning across the Pacific shows the history without a click first.
- [x] Hide cells whose every training report was over land (`DIST2LAND` = 0): 52 of 801 — the landfall tracks across Guangdong, Vietnam, Mexico and Central America.
- [x] Stop one outlier from setting the colour ceiling: across both basins a single 71 % cell would have pushed the whole map into the blue end, so the ramp ends at the 98th percentile of well-sampled cells (35 %+, capped at 40).
- [x] Fall back for the 24 h movement when a cell-month has fewer than five storms: this cell and month → this cell's nearest month → the ring of neighbouring cells, with the source named in the panel.
- [x] Add a custom storm: latitude/longitude boxes place one at exact coordinates, its features stay editable, and its 24 h position keeps the typical direction while using the motion that was typed.
- [x] Fill the conditions table for every drawn cell. `era5_env.json` went from 196 cells to all 801 (9,732 rows): ERA5 entries kept, the rest from NCEP/NCAR 1991–2020 on the same 2.5° grid, each row tagged with its source. CAPE is not in the NCEP monthly means and is not invented — those cells say "no ERA5 CAPE here". (`tools/env_climatology.py`)
- [x] Re-run the stress test on the original corridor model after all of this: alarm rate 42.5 % against the 4.7 % base rate, +0.0 pp from the baseline.
- [x] Record the readings in `README.md`: the two fits' metrics, the per-basin base rates, the coefficient table that explains why the same storm scores differently either side of the dateline, the environment-source note and the limitations that moved with them.

## Progress log

- **Mon Nov 16 — impossible values are refused, not scored.** The ten input boxes accepted anything a number field can parse, so a negative wind (−15 kt) or a negative distance went straight into the fit and came back as a confident probability for a storm that cannot exist. Wind speed, pressure, SST, distance to land and motion now have a floor of zero (`FLOOR` in `index.html`, checked by `rangeGate()`): typing one withdraws the result, the bars and the 24 h line, and the verdict box reads `IMPOSSIBLE VALUE — WIND KT -15 …` naming up to three offending boxes; fixing them clears the message. ΔWIND and ΔPRES are differences and stay free, since a −5 kt day is exactly the weakening case the model is meant to catch. The gate runs on input (not only on RUN) and before the fit, next to the existing missing-value and out-of-box refusals.

- **Mon Nov 16 — the per-storm numbers moved onto the globe.** The live box had grown into a block of five lines per storm, which by three storms was most of the panel. It is now a compact index — `● LAST Cempaka 64 kt`, clickable — and everything else appears when the pointer is on the cyclone: status, wind, pressure and report time, the dissipation line for a storm that is over, the replay frame's own wind and pressure, the three model probabilities with the 24 h wind and pressure change, or for a storm outside the model's basins its `not scored · south Indian` line and the motion-only 24 h point. The text is painted by the pointer readout that already follows the cursor (a `.storm` class gives it wrapping and a 270 px measure), because the spiral is a custom-layer mesh whose labels never fired a hover tooltip in this build - `.customLayerLabel` was tried first and returned nothing on hover. Hovering anywhere else leaves the old coordinate/wind/pressure line untouched, and clicking a name in the index still loads that storm.

- **Mon Nov 16 — live tracking for the whole planet.** Live mode only read WP, EP and CP storms, so a real cyclone anywhere else was invisible even when IBTrACS had it. `tools/update_live.py` now takes every storm in the ACTIVE feed (any basin, any hemisphere), keeps its name, basin, position, wind, pressure and full history, and scores only those inside the model's boxes: a storm in the North Atlantic, the Indian Ocean or the southern hemisphere is tracked and drawn — same spiral, same orange trail, same pale-gold branch — but its `features`/`prediction` are null and its 24 h position is extrapolated from its own last reported step, not from a climatology the model never saw. The 24 h branch, the head-dot tooltip and the `#track` line all say "storm motion only · no local climatology" for those storms and keep the 358 km blend error only where the blend is real. The live panel labels them "not scored · south Indian, outside the model's basins", and clicking one fills the form with the storm's own wind and pressure and prints TRACKED, NOT SCORED with RUN disabled instead of a probability; cells with no environment climatology now say so rather than crediting ERA5. Verified against today's feed: Pena (EP) and Cempaka (WP) scored as before, and the unnamed South Indian system of 7–10 Nov (19 kt at 9.2 S 47.8 E) tracked at 11.3 S 50.5 E on the 8 Nov frame with a motion-only 24 h point at 5.2 S 43.0 E.

- **Sun Nov 15, night (6) — the replay branch, and the inverted movement tables.** The pale 24 h branch reused `live_storms.json`'s single `track_24h` at every replay frame, so an early report was drawn along the storm's *final* heading: Cempaka on 7 Nov at 19.0 N 144.2 E pointed to 21.5 N 151.5 E while it actually reached 21.3 N 139.3 E, 1,300 km and one hemisphere of longitude away. It is now rebuilt per frame from that frame's own 6-hour motion (x4 to 24 h) blended half-and-half with the climatology for the cell and month the storm sits in then — the updater's method — which lands 156 km from Cempaka's real 8 Nov position (196 km at the next frame, 586 km at the 9 Nov recurvature, 213 km at 10 Nov; mean well inside the documented 358 km). Chasing that exposed the real reason the direction was backwards: every `dLat`/`dLon` in `track_climatology*.json` had the opposite sign to the archive. Checked against IBTrACS last-3-years 24 h pairs bucketed by first fix, across 1,159 cell-months: |table + real| = 2.0° vs |table − real| = 4.6°, the swap hypothesis no better than no correction (4.6°), the busiest western cell-months reading (−2.1, +1.5) where the archive moves (+1.5, −1.0), and the table's magnitudes (4.75° mean) matching the archive's (3.88°). All three files now carry the archive's sign, so a clicked November cell at 16 N draws its 24 h track west-north-west (18.3 N 123.4 E from 16.3 N 127.5 E) instead of east, and the notebook's 358 km now describes the shipped table. The live box keeps its stored final-point vector; the next scheduled updater run rebuilds it from the corrected table.

- **Sun Nov 15, night (5) — review fixes.** Four things a reviewer found in the pushed page. (1) `trackInfo()` read `e.t.src` before it knew whether the 24 h point came from a live storm — live points carry no climatology entry, so tapping Pena or Cempaka threw before `predict()` and the camera move ran; the live branch now renders and returns first, and the draw-path test is guarded. (2) `loadLive()` never wrote the latitude/longitude boxes, so after tapping a live storm PLACE still held whatever the previous cell left there; it now syncs them the way `pick()` does. (3) NOW jumped to the current frame without stopping playback, so the next tick stepped off it again; it calls `setPlaying(false)` first, like −3 h and +3 h. (4) Longitudes were printed raw, so any position computed by adding a displacement (a 24 h point, a drawn path, a storm near the dateline) could read a hemisphere it is not in — `215.7°` printed as "215.7° E" instead of 144.3° W. `fmtLng()` now normalises to −180…180 for every readout, the cursor box wraps its coordinate once and samples with the wrapped value so the text and the numbers agree, and `pick()` wraps what the globe hands it so selections keep the same convention as the cell keys. Verified in the browser: tapping both live storms renders the track line, fills the forecast and moves the camera with no exception; the boxes follow the tapped storm; NOW stops playback on the current frame; and a 168.3° E storm heading east 50 kt reports its 24 h point at 171.6° W while `fmtLng(215.7)` returns 144.3° W.

- **Sun Nov 15, late (4) — the ERA5 environment table is complete.** Roshan's `7ad02c8` landed a finished `era5_env.json`: 9,732 → 12,096 cell-months (1,008 cells × 12), the 2,364 missing entries filled and CAPE added to every entry, with the values that were already there left byte-identical (checked: 0 changed fields across the 9,732 shared entries, 2,364 keys added, none removed). The one user-visible change is the Instability row: it used to appear only for the 196-cell original ERA5 corridor and read "no ERA5 CAPE here" everywhere the widened fits had gone, and every drawn cell now has shear, humidity and CAPE. Because NCEP's monthly means carry no CAPE of their own, the CAPE on NCEP-tagged cells comes from the ERA5 export, and the panel says so per cell ("NCEP/NCAR shear & humidity, ERA5 CAPE") instead of crediting one source for both. Nothing else moved: the environment values remain context and never model inputs (Step 15 rejected them on accuracy), the model, defaults, climatology and wind frames are untouched, and the page needed only that source note. Verified in the browser on four cells — the old ERA5 corridor (15 N 125 E), an NCEP cell in the WP box (10 N 150 E) and two east-Pacific cells — all three rows populated on each, console clean.

- **Sun Nov 15, evening (3) — draw the cyclone's pathway, then predict from it.** Item 6's custom storm could only be placed at one point, so the user had to guess the storm's motion. It can now be *drawn*: DRAW turns each globe click into a 6-hour fix, UNDO drops the last one, CLEAR starts over, and the pathway is drawn in light blue with a dot per fix (the panel counts fixes) so the user's own input reads apart from the reported tracks. Two fixes are enough to predict: the last leg gives the heading, its length over 6 h gives the speed in knots, that speed is written into MOTION KT (so the model's SPEED feature is the drawn storm's, not the cell's typical one), and the next 24 h is the same leg projected 24 hours on — a pale-gold branch ending in the 24 h dot, worth 34 kt over the test path 13.5 N 129 E → 18.2 N 137.4 E. The 24 h panel line names the source and refuses to quote an error it cannot measure ("a hand-drawn path has no measured error"), and typing a different MOTION KT keeps the drawn heading but rescales the distance. The fixes carry position only: wind, pressure and D24 are still the form's, pre-filled from the first fix's cell, so a drawn storm is scored by the same model as a live one. Clearing the path returns the page to an ordinary clicked cell.

- **Sun Nov 15, evening (2) — 24 h forecast numbers.** The page could say *how likely* a rapid change was but never *what* the numbers would be. It now prints a 24 h forecast beside the probabilities: wind *now → +24 h* with the implied change, and pressure the same, for a live storm (in its box and in the 24 h marker's tooltip), a hand-placed storm, and any clicked cell. The rule is deliberately plain so it can be checked by hand: the expected wind change is ±30 kt — the smallest change each rapid class allows — weighted by the RI and weakening probabilities the bars already show, so the forecast is conservative and moves only when those bars move; pressure then follows the Atkinson–Holliday wind–pressure exponent (0.644), anchored on the storm's own current pressure so the pair begins exactly at the observation. Worked examples: Cempaka 64 kt / 983 hPa at RI 38.5 % → 72 kt / 978 hPa; Pena 40 kt / 1002 hPa at 14.3 % → 44 kt / 1001 hPa; the 16.3 N 127.5 E November cell at 48 kt / 991 hPa and RI 11 % → 45 kt / 992 hPa. No retraining and no second model: it is the same three probabilities read as a number, and it is labelled that way on the page.

- **Sat Nov 14, morning:** Model locked (logistic regression + SST + pressure trend, t = 0.25): RI 92.9%, WEAKEN 89.9% on 2021–2025. Globe live and verified. Extra validation on Typhoon Noru (never used in training): RI 9/11, WEAKEN 4/4; both misses were the post-landfall "second wind-up", the same failure pattern as Rai.
- **Sat Nov 14, midday:** ERA5 upgrade tested and honestly rejected (Step 16 above). Live replay of Typhoon Samuel (Cempaka) from IBTrACS ACTIVE: at Nov 9, 06:00 UTC the model said RI 92.9% with Samuel's real numbers, but its actual next-24-hour change was +5 kt — a confident false alarm caused by momentum overshoot right after its own intensification finished. Kept for the report's limitations.
- **Sun Nov 15:** Widened the model past 145 °E (Step 23). Two fits replace one: west 110–180 °E (3.31 % RI base, 85.4 % of RI caught at t = 0.25) and east 180 °E–70 °W (5.56 %, 90.8 %); the original 110–145 °E file is now only the stress baseline and is dropped by the page. The globe draws all 801 cells (749 once land-only cells are hidden), switches model by box at the dateline, falls back for 24 h movement where a cell-month is thin, and takes a hand-placed storm whose forecast uses its typed speed. The conditions table was filled for every cell (ERA5 where it existed, NCEP/NCAR 1991–2020 elsewhere; CAPE stays ERA5-only). Fixed on the way: the defaults export wrote `NaN`, which is not JSON, so both `region_defaults` files were rejected by the browser and the SST layer and pre-fill ran silently empty; data fetches now revalidate instead of trusting the HTTP cache.
- **Sun Nov 15, evening — live layer reviewed, simplified and fixed.** The four busy-hours fixes are on `main` as pushed earlier (data fetches revalidate instead of trusting the HTTP cache, the retraining tool writes strict JSON, the moved tools find the repo root themselves, and the page/README carry the two-basin numbers); re-reading them turned up one real break and two pieces of leftover machinery. `update_live.py` would have crashed at the end of its storm loop: a helper named `at(hours, col)` inside the loop shadowed the module-level `at(name)` path helper, so `json.dump(out, open(at('live_storms.json')))` raised `TypeError` and no storm file would have been written — renamed to `past()`. The wind-frame list was built by two loops plus a `seen` set to stop the 6 h/3 h join emitting a timestamp twice; that is now one sorted set, which made the page's duplicate-frame filter dead code, so the page lost it too. Live coverage now follows the page's own rule rather than the WP basin alone: every North Pacific storm in the IBTrACS ACTIVE feed (WP, EP, CP) is drawn and scored, each by the fit for the box it sits in (`_wp` west of the dateline, `_east` across it) with that box's defaults, track climatology and cyclic longitude pair — the corridor model is only the fallback outside both boxes. That also settles an old disagreement: the live box used to print the corridor model's numbers for Cempaka (RI 13.4 %) while clicking the same storm scored it with the widened fit (RI 38.5 %), because the panel read the file and the form used the region. The 6–17 Nov window therefore holds two storms, not one: Cempaka (6–11 Nov) and Pena (2–12 Nov, east Pacific). A storm now exists on the globe only while the frame is inside its tracked history; after its last report it is removed instead of being pinned at its final position, and the panel keeps the LAST STORM label while saying `dissipated 11 Nov 12:00 UTC · its last report`. The orange bar is gone: each storm is a hurricane spiral drawn as geometry on the globe (three tapered arms around a pale eye, coloured per vertex like the cell meshes). It is built rather than textured on purpose — a textured sprite or plane does not draw in this renderer, which cost a detour through `cyclone.svg` before the plain-colour meshes proved it; that file is now the icon beside the storm name in the panel. Each storm also carries its pathway now: the track it has travelled (first fix to its position at the frame on screen, orange) and the 24 h branch it is heading along (pale gold, ending in a 24 h position dot), rebuilt with the rest of the per-frame layer so scrubbing the time bar walks the track forward and a dateline crossing is unwrapped instead of drawn across the globe. The custom storm was brought up to the same standard: it gets the spiral marker instead of a floating selection dot, its pale-gold 24 h branch (there is no observed past to draw, only the heading), and the coordinate boxes now say what is wrong ("enter a latitude and a longitude", "latitude −90…90, longitude −180…180") and confirm the placement with its heading, instead of tinting the row and doing nothing. A point outside both training boxes is placed and labelled as unpredicted rather than silently refused. The click-forecast line that used to be drawn for a selected storm was removed with it — the storm's own pathway already shows exactly that line, and drawing both put two start points on one forecast.
- **Sat Nov 14, afternoon:** Track forecast added (Step 17) and live mode built (Step 18). Kaggle notebook updated to Version 6: Step 15 = ERA5 verdict, Step 16 = track (self-healing: downloads IBTrACS itself in a fresh session; printed 411/371/358/377 km), Step 17 = live mode. Globe redesigned (dark Earth, drawn coastlines, shape options Cells/Field/Columns/Dots); verified: model math exact, SST layer works, Dots clean. Known issue: the Cells shape renders as jagged shards (globe.gl polygon rendering) and overloads the browser — use Dots as the default until fixed.
