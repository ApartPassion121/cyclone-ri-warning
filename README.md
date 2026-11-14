# Cyclone RI Warning

An AI system that predicts whether an existing tropical cyclone will rapidly intensify, rapidly weaken, or see no rapid change during the next 24 hours, presented through an interactive 3D globe for the Philippines/West Pacific corridor.

**Live demo:** https://apartpassion121.github.io/cyclone-ri-warning/ (Not yet set up)
**Model notebook:** `cyclone_ri_warning.ipynb` (Kaggle, runs top to bottom)

## Desired outcomes

- A user can select a location on an interactive 3D globe.
- The system shows the selected location's historical tropical-cyclone activity.
- The user can enter the current storm values and the region's environmental factor values.
- The AI returns probabilities for all three possible outcomes.
- The system displays the six cyclone conditions as favourable or unfavourable for the selected storm and region.
- The values the user gives were chosen deliberately: they are exactly the values the model was trained on (listed below).
- Every value is pre-filled with a sensible default for the selected point — the typical value for that location and month from the dataset — and the user can override any of them.
- The model is compared against simple forecasts and tested on years it was not trained on.
- The final project provides both a reproducible model notebook and the interactive 3D globe.

## Prediction outcomes

| Outcome | Meaning |
| --- | --- |
| **Rapid intensification (RI)** | Wind speed increases by at least 30 knots in the next 24 hours. |
| **Rapid weakening** | Wind speed decreases by at least 30 knots in the next 24 hours. |
| **No rapid change (STEADY)** | Wind speed changes by less than 30 knots in either direction during the next 24 hours. |

## How it works

**Data.** NOAA IBTrACS best-track archive (full archive, records since 1842), filtered to main tracks in the West Pacific corridor (5–40°N, 110–145°E) from 2000 onward: 31,682 storm records, 654 storms. Agency wind/pressure columns are unified into single WIND/PRES series (98.3% / 99.2% coverage). Sea-surface temperature comes from NOAA OISST via the CoastWatch ERDDAP service, matched to each record's position and month (93.8% coverage; gaps are coastal cells filled with the training median, 28.6 °C).

**Features (13).** Current wind and pressure; wind change over the last 6, 12 and 24 hours; pressure change over the last 6 and 24 hours (found by exact time matching, never by row shifting); latitude; longitude; distance to land; month; storm motion speed; sea-surface temperature. The 24-hour pressure trend was added after a failure analysis of Typhoon Rai (see below).

**Model.** Multinomial logistic regression with balanced class weights (scikit-learn), trained on 2000–2020 and tested on 2021–2025 — years the model never saw in training (17,287 training points, 3,152 test points). A random forest is trained for comparison, and a second logistic regression written from scratch in NumPy (softmax + gradient descent, no ML library) reproduces the library model's key results. The trained weights are exported to `model_weights.json`; the globe recomputes the same probabilities in JavaScript, verified to match the Python model exactly (max difference 0.0 on the test set).

**Warnings.** A warning is issued when a dangerous outcome's probability reaches **0.25**, chosen from a threshold sweep as the balance point between catching storms and false alarms. A warning means *prepare*, not *evacuate*; the globe always shows the raw probabilities.

**The globe.** Clicking a location shows its history (storms since 2000, historical RI and weakening rates for that 2.5° cell), pre-fills the storm form with that cell's typical values for the chosen month, and displays a six-condition panel. Three of the six classic conditions (warm ocean, Coriolis/latitude, pre-existing storm) come from model inputs; instability, humidity and wind shear are shown as forecaster notes only, because our data does not contain them — they are **not** part of the score.

## Results (test years 2021–2025, 3,152 points)

| Forecast | Accuracy | RI caught | Weakening caught |
| --- | --- | --- | --- |
| Always STEADY (naive baseline) | 89.1% | **0%** | **0%** |
| Trend rule (past 24 h change) | 62.5% | 72.1% | 32.4% |
| Random forest | 89.6% | 14.9% | 23.9% |
| Logistic regression (argmax) | 56.9% | 83.8% | 88.3% |
| **Final system (LR + warning threshold 0.25)** | 37.2% | **92.9%** | **89.9%** |
| From-scratch NumPy logistic regression | 59.6% | 79.9% | 89.9% |

Accuracy is the wrong scoreboard here: the always-STEADY baseline is "89% accurate" and warns about nothing. The final system trades accuracy for catching the dangerous cases, at the cost of 1,334 false RI alarms across five typhoon seasons of test data.

**Named storms (final system):** Rai (2021) — 6 of 7 RI points warned, including its pre-landfall explosion at 92–97% probability; Doksuri (2023) — 9 of 9; Yagi (2024) — 10 of 11; Mawar (2023) — no RI points inside the corridor set. Rai originally scored 4 of 7: all three misses were its second intensification over the South China Sea after crossing land, when the 24-hour wind trend still pointed down. Adding the 24-hour *pressure* trend (which had already turned) fixed two of the three without hurting the other storms. An SST-anomaly feature was also tried and rejected — it did not help.

## Limitations

1. **No wind shear, humidity, or instability data.** Vertical wind shear is a known storm-killer and is the first feature to add with a reanalysis dataset (e.g. ERA5). The model can therefore miss weakening caused by shear.
2. SST is a monthly satellite product matched to the nearest grid point, not a real-time feed, and ocean heat content below the surface is not used.
3. The warning threshold deliberately accepts many false alarms (roughly 6 false RI alarms per true one at the warning level). It is a *watch* trigger for preparedness, not an evacuation signal.
4. The model predicts change in wind speed only — not storm track, rainfall, or storm surge, which cause much of the damage.
5. Results are for one corridor and one basin; behaviour elsewhere is untested.

## Sources

- Knapp, K. R., et al. IBTrACS v04r01, NOAA NCEI. DOI: 10.25921/82ty-9e16. https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/
- NOAA OISST v2.1 via CoastWatch ERDDAP. https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.html
- scikit-learn, pandas, NumPy, xarray (model and data pipeline); Globe.gl / three.js (globe).
- Design inspiration for the globe's look and data layers: earth.nullschool.net and the Monsoon Jr cyclone simulator (https://monsoonjr99.github.io/cyclone-sim/) — inspiration only; all data and the model are from the NOAA sources above.

## AI assistance disclosure

Built by Roshan Sudhakar (solo) for the Veritas AI International High School Hackathon, November 2026, with Muse (Meta AI) used as a disclosed build partner for code review, and verification. All modelling decisions, thresholds, and results were checked against independently reproduced runs; the numbers in this README come from the submitted notebook.

*Student research project — not an official forecast.*
