# Cyclone RI Warning

An AI system that predicts whether an existing tropical cyclone will rapidly intensify, rapidly weaken, or see no rapid change during the next 24 hours, presented through an interactive 3D globe for the Philippines/West Pacific corridor. It also forecasts where the cyclone will move over the same 24 hours, and it can run on live storms in progress, not only on history.

**Live demo:** https://apartpassion121.github.io/cyclone-ri-warning/
**Model notebook:** `cyclone_ri_warning.ipynb` (Kaggle, runs top to bottom)

## Desired outcomes


- A user can select a location on an interactive 3D globe.
- The system shows the selected location's historical tropical-cyclone activity.
- The user can enter the current storm values and the region's environmental factor values.
- The AI returns probabilities for all three possible outcomes.
- The system also predicts where the cyclone will be in 24 hours, with its measured average error shown.
- When a real storm is active, the globe can load that storm's live values from NOAA's provisional feed and run the same prediction on it.
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

**The globe.** Clicking a location shows its history (storms since 2000, historical RI and weakening rates for that 2.5° cell), pre-fills the storm form with that cell's typical values for the chosen month, and displays a six-condition panel. Three of the six classic conditions (warm ocean, Coriolis/latitude, pre-existing storm) come from model inputs. The other three — instability, humidity and wind shear — are now shown with real values from an ERA5 climatology (typical for that place and month, 2000–2020; see the ERA5 experiment below). They are context for the forecaster, **not** part of the model's score: adding them to the model was tested and made it worse.

**What the month selector drives.** It sets the SST layer, the form pre-fill, the ERA5 condition rows, the 24-hour movement climatology and the model's month feature. When a cell had no storms in the chosen month it borrows that cell's nearest month and says so under the location line. The RI-history layer itself is all-season: `region_stats.json` is exported per cell, not per cell × month, so a month-filtered RI map would need a re-export from the notebook — the legend states the scope rather than pretending the layer is seasonal.

**Track forecast.** The globe also predicts *where* the cyclone will be in 24 hours. Four methods were compared on the same unseen years (2021–2025, 6,391 points): persistence (the storm keeps its last-6-hour motion) — 411 km mean error; climatology (the typical 24-hour move for that 2.5° cell and month, from training years only) — 371 km; a linear regression — 377 km; and a 50/50 blend of persistence and climatology — **358 km**, the winner. The globe uses the blend for live storms (which have a real motion) and climatology for a clicked location, and prints the 358 km figure next to the forecast rather than hiding it. For scale, operational centres with full physics models average roughly 100–150 km at 24 hours; this is a statistics-only method and says so.

**ERA5 experiment (tested, not adopted).** The obvious upgrade is the air around the storm: vertical wind shear (200–850 hPa), mid-level humidity (RH700) and instability (CAPE) from ERA5 monthly means (Copernicus Climate Data Store, CC BY 4.0), built into a climatology — one typical value per cell × month from 2000–2020, training years only. Retrained and compared at the same warning threshold: +shear+humidity (15 features) caught RI 91.6% / weakening 90.4%; +CAPE (14) RI 90.9% / 89.9%; +all three (16) RI 90.9% / 90.4% — none holds the shipped model's 92.9% RI, and none of the three features ranks among the model's important inputs. The reason is structural: a monthly average gives every storm in the same cell and month the same values — it describes the typical atmosphere, not the atmosphere that storm actually faced. The values are still displayed on the conditions panel as context. The honest upgrade is daily ERA5 at each storm point (future work).

**Live mode.** NOAA publishes IBTrACS ACTIVE — the same dataset family as the training archive, with provisional rows for storms in progress. A GitHub Action in this repo (`update_live.py`, every 6 hours) rebuilds each active storm's 13 features from its own last 24 hours of reports, runs the exported weights, adds the blend track, and writes `live_storms.json` for the globe. Worked example: Typhoon Cempaka (Philippine name Samuel), November 2026 — at Nov 9, 06:00 UTC, with its real numbers (84 kt, pressure down ~20 hPa in a day after intensifying 45→84 kt in ~36 hours), the model said RI 92.9%. Samuel's actual next-24-hour wind change was +5 kt; it peaked at 93 kt and decayed. A confident false alarm, kept here on purpose: the model reads fresh momentum as future momentum and cannot see the hostile environment ahead — the same limitation the ERA5 experiment measured. By Samuel's last tracked point its RI probability had fallen to 13.4%.

**Wind flow and time replay.** The globe also animates the wind itself: current 10 m wind from Open-Meteo (free, no API key) on a global 2.5° grid, drawn as ~10,000 moving particles (4,000 on small or low-power screens) with fading trails, in the style of earth.nullschool.net. The wind is stored as time-stamped frames in `wind_frames.json` — 6-hourly frames back to the earliest live-storm history point, then 3-hourly frames for the last 24 hours plus a 48-hour forecast (58 frames, 6–17 Nov 2026, in the submitted snapshot) — and a time bar under the globe steps or plays through them, each frame labelled past, now, or forecast. Because `live_storms.json` carries each storm's full tracked history, stepping back replays the storm's real positions, interpolated between fixes, with its wind at that time: Cempaka can be watched intensifying and decaying across 6–11 Nov even though it no longer exists, labelled as a replay rather than presented as live. The wind field is visualisation and context only — it is not an input to the prediction model.

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

**Named storms (final system):** Rai (2021) — 6 of 7 RI points warned, including its pre-landfall explosion at 92–97% probability; Doksuri (2023) — 9 of 9; Yagi (2024) — 10 of 11; Mawar (2023) — no RI points inside the corridor set. Rai originally scored 4 of 7: all three misses were its second intensification over the South China Sea after crossing land, when the 24-hour wind trend still pointed down. Adding the 24-hour *pressure* trend (which had already turned) fixed two of the three without hurting the other storms. An SST-anomaly feature was also tried and rejected — it did not help. **Extra validation:** Typhoon Noru (2022), never used in training or testing, scored RI 9 of 11 warned and weakening 4 of 4; both misses were its re-intensification after crossing Luzon — the same "second wind-up" pattern as Rai. **Live replay:** Typhoon Samuel (Cempaka, 2026) is described under Live mode above — a confident false alarm, reported as one.

## Limitations

1. **The model does not see the storm's actual environment.** ERA5 shear, humidity and CAPE were tested as monthly climatology and did not help (see the ERA5 experiment); the model can therefore miss changes driven by the atmosphere around the storm — Samuel (below) is a worked example. Daily, storm-centred reanalysis is the first upgrade to make.
2. SST is a monthly satellite product matched to the nearest grid point, not a real-time feed, and ocean heat content below the surface is not used.
3. The warning threshold deliberately accepts many false alarms (roughly 6 false RI alarms per true one at the warning level). It is a *watch* trigger for preparedness, not an evacuation signal.
4. The track forecast is a statistical blend (358 km mean error at 24 hours), not a physics-based forecast, and rainfall and storm surge — which cause much of the damage — are not predicted at all.
5. Results are for one corridor and one basin; behaviour elsewhere is untested.
6. Known display issue: on the globe, the "Cells" tile shape can render as jagged shards in some browsers (a polygon-rendering limitation); the Dots shape is the reliable default.

## Repository contents

- `index.html` — the globe (GitHub Pages). Loads the JSON files below from the same directory.
- `model_weights.json` — the shipped intensity model: logistic-regression weights, intercepts and scaler.
- `region_stats.json` / `region_defaults.json` — per-cell storm history and per-cell-month typical storm values that pre-fill the form.
- `track_climatology.json` — typical 24-hour movement per cell × month (training years only), exported by the notebook.
- `live_storms.json` — active storms from IBTrACS ACTIVE with real features, predictions, tracks and each storm's full tracked history (for the time-bar replay); regenerated by the Action.
- `wind_frames.json` — global 10 m wind frames (Open-Meteo) for the wind animation and time bar; regenerated by the Action.
- `update_live.py` + `.github/workflows/live.yml` — the 6-hourly live-data refresher: storms, tracks and wind frames (no server).
- `cyclone_ri_warning.ipynb` — the model notebook (Kaggle): data, features, baselines, models, the ERA5 experiment (Step 15), the track comparison (Step 16), live mode (Step 17) and the wind frames + time replay (Step 18).

## Sources

- Knapp, K. R., et al. IBTrACS v04r01, NOAA NCEI. DOI: 10.25921/82ty-9e16. https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/
- NOAA OISST v2.1 via CoastWatch ERDDAP. https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.html
- Live storm reports: IBTrACS ACTIVE (provisional), NOAA NCEI, same DOI family as above.
- Wind data: Open-Meteo (free, no API key) — current and hourly 10 m wind, GFS/ICON blend. https://open-meteo.com
- ERA5 monthly averaged data on pressure levels and on single levels, Copernicus Climate Change Service (C3S), Climate Data Store, CC BY 4.0. Contains modified Copernicus Climate Change Service information. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
- scikit-learn, pandas, NumPy, xarray (model and data pipeline); Globe.gl / three.js (globe).
- Design inspiration for the globe's look and data layers: earth.nullschool.net and the Monsoon Jr cyclone simulator (https://monsoonjr99.github.io/cyclone-sim/) — inspiration only; all data and the model are from the NOAA sources above.

## AI assistance disclosure

Built by Roshan Sudhakar (solo) for the Veritas AI International High School Hackathon, November 2026, with Muse (Meta AI) used as a disclosed build partner for planning, code review, and verification. All modelling decisions, thresholds, and results were checked against independently reproduced runs; the numbers in this README come from the submitted notebook.

*Student research project — not an official forecast.*
