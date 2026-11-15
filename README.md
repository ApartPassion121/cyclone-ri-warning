# Cyclone RI Warning

An AI system that predicts whether an existing tropical cyclone will rapidly intensify, rapidly weaken, or see no rapid change during the next 24 hours, presented through an interactive 3D globe covering the North Pacific from 110 °E eastward to 70 °W. It also forecasts where the cyclone will move over the same 24 hours, and it can run on live storms in progress, not only on history.

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

**Data.** NOAA IBTrACS best-track archive (full archive, records since 1842), filtered to main tracks in the West Pacific corridor (5–40°N, 110–145°E) from 2000 onward: 31,682 storm records, 654 storms. Agency wind/pressure columns are unified into single WIND/PRES series (98.3% / 99.2% coverage). Sea-surface temperature comes from NOAA OISST via the CoastWatch ERDDAP service, matched to each record's position and month (93.8% coverage; gaps are coastal cells filled with the training median, 28.6 °C). The two widened fits shipped beside it reuse the same archive over wider boxes: 36,728 records from 723 storms over 5–40 °N, 110–180 °E, and 36,046 records from 767 storms over 5–40 °N, 180 °E–70 °W (census reproduced by `tools/domain_census.py`).

**Features (13).** Current wind and pressure; wind change over the last 6, 12 and 24 hours; pressure change over the last 6 and 24 hours (found by exact time matching, never by row shifting); latitude; longitude; distance to land; month; storm motion speed; sea-surface temperature. The 24-hour pressure trend was added after a failure analysis of Typhoon Rai (see below).

**Model.** Multinomial logistic regression with balanced class weights (scikit-learn), trained on 2000–2020 and tested on 2021–2025 — years the model never saw in training (17,287 training points, 3,152 test points). A random forest is trained for comparison, and a second logistic regression written from scratch in NumPy (softmax + gradient descent, no ML library) reproduces the library model's key results. The trained weights are exported to `model_weights.json`; the globe recomputes the same probabilities in JavaScript, verified to match the Python model exactly (max difference 0.0 on the test set).

**Warnings.** A warning is issued when a dangerous outcome's probability reaches **0.25**, chosen from a threshold sweep as the balance point between catching storms and false alarms. A warning means *prepare*, not *evacuate*; the globe always shows the raw probabilities.

**The globe.** Clicking a location shows its history (storms since 2000, historical RI and weakening rates for that 2.5° cell), pre-fills the storm form with that cell's typical values for the chosen month, and displays a six-condition panel. Three of the six classic conditions (warm ocean, Coriolis/latitude, pre-existing storm) come from model inputs. The other three — instability, humidity and wind shear — are now shown with real values from an ERA5 climatology (typical for that place and month, 2000–2020; see the ERA5 experiment below). They are context for the forecaster, **not** part of the model's score: adding them to the model was tested and made it worse.

A cleared input is treated as missing rather than zero: with an empty field the page withdraws the probabilities and asks for the value instead of scoring an incomplete storm. Outside the box a weights file declares, the page refuses to predict instead of extrapolating, and shows which box that was: the fit has no training data there, and the linear latitude term would otherwise push RI probability to about 97 % at 3 °N, where Coriolis is absent and no such storm exists. Which box counts is not hardcoded any more — each weights file carries its own `domain` block, and the page gates on the boxes of whatever files it finds (see *Model files and coverage* below).

**Model files and coverage.** The page loads `model_weights.json` and, if present, `model_weights_wp.json` and `model_weights_east.json`, each with its own twin of `region_stats`, `region_defaults` and `track_climatology` (`_wp`, `_east`). Every weights file declares the area it was fitted on, in a `domain` block — `{"lat":[5,40], "lon_pm180":[110,180]}` for the western fit — and a file without one, including the shipped `model_weights.json`, falls back to the original 5–40 °N, 110–145 °E corridor. The page gates on those declarations rather than on a constant, so the frame drawn on the globe always shows the box of the model that would score the click, switching the model also switches its climatology, pre-fill, track table and base rate, and where a widened box contains the old one the redundant shipped file is dropped, so the globe never draws one box inside another. The cell layer is the union of every loaded region — 801 cells with both widened fits — so panning across the Pacific shows the history without needing a click first; the click decides which model scores the point and which climatology, pre-fill, track table and base rate the panel shows. A cell whose every training report was over land is hidden from the layer (IBTrACS `DIST2LAND` = 0: 52 of the 801, the landfall tracks across Guangdong, Vietnam, Mexico and Central America) because an inland column reads as ocean climatology and is not one; the legend counts them, and a click there still scores with the neighbouring ocean values.

That is what makes a click east of 145 °E work: the `_wp` fit was trained on 110–180 °E — 723 storms against the shipped 654 — so it has data where the shipped fit had none and the page scores the point instead of refusing it. `_east` extends the same idea across the dateline to 70 °W with its own operating point, because the central and eastern Pacific intensify about twice as often as the western corridor and a threshold tuned on one is wrong for the other. A widened model drops the linear longitude feature for a cyclic pair (`LON_SIN`, `LON_COS`, computed from the clicked longitude) — a straight-line term across 190° of longitude is not a driver of anything — and both feature sets score on the same page, since the loader reads each file's own `features` list. Both widened fits are retrained outside the notebook by `tools/ri_domain_model.py` and shipped here: the western one on 23,917 labelled points over 110–180 °E (3.3 % RI base rate; at its 0.25 operating point it caught 85.4 % of RI points and 85.2 % of weakening points) and the eastern one on 20,234 points over 180 °E–70 °W (5.6 %; 90.8 % and 88.2 %). The eastern box holds three basins, so it is fitted on the central Pacific and the Gulf/Caribbean part of the Atlantic as well as the east Pacific — leaving CP and NA out left a visible hole in the middle of the Pacific and over the Gulf, inside a box the page claims to cover. Each threshold comes from its own sweep rather than being copied across: the eastern sweep is balanced at 0.25 as well, so the two land on the same number for different reasons. With one basin the colour ceiling was the highest well-sampled rate; across both that is a 71 % cell which would push the whole map into the blue end, so the ramp now tops out at the 98th percentile of well-sampled cells (35 %, capped at 40) and the rare top cells read "35 %+".

**Custom storm.** Two boxes under the location line take a latitude and a longitude; PLACE drops a storm there, at those exact coordinates rather than at a cell centre, and fills the form from the cell it lands in so the features can be edited and scored. Its 24-hour position keeps the typical direction for that cell and uses the motion you type into MOTION KT: a 25 kt storm is projected 25 kt × 24 h along it, a 5 kt storm barely moves. Clicking the globe returns to an ordinary cell.

**24-hour movement.** The forecast is the typical displacement for the cell and month. Outside the old corridor a cell-month often has a handful of storms or none, which used to mean no forecast at all, so the page falls back to that cell's nearest month and then to the ring of neighbouring cells, and names which one it used ("this cell, Aug (nearest month), 5 past storms"). When nothing within that reach has five storms, no track is drawn rather than a guess.

**What the month selector drives.** It sets the SST layer, the form pre-fill, the ERA5 condition rows, the 24-hour movement climatology and the model's month feature. When a cell had no storms in the chosen month it borrows that cell's nearest month and says so under the location line. The RI-history layer itself is all-season: `region_stats.json` is exported per cell, not per cell × month, so a month-filtered RI map would need a re-export from the notebook — the legend states the scope rather than pretending the layer is seasonal.

**Track forecast.** The globe also predicts *where* the cyclone will be in 24 hours. Four methods were compared on the same unseen years (2021–2025, 6,391 points): persistence (the storm keeps its last-6-hour motion) — 411 km mean error; climatology (the typical 24-hour move for that 2.5° cell and month, from training years only) — 371 km; a linear regression — 377 km; and a 50/50 blend of persistence and climatology — **358 km**, the winner. The globe uses the blend for live storms (which have a real motion) and climatology for a clicked location, and prints the 358 km figure next to the forecast rather than hiding it. For scale, operational centres with full physics models average roughly 100–150 km at 24 hours; this is a statistics-only method and says so.

**ERA5 experiment (tested, not adopted).** The obvious upgrade is the air around the storm: vertical wind shear (200–850 hPa), mid-level humidity (RH700) and instability (CAPE) from ERA5 monthly means (Copernicus Climate Data Store, CC BY 4.0), built into a climatology — one typical value per cell × month from 2000–2020, training years only. Retrained and compared at the same warning threshold: +shear+humidity (15 features) caught RI 91.6% / weakening 90.4%; +CAPE (14) RI 90.9% / 89.9%; +all three (16) RI 90.9% / 90.4% — none holds the shipped model's 92.9% RI, and none of the three features ranks among the model's important inputs. The reason is structural: a monthly average gives every storm in the same cell and month the same values — it describes the typical atmosphere, not the atmosphere that storm actually faced. The values are still displayed on the conditions panel as context. The honest upgrade is daily ERA5 at each storm point (future work).

**Live mode.** NOAA publishes IBTrACS ACTIVE — the same dataset family as the training archive, with provisional rows for storms in progress. A GitHub Action in this repo (`tools/update_live.py`, every 6 hours) rebuilds each active storm's 13 features from its own last 24 hours of reports, runs the exported weights, adds the blend track, and writes `live_storms.json` for the globe. Worked example: Typhoon Cempaka (Philippine name Samuel), November 2026 — at Nov 9, 06:00 UTC, with its real numbers (84 kt, pressure down ~20 hPa in a day after intensifying 45→84 kt in ~36 hours), the model said RI 92.9%. Samuel's actual next-24-hour wind change was +5 kt; it peaked at 93 kt and decayed. A confident false alarm, kept here on purpose: the model reads fresh momentum as future momentum and cannot see the hostile environment ahead — the same limitation the ERA5 experiment measured. By Samuel's last tracked point its RI probability had fallen to 13.4%.

**Stress test.** `tools/stress_test.py` exercises the shipped weights without retraining: single-input sweeps (pressure, the wind and pressure trends, SST, distance to land and month must each move RI the way the physics says), a consistent wind-plus-pressure intensity profile, the latitude edge, eight hand-built storms including two impossible pairings, fourteen numerical extremes, and all 1,117 cell × month rows. Its headline number is the **alarm rate** — the share of default rows that cross the warning threshold — compared against `stress_baseline.json` and failed when it moves by more than 3 points, so a retrain or a threshold change cannot pass unrecorded. It currently reports 475/1117 = 42.5 % against a 4.7 % observed base rate (9×). Run `python3 tools/stress_test.py` from the repo root (or from anywhere — it finds the exported data itself); add `--update-baseline` only when a change is intended, and `--json path.json` to capture the summary for CI.

**Measurement under the pointer.** Pointing at the globe shows the wind at that spot for the frame on screen — speed and direction, from the same bilinear sampler that draws the streaks — plus sea-level pressure from that frame (MSLP on a 5° grid, written by `update_live.py`) and, when the pointer is within 250 km of a storm, that storm's own reported wind and central pressure.

**Wind flow and time replay.** The globe also animates the wind itself: current 10 m wind from Open-Meteo (free, no API key) on a global 2.5° grid, drawn as ~10,000 moving particles (4,000 on small or low-power screens) with fading trails, in the style of earth.nullschool.net. The wind is stored as time-stamped frames in `wind_frames.json` — 6-hourly frames back to the earliest live-storm history point, then 3-hourly frames for the last 24 hours plus a 48-hour forecast (57 frames, 6–17 Nov 2026, in the submitted snapshot) — and a time bar under the globe steps or plays through them, each frame labelled past, now, or forecast. Because `live_storms.json` carries each storm's full tracked history, stepping back replays the storm's real positions, interpolated between fixes, with its wind at that time: Cempaka can be watched intensifying and decaying across 6–11 Nov even though it no longer exists, labelled as a replay rather than presented as live. The wind field is visualisation and context only — it is not an input to the prediction model.

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
3. The warning threshold deliberately accepts many false alarms (roughly 6 false RI alarms per true one at the warning level). It is a *watch* trigger for preparedness, not an evacuation signal. The measured alarm rate — 42.5 % of the cell × month default rows cross the threshold against a 4.7 % observed base rate — is tracked by `stress_test.py`; the score ranks storms, it is not a calibrated probability. The panel prints that observed base rate beside the bars for the same reason.
4. The track forecast is a statistical blend (358 km mean error at 24 hours), not a physics-based forecast, and rainfall and storm surge — which cause much of the damage — are not predicted at all.
5. Results cover two basins — 5–40 °N from 110 °E eastward to 70 °W — and nothing else: south of 5 °N, the extra-tropics and the Indian Ocean are still untested, and the eastern fit was validated only on its own 2021–2025 years. Its base rate is 5.6 % against 3.3 % west, so the same operating point fires more often in absolute terms, and each basin is scored by its own logistic regression: the model knows longitude, not which basin it is looking at.
6. Known display issue: on the globe, the "Cells" tile shape can render as jagged shards in some browsers (a polygon-rendering limitation); the Dots shape is the reliable default.

## Repository contents

The page is served from the repo root by GitHub Pages, so `index.html` and every JSON it fetches stay at the top level. Everything the page does not load lives in `tools/`.

- `index.html` — the globe (GitHub Pages). Loads the JSON files below from the same directory.
- `model_weights.json` — the shipped intensity model: logistic-regression weights, intercepts, scaler, warning threshold and SST fill; 13 features on 5–40 °N, 110–145 °E, no `domain` block, so a page without the widened files falls back to that corridor. Kept as the original fit and as the subject of `tools/stress_test.py`.
- `model_weights_wp.json` / `model_weights_east.json` — the widened fits, 14 features (cyclic longitude) each carrying its own `domain`, `warning_threshold` and `ri_base_rate`, with `region_stats_wp|_east` (364 and 313 cells), `region_defaults_wp|_east` (per cell × month pre-fill) and `track_climatology_wp|_east` (24-hour movement per cell × month) travelling beside them. The page drops the 110–145 °E weights file because `_wp` contains its box, and runs on the widened files alone.
- `region_stats.json` / `region_defaults.json` — per-cell storm history and per-cell-month typical storm values that pre-fill the form.
- `track_climatology.json` — typical 24-hour movement per cell × month (training years only), exported by the notebook.
- `era5_env.json` — ERA5 monthly climatology per cell × month behind the environment rows of the conditions panel (context only, never scored).
- `live_storms.json` — active storms from IBTrACS ACTIVE with real features, predictions, tracks and each storm's full tracked history (for the time-bar replay); regenerated by the Action.
- `wind_frames.json` — global 10 m wind frames (Open-Meteo) for the wind animation, the time bar and the cursor readout, each frame carrying MSLP on a 5° grid; regenerated by the Action.
- `tools/update_live.py` + `.github/workflows/live.yml` — the 6-hourly live-data refresher: storms, tracks and wind frames (no server). It writes its output to the repo root, where the page reads it.
- `tools/stress_test.py` + `tools/stress_baseline.json` — the model stress test (sweeps, custom storms, numerical extremes, dataset alarm rate) and the tracked alarm-rate baseline it is compared against; exits non-zero when a check fails or the alarm rate drifts.
- `tools/domain_census.py` — how many storms, records, RI points and cells each candidate domain adds, and how different their RI base rates are; the measurement that sized the widened domains.
- `tools/kaggle_patch_domain.py` — applies the domain-widening edits to the model notebook (domain mask, cyclic longitude, SST subset, `_wp`/`_east` exports), refusing to write if an anchor or a code cell does not check out.
- `tools/ri_domain_model.py` — retrains one domain outside the notebook, with the same conventions as the exported weights, and writes `model_weights_<domain>.json` plus its three climatology files (`region_stats`, `region_defaults`, `track_climatology`) and a threshold sweep (needs scikit-learn, pandas, xarray and netCDF4). `--domain wp` and `--domain east` produced the shipped widened fits.
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
