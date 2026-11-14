# Cyclone RI Warning

An AI system that predicts whether an existing tropical cyclone will rapidly intensify, rapidly weaken, or see no rapid change during the next 24 hours, presented through an interactive 3D globe for the Philippines/West Pacific corridor.

## Desired outcomes

- A user can select a location on an interactive 3D globe.
- The system shows the selected location's historical tropical-cyclone activity.
- The user can enter the current storm values and the region's environmental factor values.
- The AI returns probabilities for all three possible outcomes.
- The system displays the six cyclone conditions as favourable or unfavourable for the selected storm and region.
- Need to choose on the values which the user will give
- Need to add extra values and set values for that point of default if the user does not enter their value.
- The model is compared against simple forecasts and tested on years it was not trained on.
- The final project provides both a reproducible model notebook and the interactive 3D globe.

## Prediction outcomes

| Outcome | Meaning |
| --- | --- |
| **Rapid intensification** | Wind speed increases by at least 30 knots in the next 24 hours. |
| **Rapid weakening** | Wind speed decreases by at least 30 knots in the next 24 hours. |
| **No rapid change** | Wind speed changes by less than 30 knots in either direction during the next 24 hours. |
