# Limitations and Responsible Interpretation

## Intended Use

This project is a student proof of concept for demonstrating an end-to-end technical workflow. It combines simulation, time-series modeling, API integration, and GIS visualization.

It is not intended to provide operational environmental monitoring. Predictions must not be used for public health guidance, regulatory reporting, enforcement, urban policy, exposure assessment, or emergency decisions.

## 1. Simulated Target Variable

The main limitation is that `NOx_Emission` is not measured by an air-quality station or roadside sensor. It is generated using:

- sampled vehicle Smog Ratings;
- simulated traffic volume;
- an assumed electric/fuel vehicle ratio;
- a simplified congestion multiplier;
- an empirical conversion constant.

Smog Rating is an ordinal vehicle attribute and does not uniquely determine real-world NOx mass emissions. Actual emissions depend on engine technology, fuel, vehicle age, maintenance, driving cycle, road grade, temperature, after-treatment behavior, and many other factors.

The target should therefore be described as a simulated NOx proxy rather than an observed concentration or validated emission inventory.

Although the vehicle source file includes CO2 fields and the simulation code may calculate an intermediate CO2 total, CO2 is not used by the active NOx proxy label, the saved training dataset, or the model input.

## 2. No Atmospheric Dispersion Model

The system estimates a traffic-related emission proxy but does not model how pollutants move or react in the atmosphere. It does not represent:

- wind speed and direction;
- street-canyon geometry;
- atmospheric stability;
- background pollution;
- chemical conversion between NO, NO2, and total NOx;
- precipitation or boundary-layer height;
- transport from neighboring roads and districts.

For this reason, the output cannot be interpreted as ambient concentration at a person's location.

## 3. Partially Simulated Online History

Even when current traffic and weather APIs are available, the model requires 24 time steps while the application mainly has current conditions. The previous 24 hours are reconstructed using time-of-day rules and random perturbations.

This creates a mismatch between the appearance of a real-time system and the actual input provenance. The dashboard should be interpreted as a scenario-based demonstration. It is not replaying measured historical traffic at each landmark.

## 4. Estimated Traffic Volume

AMap may provide traffic descriptions and road speeds, but the application does not receive a directly measured traffic count for every landmark. Traffic volume is estimated from a manually assigned base volume and a congestion factor.

The selected base values are heuristic. They have not been calibrated against loop detectors, camera counts, or official road statistics.

The offline training-data generator also samples traffic volume and speed from simplified random distributions without strict physical bounds. As a result, some generated records may contain negative traffic flow, overly high speed, or other physically implausible values.

## 5. Spatial Limitations

The application uses 12 manually selected landmarks. A point receives a predicted value based on local assumptions, but the model does not learn spatial relationships between points.

Point radius and color are presentation choices. They do not represent a validated spatial interpolation, emission plume, or district-wide heat map.

## 6. Training and Validation Limitations

The current training script:

- fits preprocessing scalers on the complete generated dataset;
- constructs all available 24-hour windows;
- trains on all windows;
- does not create chronological training, validation, and test partitions;
- does not use early stopping or model selection on a held-out set;
- does not set complete deterministic random seeds.

As a result, training loss and in-sample behavior cannot be interpreted as out-of-sample forecasting performance.

## 7. Comparison Script Limitations

The current comparison script takes the last 200 rows from the same simulated dataset. For each row, it creates a 24-step tensor by repeating that row rather than using the actual preceding 24-hour window.

This input construction differs from the training sequence construction. The comparison figure is retained to document development history, but it is not a rigorous benchmark.

Some historical baseline models can also produce negative values because they use unconstrained linear outputs. The dashboard clips the active model output at zero, but this does not make the model physically calibrated.

## 8. Limited Evidence of Generalization

The project currently provides no evidence that the model generalizes to:

- real monitoring-station NOx measurements;
- unseen seasons or years;
- other districts or cities;
- different fleet compositions;
- unusual traffic incidents or weather events;
- policy changes affecting vehicle emissions.

Performance on generated labels mainly indicates how well the network learns patterns embedded in the simulation assumptions.

## 9. Units and Interpretation

The simulation and interface use simplified labels such as `NOx_Emission` and `mg`. The generated quantity is not fully tied to a documented physical area, road length, averaging period, or concentration unit. It should be treated as a model-output unit for the simulated proxy task, not as calibrated milligrams, an ambient concentration, or a regulatory emission estimate.

Future work must define a clear target such as `g/hour per road segment` or `micrograms/m3 at a sensor`, then ensure every transformation preserves that definition.

## 10. API and Reproducibility Constraints

Traffic API results may vary by time, account permissions, quota, coverage, and response structure. Weather services may be unavailable. Offline fallbacks include random values, so repeated runs can differ.

The included model weights depend on one generated dataset realization. Regenerating the dataset and retraining will not necessarily reproduce the same model output.

## 11. Data Provenance

The active vehicle sample includes fuel-consumption, CO2, and Smog Rating fields. CO2 is retained as source metadata but is not used by the active model pipeline. Before broader redistribution or academic use, the sample's original source and license should be verified and cited precisely.

Historical datasets stored locally under `legacy/data/` are excluded from version control because they are not part of the active pipeline and have not yet been documented for public redistribution.

## Recommended Interpretation

The strongest defensible interpretation is:

> This project demonstrates a prototype engineering pipeline for combining simulated emission labels, temporal deep learning, optional real-time contextual APIs, offline fallbacks, and GIS visualization.

It does not demonstrate validated real-world NOx monitoring accuracy.
