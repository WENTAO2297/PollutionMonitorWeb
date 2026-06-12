# Methodology

## Scope

This document describes the current proof-of-concept pipeline. It does not describe an operational emission-monitoring system. The target variable is simulated, and part of the online temporal context is reconstructed rather than measured.

## 1. Vehicle Attribute Sample

The simulation starts from `data/sample/vehicle_emissions_2026.csv`. For the active NOx proxy calculation, the relevant source field is:

- `Smog rating`

Missing or invalid Smog Ratings are replaced with 5. The source file also contains CO2 metadata, and the script may calculate an intermediate CO2 total, but that value is not written to the active training dataset and is not used to calculate the NOx proxy or as a model input. The current simulation represents the fleet as 35% electric vehicles and 65% fuel vehicles. These are modeling assumptions, not measured fleet statistics for each Xuhui road.

## 2. Hourly Traffic Simulation

`scripts/generate_simulation_data.py` generates 90 days of hourly records, producing 2,160 rows. Traffic volume and average speed are sampled from different normal distributions according to four time periods:

- morning and evening peaks;
- daytime off-peak hours;
- late-night hours;
- transition hours.

Higher congestion multipliers are assigned to peak and daytime periods. Temperature and humidity follow simplified daily curves with random noise.

The simulator can produce physically implausible samples because it is an exploratory generator rather than a calibrated traffic model. Traffic volume and speed are sampled from simplified random distributions without strict physical upper and lower bounds. This can produce negative traffic volume, overly high speed, or other unrealistic values in some generated rows.

## 3. Simulated NOx Label

For each hour, the script estimates fuel-vehicle count from simulated traffic volume. It samples vehicle records with replacement and calculates an approximate NOx quantity from Smog Rating:

```text
vehicle NOx proxy = (11 - Smog Rating) * 8.0
```

The sum is multiplied by the congestion factor and divided by 1,000 before storage. This formula is an educational proxy. It is not a regulatory emission factor, a chemical dispersion model, or a conversion to a calibrated roadside concentration.

The resulting `NOx_Emission` column is therefore a generated learning target. CO2 values do not enter this target calculation.

## 4. Preprocessing and Sequence Construction

The model uses five input features:

```text
Traffic_Volume
Average_Speed
Temperature
Humidity
Hour
```

`StandardScaler` is fitted to these features. `MinMaxScaler` is fitted to the simulated NOx target.

The sequence builder uses a sliding window of 24 consecutive hourly rows:

```text
X[t] = rows t ... t+23
y[t] = NOx value at row t+24
```

Thus, each training tensor has shape `(24, 5)` before batching.

## 5. BiLSTM-Attention Model

The active network contains:

1. A three-layer bidirectional LSTM with hidden size 128 per direction.
2. Dropout of 0.3 between recurrent layers.
3. A temporal attention layer applied to all 24 recurrent outputs.
4. A weighted sum that produces a 256-dimensional context vector.
5. A fully connected layer from 256 to 64 dimensions.
6. ReLU activation.
7. A linear scalar output.

The temporal attention implementation calculates one score per time step, applies softmax over the sequence dimension, and forms a weighted sum of the BiLSTM outputs.

The training script uses mean squared error and Adam with a learning rate of 0.0005 for 100 epochs. The current implementation trains on all generated windows without a separate validation set.

## 6. Online Traffic Input

The Streamlit application defines 12 landmarks in Xuhui District. If `AMAP_API_KEY` or a compatible Streamlit secret is configured, `src/traffic_service.py` requests traffic information within a 1,000-meter radius of each landmark.

The returned road speeds are averaged, and the textual congestion evaluation is mapped to a factor. If the API fails, returns an error, or has no configured key, the service creates landmark-specific fallback speeds and congestion factors based on time of day and random variation.

These current API or fallback values are used only by the dashboard inference pipeline. They are not added to the stored dataset and are not used to train the included model weights.

## 7. Online Weather Input

`src/weather_service.py` requests current temperature and relative humidity using this order:

1. Open-Meteo;
2. wttr.in;
3. local time-based simulation.

The weather source is displayed in the interface.

## 8. Online 24-Hour Sequence Reconstruction

The online application does not retrieve 24 measured historical observations for every landmark. Instead, it uses the current traffic speed, a landmark-specific base traffic volume, the current weather, hour-of-day factors, and random perturbations to construct 24 prior time steps.

For each reconstructed hour:

- traffic volume is adjusted by peak/off-peak factors and random noise;
- speed is changed inversely with the time factor;
- temperature is slightly reduced by a random amount;
- current humidity is reused;
- the reconstructed hour is included as a feature.

The sequence is standardized using scalers fitted from the complete processed simulation dataset, then passed to the saved model. The output is inverse-transformed and clipped at zero.

This online sequence should be understood as a scenario generator, not measured traffic history.

## 9. GIS Visualization

The predicted scalar is converted into a point color and radius. Streamlit metrics summarize the mean, highest, and lowest predicted points. PyDeck displays the 12 locations on an interactive map.

These visual encodings show relative model output within the demonstration. They are not calibrated pollution contours or an atmospheric dispersion map.

## 10. Development Evaluation

`scripts/evaluate_models.py` compares the active BiLSTM-Attention weights with historical BiLSTM, GRU, and LSTM weights. It uses the last 200 rows of the generated dataset and repeats each row 24 times to form an input tensor.

This is useful for visual inspection of earlier experiments, but it is not a rigorous time-series evaluation. A future implementation should use chronological held-out windows, fixed seeds, saved preprocessing objects, quantitative metrics, and repeated experiments.
