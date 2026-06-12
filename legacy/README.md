# Legacy Experiments

This directory contains historical code, model weights, and exploratory analysis from earlier stages of the student project. These files are preserved for traceability, but they are not part of the main application workflow.

## Historical Python Code

- `main.py`: an earlier Attention-LSTM training entry point. It references a missing `load_and_process_data` function and is not currently runnable.
- `LSTM.py`: the earlier unidirectional Attention-LSTM model definition used by `main.py`.
- `data.py`: an older simulation approach based on separate electric, plug-in hybrid, and fuel vehicle files. Its original relative paths no longer match the reorganized repository.

These scripts are retained as project history and should not be used as the documented entry points.

## Historical Model Weights

- `Bi-LSTM_model.pth`
- `GRU_model.pth`
- `LSTM_model.pth`

The active application uses `models/bilstm_attention.pth`. The historical weights are loaded only by `scripts/evaluate_models.py` for the development comparison figure.

## PCA Experiment

`pca_experiment/` contains an earlier exploratory analysis of fuel-consumption and vehicle emission attributes, including cleaned datasets, standardized data, PCA output, correlation plots, and a Chinese-language processing report.

This experiment does not feed the active Streamlit application or BiLSTM-Attention training pipeline. It is included to document earlier data exploration.

## Local Historical Data

`legacy/data/` may contain unused source datasets from previous experiments. The directory is excluded by `.gitignore` because these files are not required by the active pipeline and their public redistribution terms have not yet been documented.

## Supported Entry Points

Use the current files instead:

```bash
streamlit run app/streamlit_app.py
python scripts/generate_simulation_data.py
python scripts/train_model.py
python scripts/evaluate_models.py
```

