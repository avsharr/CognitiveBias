"""Lightweight checks (avoid importing TensorFlow-heavy stacks in CI)."""


def test_paths_module():
    from scripts import paths

    assert 'datasets' in paths.DATASETS_DIR
    assert 'preprocessing' in paths.PREPROCESSING_DIR
    assert 'models' in paths.MODELS_DIR
    assert 'outputs' in paths.OUTPUTS_DIR
    assert paths.RESULTS_DIR == paths.OUTPUTS_DIR


def test_preprocessing_package():
    from scripts.preprocessing import PAIR_FEATURE_COLS, preprocess_data_lab
    import pandas as pd

    assert isinstance(PAIR_FEATURE_COLS, list)
    df = pd.DataFrame({'sequenceA1': [1], 'sequenceB1': [2], 'sequenceChoiceLeft': [0],
                       'isDifferentMeanSameVariance': [0]})
    for i in range(2, 9):
        df[f'sequenceA{i}'] = 1
        df[f'sequenceB{i}'] = 1
    out, ca, cb = preprocess_data_lab(df)
    assert len(ca) == 8 and 'EV_diff' in out.columns
