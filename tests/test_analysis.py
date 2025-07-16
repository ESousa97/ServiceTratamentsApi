import pandas as pd
from analysis.statistical_analyzer import statistical_analyzer

def test_basic_statistics():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    stats = statistical_analyzer.analyze_dataset(df)
    assert "basic_stats" in stats
