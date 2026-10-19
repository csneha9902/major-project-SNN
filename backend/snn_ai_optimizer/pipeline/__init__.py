from .preprocess import preprocess_run as preprocess_run
from .baseline import baseline_run as baseline_run
from .snn_pipeline import snn_run as snn_run

def pipeline_run():
    preprocess_run()
    baseline_run()
    snn_run()

