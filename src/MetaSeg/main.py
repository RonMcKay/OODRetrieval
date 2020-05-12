#!/usr/bin/env python3
"""
main script executing tasks defined in global settings file
"""

from configuration import CONFIG
from src.MetaSeg.functions.main_functions import ComputeMetrics, VisualizeMetaPrediction, AnalyzeMetrics
import pickle as pkl
import inspect
from os.path import join, dirname


def main():
    if CONFIG.CLASSINDEX is not None and CONFIG.DATASET.name == 'a2d2':
        with open(join(dirname(inspect.getmodule(CONFIG).__file__), 'a2d2_dataset_overview.p'), 'rb') as f:
            found_images = pkl.load(f)
        found_images = found_images[CONFIG.CLASSINDEX]
        additional_arguments = {'num_imgs': found_images}
        print(
            'CLASSINDEX {} will be processed. Number of total images: {}'.format(CONFIG.CLASSINDEX, len(found_images)))
    else:
        additional_arguments = {}

    """ COMMENT:
    From this line on, it is assumed that the INPUT_DIR defined in "global_defs.py" contains hdf5 files for each image.
    These hdf5 files should contain following data:
    
        - 3D np array with softmax output in the form (height, width, channels)
        - 2D np array with ground truth class indices 
        - full path to corresponding raw image
  
    Resulting metrics files are stored in METRICS_DIR and connected components (marked segments) files in COMPONENTS_DIR
    as pickle (*.p) files.
    """
    if CONFIG.COMPUTE_METRICS:
        run = ComputeMetrics(**additional_arguments)
        run.compute_metrics_per_image()

    """ COMMENT:
    For visualizing the rating by MetaSeg, the underlying metrics for the meta model need to be computed and saved in
    METRICS_DIR defined in "global_defs.py". In IOU_SEG_VIS_DIR the resulting visualization images (*.png) are stored.
    Refer to paper for interpretation.
    """
    if CONFIG.VISUALIZE_RATING:
        run = VisualizeMetaPrediction(**additional_arguments)
        run.visualize_regression_per_image()

    """ COMMENT:
    For analyzing MetaSeg performance based on the derived metrics, the underlying metrics for the meta model need to be
    computed and saved in METRICS_DIR defined in "global_defs.py". Results for viewing are saved in RESULTS_DIR. The
    calculation results file is saved in STATS_DIR. 
    """
    if CONFIG.ANALYZE_METRICS:
        run = AnalyzeMetrics()
        run.prepare_analysis()


if __name__ == '__main__':
    print("===== METASEG START =====")
    main()
    print("===== METASEG DONE! =====")

"""
Tips:
In order to include additional dispersion heatmaps as metrics use the function "add_heatmap_as_metric" from class
"compute_metrics". As an example on the devcube ip:192.168.1.29, add the following line

    run.add_heatmaps_as_metric( heat_dir = "/data/ai_data_and_models/inference_results/FRRNA_Softmax_Output/nparrays/
        gal/confidence/", key = "DP" )

after the metrics are computed, e.g. after run.compute_metrics_per_image() is called. Note, that the heatmap files are
assumed to be numpy arrays with a particular file name for identifying the corresponding image. This has been tested for
DS20k only.
"""
