##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters

# Set up the file names and locations.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter3_result_outliers_goat.csv'
RESULT_FNAME = 'chapter3_result_final_goat.csv'
ORIG_DATASET_FNAME = 'chapter2_result_goat.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():

    print_flags()

    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (
        dataset.index[1] - dataset.index[0]).microseconds/1000

    MisVal = ImputationMissingValues()
    LowPass = LowPassFilter()
    PCA = PrincipalComponentAnalysis()

    if FLAGS.mode == 'imputation':
        # Let us impute the missing values and plot an example.
        for i in dataset.columns:
            print(i,dataset[i].isna().sum()/len(dataset[i]))

        # imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'pressure')
        # imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'pressure')
        imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'ax')
        # imputed_kalman_filter_dataset = MisVal.kalman_filter(copy.deepcopy(dataset), 'pressure')
        print(imputed_interpolation_dataset.shape)
        # DataViz.plot_dataset(imputed_interpolation_dataset,['pressure'],['like'],['line'])
        DataViz.plot_imputed_values(dataset, ['original', 'interpolation'], 'ax', imputed_interpolation_dataset['ax'])

    elif FLAGS.mode == 'kalman':
        # Using the result from Chapter 2, let us try the Kalman filter on the light_phone_lux attribute and study the result.
        try:
            original_dataset = pd.read_csv(
            DATA_PATH / ORIG_DATASET_FNAME, index_col=0)
            original_dataset.index = pd.to_datetime(original_dataset.index)
        except IOError as e:
            print('File not found, try to run previous crowdsignals scripts first!')
            raise e

        KalFilter = KalmanFilters()
        kalman_dataset = KalFilter.apply_kalman_filter(
            original_dataset, 'pressure')
        DataViz.plot_imputed_values(kalman_dataset, [
                                    'original', 'kalman'], 'pressure', kalman_dataset['pressure_kalman'])
        DataViz.plot_dataset(kalman_dataset, ['pressure', 'pressure_kalman'], [
                             'exact', 'exact'], ['line', 'line'])

        # We ignore the Kalman filter output for now...

    elif FLAGS.mode == 'lowpass':

        # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

        # Determine the sampling frequency.
        fs = float(1000)/milliseconds_per_instance
        cutoff = 1.5
        # Let us study acc_phone_x:
        new_dataset = LowPass.low_pass_filter(copy.deepcopy(
            dataset), 'acc_phone_x', fs, cutoff, order=10)
        DataViz.plot_dataset(new_dataset.iloc[int(0.4*len(new_dataset.index)):int(0.43*len(new_dataset.index)), :],
                             ['acc_phone_x', 'acc_phone_x_lowpass'], ['exact', 'exact'], ['line', 'line'])

    elif FLAGS.mode == 'PCA':

        #first impute again, as PCA can not deal with missing values
        for col in [c for c in dataset.columns if not 'label' in c]:
            dataset = MisVal.impute_interpolate(dataset, col)


        selected_predictor_cols = [c for c in dataset.columns if (
            not ('label' in c)) and (not (c == 'press_phone_X (hPa)'))]
        pc_values = PCA.determine_pc_explained_variance(
            dataset, selected_predictor_cols)

        # Plot the variance explained.
        DataViz.plot_xy(x=[range(1, len(selected_predictor_cols)+1)], y=[pc_values],
                        xlabel='principal component number', ylabel='explained variance',
                        ylim=[0, 1], line_styles=['b-'])

        # We select 7 as the best number of PC's as this explains most of the variance

        n_pcs = 7

        dataset = PCA.apply_pca(copy.deepcopy(
            dataset), selected_predictor_cols, n_pcs)

        # And we visualize the result of the PC's
        DataViz.plot_dataset(dataset, ['pca_', 'label'], [
                             'like', 'like'], ['line', 'points'])

    elif FLAGS.mode == 'final':
        # Now, for the final version.
        # We first start with imputation by interpolation

        for col in [c for c in dataset.columns if not 'label' in c]:
            dataset = MisVal.impute_interpolate(dataset, col)

        # And now let us include all LOWPASS measurements that have a form of periodicity (and filter them):
        periodic_measurements = ['ax', 'ay', 'az', 'gx', 'gy',
                                 'gz','pressure']


        # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

        # Determine the sampling frequency.
        fs = float(1000)/milliseconds_per_instance
        cutoff = 1.5

        for col in periodic_measurements:
            dataset = LowPass.low_pass_filter(
                dataset, col, fs, cutoff, order=10)
            dataset[col] = dataset[col + '_lowpass']
            del dataset[col + '_lowpass']

        # We used the optimal found parameter n_pcs = 7, to apply PCA to the final dataset

        selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (c == 'ax'))]
        print(dataset.isna().sum() )
        n_pcs = 6
        kopie = copy.deepcopy(dataset).fillna(0)
        dataset = PCA.apply_pca(kopie, selected_predictor_cols, n_pcs)
        print(dataset.isna().sum())
        # And the overall final dataset:
        DataViz.plot_dataset(dataset, ['a', 'g', 'temp'],
                             ['like', 'like', 'like', 'like'],
                             ['line', 'line', 'line', 'line'])

        # Store the final outcome.

        dataset.to_csv(DATA_PATH / RESULT_FNAME)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='imputation',
                        help="Select what version to run: final, imputation, lowpass or PCA \
                        'lowpass' applies the lowpass-filter to a single variable \
                        'imputation' is used for the next chapter \
                        'PCA' is to study the effect of PCA and plot the results\
                        'final' is used for the next chapter", choices=['lowpass', 'imputation', 'PCA', 'final'])


    FLAGS, unparsed = parser.parse_known_args()

    main()
