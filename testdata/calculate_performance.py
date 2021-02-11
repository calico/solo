#!/usr/bin/env python

import anndata 
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import datetime  
import pandas as pd
from glob import glob
'''
calculate performance
'''

###############################################################################
# main
###############################################################################

experiment_name_to_dataset = {'pbmc': '2c.h5ad',
                              'kidney': 'gene_ad_filtered_PoolB4FACs_L4_Rep1.h5ad'}

def main():
    for result in glob('results_*/softmax_scores.npy'):
        experiment_name = result.split("/")[0].split("_")[1]
        experiment_number = result.split("/")[0].split("_")[2]
        scores = np.load(result)
        adata = anndata.read(experiment_name_to_dataset[experiment_name])
        true_labels = adata.obs.doublet_bool
        apr = average_precision_score(true_labels, scores)
        auc = roc_auc_score(true_labels, scores)
        time = datetime.datetime.now().strftime("%Y-%m-%d %H")
        with open('tracking_performance.csv', 'a') as file:
            file.write(f'{time},{experiment_name},{experiment_number},{apr},{auc}\n')

    performance_tracking = pd.read_csv('tracking_performance.csv')
    performance_tracking['date (dt)'] = pd.to_datetime(performance_tracking['date'], format="%Y-%m-%d %H")
    for experiment_name, group in performance_tracking.groupby('experiment_name'):
        fig, axes = plt.subplots(2, 1, figsize=(10,20))
        ax = axes[0]
        ax.plot(group['date'], group['average_precision'], '.')
        ax.set_xlabel('date')
        ax.set_ylabel('average precision')
        ax = axes[1]
        ax.plot(group['date'], group['AUROC'], '.')
        ax.set_xlabel('date')
        ax.set_ylabel('AUROC')
        fig.savefig(f'{experiment_name}_performance_tracking.png')
        second_to_last, most_recent = group['date (dt)'].drop_duplicates().sort_values()[-2:]
        second_to_last_df = group[group['date (dt)'] == second_to_last]
        most_recent_df = group[group['date (dt)'] == most_recent]
        for metric in ['AUROC', 'average_precision']:
            mean_change = most_recent_df[metric].mean() - second_to_last_df[metric].mean()
            pvalue = mannwhitneyu(most_recent_df[metric], second_to_last_df[metric]).pvalue
            print(f'Mean {metric} has changed by for {experiment_name}: {mean_change}')
            print(f'P value for metric change {metric} in experiment {experiment_name}: {pvalue}')
            if mean_change < 0 and pvalue < .05:
                for x in range(0,5):
                    print('WARNING!')
                print(f'WARNING {metric} HAS GOTTEN SIGNIFICANTLY WORSE for {experiment_name}!')
            if mean_change > 0 and pvalue < .05:
                for x in range(0,5):
                    print('NICE JOB!')
                print(f'NICE JOB {metric} HAS GOTTEN SIGNIFICANTLY BETTER for {experiment_name}!')

        
###############################################################################
# __main__
###############################################################################

if __name__ == '__main__':
    main()
