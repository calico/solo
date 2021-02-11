#!/usr/bin/env python

import anndata 
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import datetime  
import pandas as pd

'''
calculate performance
'''

###############################################################################
# main
###############################################################################


def main():
        scores = np.load('testdata/results/softmax_scores.npy')
        true_labels = anndata.read('testdata/2c.h5ad').obs.doublet > .5
        
        apr = average_precision_score(true_labels, scores)
        auc = roc_auc_score(true_labels, scores)
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open('tracking_performance.csv', 'a') as file:
            file.write(f'{time},{apr},{auc}\n')
        performance_tracking = pd.read_csv('tracking_performance.csv', header=None)
        
        fig, axes = plt.subplots(2, 1, figsize=(10,20))
        ax = axes[0]
        ax.plot(performance_tracking[0], performance_tracking[1])
        ax.set_xlabel('date')
        ax.set_ylabel('average precision')
        ax = axes[1]
        ax.plot(performance_tracking[0], performance_tracking[2])
        ax.set_xlabel('date')
        ax.set_ylabel('AUROC')
        fig.savefig('performance_tracking.png')

        apr_change, auroc_change = performance_tracking.iloc[-1, 1:] - performance_tracking.iloc[-2, 1:]
        print(f'Average precision has changed by: {apr_change}')
        print(f'AUROC has changed by: {auroc_change}')
        
###############################################################################
# __main__
###############################################################################

if __name__ == '__main__':
    main()
