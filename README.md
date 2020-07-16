# BootstrapCCpy

Consensus Clustering comes to address the issue of determining the optimal amount of clusters conformed by the dataset. 

The benefit of this approach is the provision of several measures in order to recognize the quality of each one of the conformed clusters.

It also provides visualization facilities out of the box, such as heatmaps. 

In addition, this python implementation:
* makes use of Bootstrap as the perturbation technique
* allows the usage of multi-core CPU to perform faster


## Getting started

Download this repository
```bash
git clone https://github.com/NNelo/BootstrapCCpy.git
```
_Please check out dependencies section in case you are having trouble._


Import the library
```python
from BootstrapCCpy import BootstrapCCpy as bcc
```

Instance Consensus Clustering
```python
CC = bcc.BootstrapCCpy(cluster=clusteringAlgoritm, K=number, B=number, n_cores=number)
```
_Please refer to method section for further explanation of the parameters._


## Methods

### constructor
```python
BootstrapCCpy(cluster, K, B, n_cores)
```
Parameters
- cluster

    The class of a clustering algorithm implementation (Mandatory)

    For example, you could head to [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster) to pick the one of your preference. Let's use KMeans and do it properly

     	cluster=KMeans().__class__

- K

	Positive Integer (Mandatory)
	
	Refers to the maximum number of clusters to try

	For example, if it's set to 4, the algorithm will process the data in 2, 3, and 4 clusters. 

- B 

	Positive Integer (Mandatory)

	Amount of bootstrap samples to be performed by the algorithm for each cluster number.

- n_cores

	Integer (Optional, default: -1)

	The number of CPU cores to be used by the algorithm to fit the data. If it's set to -1, all available cores will be used.



### fit
```python
fit(data, verbose)
```

Trains the algorithm with the provided data to discover the optimal number of clusters. This function can be called just once per object instance.


:warning: Take into account that this method is CPU and memory intensive, it may take a long time to be completed. :warning:

Parameters
- data
	
	ndarray (Mandatory)

- :construction: verbose

	boolean (Optional, default: False)

	Determines if it should print messages when fitting

	This method is not completely developed, please refer to [this issue](../../issues/1)



### get_best_k
```python
get_best_k()
```

This returns the optimal number of clusters discovered by analytical methods

Returns

- k
	
	Positive Integer


### plot_consensus_distribution
```python
plot_consensus_distribution()
```

### plot_consensus_heatmap
```python
plot_consensus_heatmap()
```

### predict
```python
predict()
```

### predict_data
```python
predict_data(data)
```


## Tips

Dependencies: kneed


## Next steps

- CPU and memory intensive [this issue](../../issues/2)

## Authors

* [Franco Bobadilla](https://github.com/FrancoBobadilla) - Faculty of Engineering, Catholic University of Córdoba (UCC) *
* [Nelo Nanfara](https://github.com/NNelo/) - Faculty of Engineering, Catholic University of Córdoba (UCC) *
* Ing. Pablo Pastore - DeepVisionAi, inc.
* Bioing. PhD Elmer Fernández - CIDIE-CONICET-UCC

*both authors must be considered as the first author
