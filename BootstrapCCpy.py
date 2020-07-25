import numpy as np
from itertools import permutations
import bisect
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib
from scipy.cluster.hierarchy import dendrogram, linkage
from kneed import KneeLocator  # !pip install kneed
from matplotlib.ticker import MaxNLocator
import warnings

class BootstrapCCpy:
    """
      Implementation of Consensus clustering, following the paper
      https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
      Args:
        * cluster -> clustering class
        * NOTE: the class is to be instantiated with parameter `n_clusters`,
          and possess a `fit_predict` method, which is invoked on data.
        * L -> smallest number of clusters to try (fijado a 2)
        * K -> biggest number of clusters to try
        * B -> numero de muestras Bootstrap para cada número de cluster
        * Mk -> consensus matrices for each k (shape =(K,data.shape[0],data.shape[0]))
                (NOTE: every consensus matrix is retained, like specified in the paper)
        * Ak -> area under CDF for each number of clusters
                (see paper: section 3.3.1. Consensus distribution.)
        * deltaK -> changes in areas under CDF
                (see paper: section 3.3.1. Consensus distribution.)
        * self.bestK -> number of clusters that was found to be best
        * mC -> cantidad minima de atributos de los datos para muestrear columnas
      """

    def __init__(self, cluster, K, B, n_cores=-1):
        self.cluster_ = cluster
        self.L_ = 2
        self.K_ = K + 1
        self.B_ = B
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None
        self.data = None
        self._internal_resample = self._internal_resample_rows
        self.mC_ = 10
        self.n_cores = n_cores

    def _internal_resample_rows(self):
        resampled_indices = np.random.choice(range(self.data.shape[0]), size=int(self.data.shape[0]), replace=True)
        #  resampled_data_unique, resampled_indices_unique= np.unique(self.data[resampled_indices, :], return_index=True, axis=0)

        # para sin eliminacion de repetidos
        resampled_data_unique, resampled_indices_unique = self.data[resampled_indices, :], resampled_indices

        return resampled_indices_unique, resampled_data_unique

    def _internal_resample_cols(self):
        resampled_indices = np.random.choice(range(self.data.shape[0]), size=int(self.data.shape[0]), replace=True)
        # resampled_indices_cols = np.unique(np.random.choice(range(self.data.shape[1]), size=int(self.data.shape[1]), replace=True))

        # para sin eliminacion de repetidos
        resampled_indices_cols = np.random.choice(range(self.data.shape[1]), size=int(self.data.shape[1]), replace=True)

        data_sampled_cols = self.data[:, resampled_indices_cols]
        #       resampled_data_unique, resampled_indices_unique= np.unique(data_sampled_cols[resampled_indices, :], return_index=True, axis=0)

        # para sin eliminacion de repetidos
        resampled_data_unique, resampled_indices_unique = data_sampled_cols[resampled_indices, :], resampled_indices

        return resampled_indices_unique, resampled_data_unique

    '''
    Next method is used to find a better n given an array of areas corresponding to CDF
    The problem it comes to address is the changing optimal k found depending on the 
        length of the given areas corresponding to the range L-K submitted by de user

    The procedure is the following
        - The amount of considered areas is increased from 2 to maximum
        - A knee point is calculated and stored for each areas array just created
        - The occurrences of each knee are counted
        - The most frequent knee is chosen as the most likely one
        - If there is another knee point that could have been picked, one more check is done
        - Some of the last areas in the array that support the most frequent knee point are 
            deleted in order to check if the most likely point remains the same.   
    '''
    @staticmethod
    def _determineBestKnee(areas, verbose):
        noKneeFoundCount = 0

        ## dealing with Warnings as Errors
        warnings.filterwarnings('error')

        knees = []
        areasInc = [areas[0]]
        for a in areas[1:]:
            areasInc.append(a)
            try:
                k = KneeLocator(range(len(areasInc)), areasInc, S=1.0, curve='concave', direction='increasing')
                knees.append(k.knee)
            except UserWarning:
                knees.append(None)
                noKneeFoundCount += 1

        kneesArray = np.array(knees)

        if isFullOfNone(kneesArray):
            knees = []

        ## True if the list is NOT empty
        if knees:
            knees = kneesArray
            unique, counts = np.unique(knees[knees != np.array(None)], return_counts=True)
            incResults = dict(zip(unique, counts))

            mostLikelyPoint = max(incResults, key=incResults.get)
            mostLikelyPointPosition = np.where(knees == mostLikelyPoint)[0][0]

            ## TODO: return a second likely point would be considered as an option

            if verbose:
                print("Count of increasing cluster number", incResults)
                print("First most likely point found:", mostLikelyPoint)
                print("First most likely point found at:", mostLikelyPointPosition)

            ## If there is at least 2 likely points
            ## If the chosen point is the greatest option, one more check is done
            # if len(incResults.keys()) > 1 and max(incResults.keys()) == mostLikelyPoint:
            if len(incResults.keys()) > 1:
                ## It includes the most frequent point and the next one

                ## the optimal points found should be sorted.
                ## let's supose [1,1,2,2,3,2] is not allowed to happen
                assert np.array_equal(knees[knees != np.array(None)], np.sort(knees[knees != np.array(None)]))

                if (max(incResults.keys()) == mostLikelyPoint):
                    mostLikelyPointNeig = mostLikelyPoint - 1
                    d = -1
                    stop = 0
                else:
                    mostLikelyPointNeig = mostLikelyPoint + 1
                    d = +1
                    stop = max(incResults.keys()) + 1

                if verbose:
                    print("full knees array", knees)

                ## Compare the number of times the greatest cluster number againts the second one

                while mostLikelyPointNeig not in incResults.keys() and stop != mostLikelyPointNeig:
                    mostLikelyPointNeig += d

                if stop != mostLikelyPointNeig:
                    if (max(incResults.keys()) == mostLikelyPoint):
                        valueDiff = incResults[mostLikelyPoint] - incResults[mostLikelyPointNeig]
                        if valueDiff > 0:
                            for i in range(valueDiff):
                                knees = knees[:-1]

                        ## Up to this moment, the higher point is less or equal than the second one
                        if (incResults[mostLikelyPoint] - valueDiff) > 1:
                            knees = knees[:-1]

                    if verbose:
                        print("knees after deleting some entries of most frequent", knees)

                    areasReduced = areas[:len(knees)]

                    if verbose:
                        print("most frequent point is the greatest NO")
                        print("previous most likely point:", mostLikelyPoint)
                        print("reevaluating over the following areas:", areasReduced)
                    try:
                        k = KneeLocator(range(len(areasReduced)), areasReduced, S=1.0, curve='concave',
                                        direction='increasing')
                        if verbose: print("new knee", k.knee)
                        mostLikelyPoint = k.knee
                    except:
                        if verbose: print("new knee not found")

                elif verbose:
                    print("mostLikelyPointNeig not found, stop:", stop)

        else:
            print("No knee found")
            mostLikelyPoint = 0

        ## back to just showing the warnings
        warnings.filterwarnings('default')

        return mostLikelyPoint

    def _forEachSample(self, k, h, verbose):  ## siendo h el numero de muestra
        if verbose:
            print("\tAt resampling h = %d, (k = %d)" % (h, k))
        resampled_indices, resample_data = self._internal_resample()
        Mh = self.cluster_(n_clusters=k).fit_predict(
            resample_data)  ## Index of the cluster each sample belongs to Mh.shape = n_samples,  (id del cluster, )  Es propia de este sample
        # find indexes of elements from same clusters with bisection
        # on sorted array => this is more efficient than brute force search
        id_clusts = np.argsort(
            Mh)  ## It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
        sorted_ = Mh[id_clusts]

        mapp = np.zeros((2, Mh.shape[0]), dtype=int)
        mapp[0] = np.arange(Mh.shape[0])
        mapp[1] = resampled_indices

        Mkh = np.zeros((self.data.shape[0],) * 2)  ## Matriz de coincidencia en k-clusters para el sample h
        for i in range(k):  # for each cluster  ## Si se buscaron 3 clusters, hay un i por cada uno de ellos
            ia = bisect.bisect_left(sorted_, i)
            ib = bisect.bisect_right(sorted_, i)
            is_ = id_clusts[ia:ib]

            is_Mk = mapp[:, is_][1]
            ids_Mk = np.array(list(permutations(is_Mk, 2))).T

            # sometimes only one element is in a cluster (no combinations)
            if ids_Mk.size != 0:
                Mkh[ids_Mk[0], ids_Mk[1]] += 1

                # increment counts
        ids_2 = np.array(list(permutations(resampled_indices, 2))).T
        Ish = np.zeros((self.data.shape[0],) * 2)
        Ish[ids_2[0], ids_2[
            1]] = 1  ## Is changes and must be returned  ## Suma 1 por cada vez que un dato original se usó dentro de una muestra ## (1 si para esta muestra se uso el dato i,j y cayeron en el mismo cluster)

        return (Mkh, Ish)

    def _forEachCluster(self, k, verbose=False):
        ## Mk -> corresponde a la fila de este cluster
        if verbose:
            print("At k = %d, aka. iteration = %d" % (k, 7))
        Mk = np.zeros((self.data.shape[0],) * 2)
        Is = np.zeros((self.data.shape[0],) * 2)

        with Parallel(n_jobs=self.n_cores, prefer="processes") as parallel:
            sout = parallel(delayed(self._forEachSample)(k, h, verbose) for h in range(self.B_))
            for so in sout:
                Mk += so[0]
                Is += so[1]

        Mk /= Is + 1e-8  # consensus matrix
        # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
        Mk[range(self.data.shape[0]), range(self.data.shape[0])] = 1  # always with self

        return (k, Mk)

    def fit(self, data, verbose=False):
        """
        Fits a consensus matrix for each number of clusters

        Args:
          * data -> (examples,attributes) format
          * verbose -> should print or not
        """
        assert self.Mk is None, "Already fit"

        self.data = data

        ## Si se supera la cantidad mínima de atributos por datos, se muestrean las columnas
        if (self.data.shape[1] >= self.mC_):
            self._internal_resample = self._internal_resample_cols

        Mk = np.zeros((self.K_ - self.L_, self.data.shape[0], self.data.shape[0]))

        with Parallel(n_jobs=self.n_cores, prefer="processes") as fparallel:
            cout = fparallel(delayed(self._forEachCluster)(k, verbose) for k in range(self.L_, self.K_))
            for co in cout:
                Mk[co[0] - self.L_] = co[1]

        self.Mk = Mk  ## Matriz de consenso

        areas = []
        for i in range(self.Mk.shape[0]):
            values, bins = np.histogram(np.ravel(self.Mk[i]), bins=10, range=(0, 1))
            cumV2 = np.cumsum(values, axis=0)
            area = sum(np.diff(bins) * cumV2)
            areas.append(area)

        self.Ak = areas

        kneePoint = self._determineBestKnee(areas, verbose=verbose)
        kneePoint += self.L_

        self.bestK = kneePoint

        self.data = None

    def predict(self):
        """
        Predicts on the consensus matrix, for best found cluster number
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(1 - self.Mk[self.bestK - self.L_])

    def predict_data(self, data):
        """
        Predicts on the data, for best found cluster number
        Args:
          * data -> (examples,attributes) format
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(data)

    def get_consensus_matrix(self):
        assert self.Mk is not None, "First run fit"
        return self.Mk

    def get_best_k(self):
        assert self.Mk is not None, "First run fit"
        return self.bestK

    def plot_consensus_distribution(self):
        assert self.Mk is not None, "First run fit"
        matplotlib.rcParams['figure.figsize'] = 12, 6

        areas = []

        for i in range(self.Mk.shape[0]):
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

            values, bins = np.histogram(np.ravel(self.Mk[i]), bins=10, range=(0, 1))
            cumValues = np.cumsum(values, axis=0)
            area = sum(np.diff(bins) * cumValues)
            areas.append(area)

            ax1.hist(bins[:-1], bins, weights=values)
            ax1.set_title(str(i + 2) + " clusters: hist")
            ax1.set_xlabel("Consensus index")
            ax1.set_yticklabels([])

            ax2.hist(np.ravel(self.Mk[i]), range=(0, 1), bins=10, align='right', cumulative=True, histtype='step')
            ax2.set_title(str(i + 2) + " clusters: CDF")
            ax2.set_xlabel("Consensus index")
            ax2.set_yticklabels([])

            plt.plot()

        f, (ax1, ax2) = plt.subplots(1, 2)

        xs = range(2, len(areas) + 2)
        ax1.set_title("Comparativa k: Areas")
        ax1.set_xlabel("k clusters")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_yticklabels([])
        ax1.plot(xs, areas)

        dAreas = [None, 0, 1]
        for i in range(1, len(areas)):
            nDArea = (areas[i] - areas[i - 1]) / areas[i - 1]
            dAreas.append(nDArea)
        ax2.vlines(self.bestK, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
        ax2.set_title("Comparativa k: Delta Areas")
        ax2.set_xlabel("k clusters")
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_yticklabels([])
        ax2.plot(dAreas)

        plt.show()

    def plot_consensus_heatmap(self):
        matplotlib.rcParams['figure.figsize'] = 12, 6

        for i in range(self.Mk.shape[0]):
            plt.subplot(1, 2, 1)
            linked = linkage(self.Mk[i], 'single')
            R = dendrogram(linked,
                           orientation='top',
                           labels=None,
                           distance_sort='descending',
                           show_leaf_counts=True)

            plt.title(str(i + 2) + " clusters: dendogram")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(1, 2, 2)
            MSorted = self.Mk[i]
            MSorted = MSorted[:, R['leaves']]
            MSorted = MSorted[R['leaves'], :]

            plt.imshow(MSorted, cmap='Blues', interpolation='nearest')

            plt.title(str(i + 2) + " clusters: heatmap")
            plt.xticks([])
            plt.yticks([])
            plt.show()

    def get_areas(self):
        assert self.Mk is not None, "First run fit"
        return self.Ak

## auxiliar function
def isFullOfNone(arr):
    for a in arr:
        if a is not None:
            return False
    return True
