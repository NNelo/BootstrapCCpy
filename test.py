from BootstrapCCpy import BootstrapCCpy as bcc
from sklearn.cluster import KMeans

if __name__ == '__main__':
    CC = bcc(cluster=KMeans().__class__, K=10, B=250, n_cores=4)
    print("main")