from BootstrapCCpy import BootstrapCCpy as bcc
from sklearn.cluster import KMeans

from scipy.sparse import dok_matrix, coo_matrix
import numpy as np

if __name__ == '__main__':
    CC = bcc(cluster=KMeans().__class__, K=10, B=250, n_cores=4)

    data_shape = 5

    Mkh = dok_matrix((data_shape, data_shape))
    print(Mkh.toarray())

    ids_Mk = np.array([[1, 2, 3], [3, 2, 2]])
    print(ids_Mk)

    Mkh1 = coo_matrix((np.ones(ids_Mk[0].shape[0]), (ids_Mk[0], ids_Mk[1])), shape=(data_shape, data_shape))

    print(Mkh1)
    print(Mkh1.toarray())

    r = Mkh + Mkh1
    print(r)
    print(Mkh1.toarray())

    ids_Mk = np.array([[3, 3, 1], [3, 2, 2]])
    print(ids_Mk)

    Mkh1 = coo_matrix((np.ones(ids_Mk[0].shape[0]), (ids_Mk[0], ids_Mk[1])), shape=(data_shape, data_shape))

    print(Mkh1)
    print(Mkh1.toarray())

    r = r + Mkh1
    print(r.toarray())

    print("empty")
    L = 2
    K = 5 + 1

    Mk = np.empty(K - L, dtype=dok_matrix)

    print(Mk)

    for idx in range(K-L):
        Mk[idx] = dok_matrix((data_shape,) * 2)

    for i in [0,1,2,3]:
        print(Mk[i].toarray())

    try:
        print(Mk[4].toarray())
    except:
        print("error")


    print(Mk.shape)

    print("main")
