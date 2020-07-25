from kneed import KneeLocator
import numpy as np
from BootstrapCCpy import BootstrapCCpy as bcc

if __name__ == '__main__':
    # from iris dataset with large K
    areas4 = [5.0328, 6.9083555555555565, 7.562222222222222, 8.041955555555557, 8.338044444444442, 8.576977777777776,
              8.737955555555555, 8.87591111111111, 8.967111111111112]  # 7.562222222222222

    # from iris dataset with small K
    areas3 = [5.018488888888889, 6.904177777777779, 7.595822222222223, 8.027377777777778, 8.35511111111111,
              8.576888888888888]  # 6.904177777777779

    # from blob dataset (5 clusters, 4 dims)
    areas5 = [3.88, 6.039999999999999, 7.4799999999999995, 8.2, 8.358121999999998, 8.540752000000001, 8.731407999999998,
              8.915552000000002, 9.068554000000002, 9.140656, 9.205768, 9.265298, 9.321243999999998, 9.369857999999999]

    # from penguins dataset
    areasp = [5.145429362880886, 6.931072808727473, 7.524417769570125, 8.123029308163195, 8.32745118156014,
              8.569850552306692, 8.791029718545877, 8.938015115762115, 9.041722239321501, 9.100270168598884,
              9.172925002564893, 9.247529154269689, 9.295201942478029, 9.338394719742828]

    # result of: CC = bcc(cluster=KMeans().__class__, K=10, B=60, n_cores=4)
    areas60 = [388000.0, 604000.0, 748000.0, 820000.0, 835915.8, 854524.8, 874012.0, 891740.6, 907195.8]

    # result of: CC = bcc(cluster=KMeans().__class__, K=4, B=20, n_cores=4)
    # max() arg is an empty sequence on --> mostLikelyPoint = max(incResults, key=incResults.get)
    areas7 = [388037.79999999993, 604005.3999999999, 748009.0]

    # CC = bcc(cluster=KMeans().__class__, K=7, B=25, n_cores=2)
    # blob_data_4dims_5clusters.csv
    areas8 = [388003.6, 604000.0, 748000.0, 820000.0, 837534.8, 854814.8]

    areas = areas5


    ## gave: by the training. Should: based on dataset knowledge

    print("*****************************************************")
    print("Gave 3, should 3:", bcc._determineBestKnee(areas3, verbose=True) + 2)
    print("*****************************************************")
    print("Gave 5, should 5:", bcc._determineBestKnee(areas5, verbose=True) + 2)
    print("*****************************************************")
    print("Gave 4, should 3:", bcc._determineBestKnee(areas4, verbose=True) + 2)
    print("*****************************************************")
    print("Gave 5, should 3/5:", bcc._determineBestKnee(areasp, verbose=True) + 2)
    print("*****************************************************")
    print("Gave 5, should 5:", bcc._determineBestKnee(areas60, verbose=True) + 2)
    print("*****************************************************")
    print("Should not throw:", bcc._determineBestKnee(areas7, verbose=True) + 2)
    print("*****************************************************")
    print("Gave 4, should 5:", bcc._determineBestKnee(areas8, verbose=True) + 2)


    ## The following comment section is a non-successful try to "ponderate" the last areas
    ##      in order to make them not change the optimal poing when K is large.

    '''
    
    # print(areas)
    # areas = np.divide(1, areas)

    # normalizando areas
    scaler = MinMaxScaler()

    nareas = scaler.fit_transform(np.array(areas).reshape(-1,1))

    nareas = nareas.ravel()



    # f = lambda x: (2. / (1+exp((x)/1.))) *x

    # f = lambda x: (2. / (1+exp((x)/10.))) * x

    # f = lambda x: (2. / (1.+ (np.exp(x/100.))))*x
    # f = lambda x: x * (1.0/exp(x))
    # f = lambda x: math.sqrt(x)
    # f = lambda x: math.sqrt(x/2.)
    # f = lambda x: x
    f = lambda x: (1-(x/exp(5))) * x

    fareas = []
    for a in nareas:
        fareas.append(f(a))



    plt.plot()
    # plt.xlim([0, len(y_normalizedInc)])
    # plt.ylim([y_normalizedInc[0], y_normalizedInc[-1]])
    plt.title('lambda f')
    plt.plot(range(len(nareas)), nareas)
    plt.plot(range(len(fareas)), fareas)
    plt.show()

    areas = fareas

    kneedle = KneeLocator(range(len(areas)), areas, S=1.0, curve='concave', direction='increasing')

    y_differnce = kneedle.y_difference
    y_normalized = kneedle.y_normalized


    plt.plot()
    plt.xlim([0, len(areas)])
    plt.ylim([0,1])
    plt.title('areas')
    plt.plot(range(len(areas)), areas)
    plt.show()

    plt.plot()
    plt.xlim([0, len(y_differnce)])
    plt.ylim([0, 1])
    plt.title('y_differnce')
    plt.plot(range(len(y_differnce)), y_differnce)
    plt.show()

    if True:
        areasInc = [areas[0]]
        for a in areas[1:]:
            areasInc.append(a)

            kneedleInc = KneeLocator(range(len(areasInc)), areasInc, S=1.0, curve='concave', direction='increasing')

            y_differnceInc = kneedleInc.y_difference
            y_normalizedInc = kneedleInc.y_normalized

            plt.plot()
            plt.xlim([0, len(y_normalizedInc)])
            plt.ylim([y_normalizedInc[0], y_normalizedInc[-1]])
            plt.title('y_normalized vs y_difference                                                     .')
            plt.suptitle(len(areasInc))
            plt.plot(range(len(y_normalizedInc)), y_normalizedInc)
            plt.plot(range(len(y_differnceInc)), y_differnceInc)
            if None is not kneedleInc.knee:
                plt.axvline(x=kneedleInc.knee)
            plt.show()


    print("main")
    '''
