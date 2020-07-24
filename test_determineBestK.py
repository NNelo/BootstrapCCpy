from kneed import KneeLocator
import numpy as np
import warnings

'''
This method is used to find a better n given an array of areas corresponding to CDF
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
def determineBestK(areas, verbose=False):
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

    ## True if the list is NOT empty
    if knees:
        knees = np.array(knees)
        unique, counts = np.unique(knees[knees != np.array(None)], return_counts=True)
        incResults = dict(zip(unique, counts))

        mostLikelyPoint = max(incResults, key=incResults.get)
        mostLikelyPointPosition = np.where(knees == mostLikelyPoint)[0][0]

        ## TODO: return a second likely point would be considered as an option

        if verbose:
            print("Count of increasing cluster number", incResults)
            print("First most likely point found at:", mostLikelyPointPosition)

        ## If there is at least 2 likely points
        ## If the chosen point is the greatest option, one more check is done
        if len(incResults.keys()) > 1 and max(incResults.keys()) == mostLikelyPoint:
            ## It includes the most frequent point and the next one

            ## the optimal points found should be sorted.
            ## let's supose [1,1,2,2,3,2] is not allowed to happen
            assert np.array_equal(knees[knees != np.array(None)], np.sort(knees[knees != np.array(None)]))

            ## Looking for the greatest's previus key
            mostLikelyPointNeig = mostLikelyPoint - 1

            if verbose:
                print("full knees array", knees)

            ## Compare the number of times the greatest cluster number againts the second one

            while mostLikelyPointNeig not in incResults.keys() and 0 != mostLikelyPointNeig:
                mostLikelyPointNeig -= 1

            if 0 != mostLikelyPointNeig:
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
                    print("most frequent point is the greatest")
                    print("previous most likely point:", mostLikelyPoint)
                    print("reevaluating over the following areas:", areasReduced)
                try:
                    k = KneeLocator(range(len(areasReduced)), areasReduced, S=1.0, curve='concave', direction='increasing')
                    if verbose: print("new knee", k.knee)
                    mostLikelyPoint = k.knee
                except:
                    if verbose: print("new knee not found")

            elif verbose:
                print("mostLikelyPointNeig is zero")

    else:
        print("No knee found")
        mostLikelyPoint = 0

    ## back to just showing the warnings
    warnings.filterwarnings('default')

    return mostLikelyPoint

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

    areas = areas5

    print("*****************************************************")
    print("Gave 3, should 3:", determineBestK(areas3, verbose=True) + 2)
    print("*****************************************************")
    print("Gave 5, should 5:", determineBestK(areas5, verbose=True) + 2)
    print("*****************************************************")
    print("Gave 4, should 3:", determineBestK(areas4, verbose=True) + 2)
    print("*****************************************************")
    print("Gave 5, should 3/5:", determineBestK(areasp, verbose=True) + 2)


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
