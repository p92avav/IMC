//============================================================================
// Introduction to computational models
// Name        : la2.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"


using namespace imc;
using namespace std;

int main(int argc, char **argv) {
	// Process the command line
system("clear");
    bool Tflag = 0, wflag = 0, pflag = 0, tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0, oflag = 0, fflag = 0, sflag = 0, nflag =0;
    char *Tvalue = NULL, *wvalue = NULL, *pvalue = NULL, *tvalue = NULL, *svalue = NULL;

    char *nIterations = NULL, *nHidden = NULL, *nNeuronsLayer = NULL, *nEta = NULL, *nMu = NULL;
    int c;

    int nIterationsValue = 1000, nHiddenValue = 1, nNeuronsLayerValue = 5, error = 0;
    bool online = false;
    double eta = 1, mu = 0.7;

    opterr = 0;
    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:o:f:p:n:s::")) != -1)
    {
        switch(c){
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 't':
                tflag = true;
                tvalue = optarg;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case 'i':
                iflag = true;
                nIterations = optarg;
                nIterationsValue = atoi(nIterations);
                break;
            case 'l':
                lflag = true;
                nHidden = optarg;
                nHiddenValue = atoi(nHidden);
                break;
            case 'h':
                hflag = true;
                nNeuronsLayer = optarg;
                nNeuronsLayerValue = atoi(nNeuronsLayer);
                break;
            case 'e':
                eflag = true;
                nEta = optarg;
                eta = atof(nEta);
                break;
            case 'm':
                mflag = true;
                nMu = optarg;
                mu = atof(nMu);
                break;
            case 's':
                sflag = true;
                break;
            case 'n':
                nflag = true;
                break;
            case 'o':
                oflag = true;
                break;
            case 'f':
                fflag = true;
                error = atoi(optarg);
                break;
            case '?':
                if (optopt == 't' || optopt == 'w' || optopt == 'p')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }


    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////
        if(!tflag)
        {
            cout << "Error: Missing arguments" << endl;
            return EXIT_FAILURE;
        }
        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value
        mlp.eta = eta;
        mlp.mu = mu;


    	// Type of error considered
    	if(oflag == true)
        {
            mlp.online = true;
        }
        else
        {
            mlp.online = false;
        }

        if(sflag == true)
        {
            mlp.outputFunction = 1;
        }   
        else
        {
            mlp.outputFunction = 0;
        }

        // Read training and test data: call to util::readData(...)
    	Dataset * trainDataset = readData(tvalue);
    	Dataset * testDataset = readData(Tvalue);

        int maxIter = nIterationsValue;

        if(nflag)
        {
            double *minTrainDatasetInputs = minDatasetInputs(trainDataset);
            double *maxTrainDatasetInputs = maxDatasetInputs(trainDataset);
            minMaxScalerDataSetInputs(trainDataset, -1, +1, minTrainDatasetInputs, maxTrainDatasetInputs);

            double minTrainDatasetOutputs = minDatasetOutputs(trainDataset);
            double maxTrainDatasetOutputs = maxDatasetOutputs(trainDataset);
            minMaxScalerDataSetOutputs(trainDataset, 0, +1, minTrainDatasetOutputs, maxTrainDatasetOutputs);

            double *minTestDatasetInputs = minDatasetInputs(testDataset);
            double *maxTestDatasetInputs = maxDatasetInputs(testDataset);
            minMaxScalerDataSetInputs(testDataset, -1, +1, minTestDatasetInputs, maxTestDatasetInputs);
            
            double minTestDatasetOutputs = minDatasetOutputs(testDataset);
            double maxTestDatasetOutputs = maxDatasetOutputs(testDataset);
            minMaxScalerDataSetOutputs(testDataset, 0, +1, minTestDatasetOutputs, maxTestDatasetOutputs);        
        }

        // Initialize topology vector
        //int *topology = new int[layers+2];
        //topology[0] = trainDataset->nOfInputs;
        //for(int i=1; i<(layers+2-1); i++)
        //    topology[i] = neurons;
        //topology[layers+2-1] = trainDataset->nOfOutputs;
        //mlp.initialize(layers+2,topology);

        int layers = nHiddenValue + 2;
        int *topology = new int[layers];

        topology[0] = trainDataset->nOfInputs;

        for(int i=1; i<(layers+2-1); i++)
            topology[i] = nNeuronsLayerValue;

        topology[layers-1] = trainDataset->nOfOutputs;

        mlp.initialize(layers,topology);

		// Seed for random numbers
		int seeds[] = {1,2,3,4,5};
		double *trainErrors = new double[5];
		double *testErrors = new double[5];
		double *trainCCRs = new double[5];
		double *testCCRs = new double[5];
		double bestTestError = DBL_MAX;
		for(int i=0; i<5; i++){
			cout << "**********" << endl;
			cout << "SEED " << seeds[i] << endl;
			cout << "**********" << endl;
			srand(seeds[i]);
			mlp.runBackPropagation(trainDataset,testDataset,maxIter,&(trainErrors[i]),&(testErrors[i]),&(trainCCRs[i]),&(testCCRs[i]),error);
			cout << "We end!! => Final test CCR: " << testCCRs[i] << endl;

			// We save the weights every time we find a better model
			if(wflag && testErrors[i] <= bestTestError)
			{
				mlp.saveWeights(wvalue);
				bestTestError = testErrors[i];
			}
		}


		double trainAverageError = 0, trainStdError = 0;
		double testAverageError = 0, testStdError = 0;
		double trainAverageCCR = 0, trainStdCCR = 0;
		double testAverageCCR = 0, testStdCCR = 0;

        // Obtain training and test averages and standard deviations

        for(int i=0; i<5; i++)
        {
            trainAverageError += trainErrors[i];
            testAverageError += testErrors[i];
            trainAverageCCR += trainCCRs[i];
            testAverageCCR += testCCRs[i];
        }

        trainAverageError /= 5;
        testAverageError /= 5;
        trainAverageCCR /= 5;
        testAverageCCR /= 5;

        for(int i=0; i<5; i++)
        {
            trainStdError += pow(trainErrors[i] - trainAverageError, 2);
            testStdError += pow(testErrors[i] - testAverageError, 2);
            trainStdCCR += pow(trainCCRs[i] - trainAverageCCR, 2);
            testStdCCR += pow(testCCRs[i] - testAverageCCR, 2);
        }

        trainStdError = sqrt(trainStdError/5);
        testStdError = sqrt(testStdError/5);
        trainStdCCR = sqrt(trainStdCCR/5);
        testStdCCR = sqrt(testStdCCR/5);

        trainAverageCCR = trainAverageCCR*100;
        testAverageCCR = testAverageCCR*100;
        trainStdCCR = trainStdCCR*100;
        testStdCCR = testStdCCR*100;

		cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

		cout << "FINAL REPORT" << endl;
		cout << "*************" << endl;
	    cout << "Train error (Mean +- SD): " << trainAverageError << " +- " << trainStdError << endl;
	    cout << "Test error (Mean +- SD): " << testAverageError << " +- " << testStdError << endl;
	    cout << "Train CCR (Mean +- SD): " << trainAverageCCR << " +- " << trainStdCCR << endl;
	    cout << "Test CCR (Mean +- SD): " << testAverageCCR << " +- " << testStdCCR << endl;
		return EXIT_SUCCESS;
    } else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // You do not have to modify anything from here.
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to readData(...)
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;

	}
}

