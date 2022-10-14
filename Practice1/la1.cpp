//============================================================================
// Introduction to computational models
// Name        : la1.cpp
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

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"


using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Process arguments of the command line
    system("clear");
    bool Tflag = 0, wflag = 0, pflag = 0, tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0, sflag = 0;
    char *Tvalue = NULL, *wvalue = NULL, *pvalue = NULL, *tvalue = NULL, *svalue = NULL;

    char *nIterations = NULL, *nHidden = NULL, *nNeuronsLayer = NULL, *nEta = NULL, *nMu = NULL;
    int c;

    int nIterationsValue = 1000, nHiddenValue = 1, nNeuronsLayerValue = 5;
    double eta = 0.1, mu = 0.9;

    opterr = 0;
    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:p:s")) != -1)
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
                svalue = optarg;
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
        if(!tflag){
            cout << "Error: Missing arguments" << endl;
            return EXIT_FAILURE;
        }

        if(!Tflag)
        {
            Tvalue = tvalue;
        }

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;
        // Parameters of the mlp. For example, mlp.eta = value;

        // Read training and test data: call to util::readData(...)
  
    	Dataset * trainDataset = readData(tvalue);
    	Dataset * testDataset = readData(Tvalue);
        
        if(sflag)
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

        //TODO: Check
        int layers = nHiddenValue + 2;

    	int * topology = new int[nNeuronsLayerValue];
        topology[0] = trainDataset->nOfInputs;

        for(int i = 1; i < nHiddenValue + 1; i++)
        {
            topology[i] = nNeuronsLayerValue;
        }
        topology[layers-1] = trainDataset->nOfOutputs;

        // Initialize the network using the topology vector
        mlp.initialize(layers,topology);

        mlp.eta = eta;
        mlp.mu = mu;

        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = 1;
        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset,testDataset,nIterationsValue,&(trainErrors[i]),&(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }
        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;


        for(int i = 0; i < 5; i++)
        {
            averageTestError += testErrors[i];
            averageTrainError += trainErrors[i];
        }

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;


        return EXIT_SUCCESS;
    }
    
    else 
    {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to util::readData(...)
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

