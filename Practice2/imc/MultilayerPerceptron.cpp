/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	//TODO: Check
	eta = 1.0;
	mu = 0.7;
	outputFunction = 0;
	online = false;
}


// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) 
{
	nOfLayers = nl;
	layers = new Layer[nl];

	for (int i = 0; i < nl; i++)
	{
		layers[i].nOfNeurons = npl[i];
		layers[i].neurons = new Neuron[layers[i].nOfNeurons];

		for(int j = 0; j < layers[i].nOfNeurons; j++)
		{
			layers[i].neurons[j].out = 0;
			layers[i].neurons[j].delta = 0;

			if(i == 0)
			{
				layers[i].neurons[j].w = NULL;
				layers[i].neurons[j].deltaW = NULL;
				layers[i].neurons[j].lastDeltaW = NULL;
				layers[i].neurons[j].wCopy = NULL;
			}
			
			else
			{
				layers[i].neurons[j].w = new double[layers[i-1].nOfNeurons + 1];
				layers[i].neurons[j].deltaW = new double[layers[i-1].nOfNeurons + 1];
				layers[i].neurons[j].lastDeltaW = new double[layers[i-1].nOfNeurons + 1];
				layers[i].neurons[j].wCopy = new double[layers[i-1].nOfNeurons + 1];
			}
		}
	}
	return 1;
}

// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() 
{
	for (int i = 0; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			delete[] layers[i].neurons[j].w;
			delete[] layers[i].neurons[j].deltaW;
			delete[] layers[i].neurons[j].lastDeltaW;
			delete[] layers[i].neurons[j].wCopy;
		}
		delete[] layers[i].neurons;
	}
	delete[] layers;
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() 
{
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
			{
				layers[i].neurons[j].w[k] = randomDouble(-1, 1);
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) 
{
	for (int i = 0; i < layers[0].nOfNeurons; i++)
	{
		layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{
	for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
	{
		output[i] = layers[nOfLayers - 1].neurons[i].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() 
{
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
			{
				layers[i].neurons[j].wCopy[k] = layers[i].neurons[j].w[k];
			}
		}
	}	
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() 
{
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
			{
				layers[i].neurons[j].w[k] = layers[i].neurons[j].wCopy[k];
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() 
{
	//TODO: Check SOFTMAX in the last layer

	for(int i = 0; i < layers[nOfLayers-1].nOfNeurons; i++)
	{
		for(int j = 0; j < layers[i].nOfNeurons; j++)
		{
			double sum1 = 0.0;
			for(int k = 0; k < layers[i-1].nOfNeurons; k++)
			{
				sum1 += layers[i].neurons[j].w[k] * layers[i-1].neurons[k].out;
			}
			sum1 += layers[i].neurons[j].w[layers[i-1].nOfNeurons];
			
			double sum2 = 0.0;
			for(int k = 0; k < layers[i].nOfNeurons; k++)
			{
				sum2 += exp(sum1);
			}
			layers[nOfLayers-1].neurons[i].out = exp(sum1)/sum2;
		}
	}

	//TODO: CHECK the hidden layers
	for (int i = 1; i < nOfLayers-1; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			double sum = 0;
			for (int k = 0; k < layers[i-1].nOfNeurons; k++)
			{
				sum += layers[i-1].neurons[k].out * layers[i].neurons[j].w[k];
			}
			sum += layers[i].neurons[j].w[layers[i-1].nOfNeurons];
			layers[i].neurons[j].out = 1/(1 + exp(-sum));
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double* target, int errorFunction) 
{
	//TODO: Check
	double error = 0;
	if(errorFunction == 0)
	{
		
		for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		{
			error += pow(target[i] - layers[nOfLayers - 1].neurons[i].out, 2);
		}
		error =  error / layers[nOfLayers - 1].nOfNeurons;
	}

	if(errorFunction == 1)
	{
		
		for(int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		{
			error += target[i] * log(layers[nOfLayers - 1].neurons[i].out);
		}
		error = error / layers[nOfLayers - 1].nOfNeurons;
	}
	return error;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double* target, int errorFunction) 
{
	//TODO: Check errorFunction and outputFunction
	if(errorFunction == 0 && outputFunction == 0)
	{
		for(int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		{
			double newDelta = -(target[i] - layers[nOfLayers - 1].neurons[i].out) * layers[nOfLayers - 1].neurons[i].out * (1 - layers[nOfLayers - 1].neurons[i].out);
			layers[nOfLayers - 1].neurons[i].delta = newDelta;
		}

		for(int i = nOfLayers - 2; i > 0; i--)
		{
			for(int j = 0; j < layers[i].nOfNeurons; j++)
			{
				double sum = 0;
				for(int k = 0; k < layers[i + 1].nOfNeurons; k++)
				{
					sum += layers[i + 1].neurons[k].delta * layers[i + 1].neurons[k].w[j];
				}
				double newDelta = sum * layers[i].neurons[j].out * (1 - layers[i].neurons[j].out);
				layers[i].neurons[j].delta = newDelta;
			}
		}
	}

	if(errorFunction == 0 && outputFunction == 1)
	{
		for(int  i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		{
			double newDelta = -(target[i] / layers[nOfLayers - 1].neurons[i].out) * layers[nOfLayers - 1].neurons[i].out * (1 - layers[nOfLayers - 1].neurons[i].out);
			layers[nOfLayers - 1].neurons[i].delta = newDelta;
		}
		for(int i = nOfLayers - 2; i > 0; i--)
		{
			for(int j = 0; j < layers[i].nOfNeurons; j++)
			{
				double sum = 0;
				for(int k = 0; k < layers[i + 1].nOfNeurons; k++)
				{
					sum += layers[i + 1].neurons[k].delta * layers[i + 1].neurons[k].w[j];
				}
				double newDelta = sum * layers[i].neurons[j].out * (1 - layers[i].neurons[j].out);
				layers[i].neurons[j].delta = newDelta;
			}
		}
	}

	//TODO: Check notation unit 1
	if(errorFunction == 1 && outputFunction == 0)
	{
		for(int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		{
			double sum = 0.0;
			for(int j = 0; j < layers[nOfLayers - 1].nOfNeurons; j++)
			{
				if( i == j)
				{
					sum += (target[i] - layers[nOfLayers-1].neurons[i].out) * layers[nOfLayers-1].neurons[i].out * (1 - layers[nOfLayers-1].neurons[i].out);
				}
				else
				{
					sum += (target[i] - layers[nOfLayers-1].neurons[i].out) * layers[nOfLayers-1].neurons[i].out * (-layers[nOfLayers-1].neurons[i].out);
				}
			}	
		}
		//TODO: ADD Hidden layers
		for(int i = nOfLayers - 2; i > 0; i--)
		{
			for(int j = 0; j < layers[i].nOfNeurons; j++)
			{
				double sum = 0;
				for(int k = 0; k < layers[i + 1].nOfNeurons; k++)
				{
					sum += layers[i + 1].neurons[k].delta * layers[i + 1].neurons[k].w[j];
				}
				double newDelta = sum * layers[i].neurons[j].out * (1 - layers[i].neurons[j].out);
				layers[i].neurons[j].delta = newDelta;
			}
		}
	}

	if(errorFunction == 1 && outputFunction == 1)
	{
		//TODO: ADD Hidden layers
		for(int i = nOfLayers - 2; i > 0; i--)
		{
			for(int j = 0; j < layers[i].nOfNeurons; j++)
			{
				double sum = 0;
				for(int k = 0; k < layers[i + 1].nOfNeurons; k++)
				{
					sum += layers[i + 1].neurons[k].delta * layers[i + 1].neurons[k].w[j];
				}
				double newDelta = sum * layers[i].neurons[j].out * (1 - layers[i].neurons[j].out);
				layers[i].neurons[j].delta = newDelta;
			}
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() 
{
	//TODO: Check
	for(int i = 1; i < nOfLayers; i++)
	{
		for(int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i - 1].nOfNeurons; k++)
			{
				layers[i].neurons[j].deltaW[k] += layers[i].neurons[j].delta * layers[i - 1].neurons[k].out;
			}
			layers[i].neurons[j].deltaW[layers[i - 1].nOfNeurons] += layers[i].neurons[j].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() 
{
	//TODO: Check
	int N = nOfTrainingPatterns;
	for (int i = 1; i < nOfLayers; i++)
	{
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (int k = 0; k < layers[i - 1].nOfNeurons; k++)
			{
				layers[i].neurons[j].w[k] -= (eta * layers[i].neurons[j].deltaW[k])/N + (mu * eta * layers[i].neurons[j].lastDeltaW[k])/N;
			}
			double value = (eta * layers[i].neurons[j].deltaW[layers[i - 1].nOfNeurons - 1])/N + (mu * eta * layers[i].neurons[j].lastDeltaW[layers[i - 1].nOfNeurons - 1])/N;
			layers[i].neurons[j].w[layers[i - 1].nOfNeurons - 1] = layers[i].neurons[j].w[layers[i - 1].nOfNeurons - 1] - value;
		}
	}	
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() 
{
	for(int i =1; i < nOfLayers; i++)
	{
		cout<<"Layer "<<i<<endl;
		for(int j = 0; j < layers[i].nOfNeurons; j++)
		{
			cout<<"Neuron "<<j<<endl;
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
			{
				cout << layers[i].neurons[j].w[k] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction) 
{
	//TODO: Check
	for(int i = 1; i < nOfLayers; i++)
	{
		for(int j = 0; j < layers[i].nOfNeurons; j++)
		{
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
			{
				layers[i].neurons[j].deltaW[k] = 0.0;
			}
		}
	}

	feedInputs(input);
	forwardPropagate();
	obtainError(target, errorFunction);
	backpropagateError(target, errorFunction);
	accumulateChange();
	if (online == true)
	{
		weightAdjustment();
	}
}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) 
{
	//TODO: Check
	for(int i = 0; i < trainDataset->nOfPatterns; i++)
	{
		performEpoch(trainDataset->inputs[i], trainDataset->outputs[i], errorFunction);
	}
	if(online == false)
	{
		weightAdjustment();
	}
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) 
{
	//TODO: Check
	double Error = 0;

	for(int i = 0; i < dataset->nOfPatterns; i++)
	{
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		Error += obtainError(dataset->outputs[i], errorFunction);
	}

	if(errorFunction == 0)
	{
		Error = Error / dataset->nOfPatterns;
	}
	if(errorFunction == 1)
	{
		Error = Error / -(dataset->nOfPatterns);
	}

	return Error;
}


// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) 
{
	//TODO: Check
	double CCR = 0;

	for(int i = 0; i < dataset->nOfPatterns; i++)
	{
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		CCR += obtainError(dataset->outputs[i], 1);
	}

	CCR = CCR/-(dataset->nOfPatterns);
	return CCR;
}


// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset* dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Category" << endl;
	
	for (i=0; i<dataset->nOfPatterns; i++){

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}



// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;


	// Learning
	do {

		train(trainDataset,errorFunction);

		double trainError = test(trainDataset,errorFunction);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while ( countTrain<maxiter );

	if ( iterWithoutImproving!=50)
		restoreWeights();

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		double* prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	*errorTest=test(testDataset,errorFunction);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(testDataset);
	*ccrTrain = testClassification(trainDataset);

}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(layers[i].neurons[j].w!=NULL)
				    f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * fileName)
{
	// Object for reading a file
	ifstream f(fileName);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(!(outputFunction==1 && (i==(nOfLayers-1)) && (k==(layers[i].nOfNeurons-1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
