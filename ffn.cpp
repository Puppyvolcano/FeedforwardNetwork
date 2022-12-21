// Creates dense feedforward model using hip kernels
// Alden Bauman
// CS691
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <vector>

#define WIDTH 1024
#define HEIGHT 1024

// Generic kernel to find mean squared error
__global__ void meanSquaredError(const float *A, const float *B, float *C, const int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    C[idx] = (A[idx] - B[idx]) * (A[idx] - B[idx]);
  }
}

// Forward propagation kernel
__global__ void forwardProp(const float *inputs, const float *weights, const float *biases, 
                            float *outputs, const int numSamples, const int numInputs, const int numOutputs) {
  // Get the global thread index
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Calculate the output for this thread for each sample
  for (int s = 0; s < numSamples; s++) {
    if (idx < numOutputs) {
      outputs[s * numOutputs + idx] = 0;
      for (int i = 0; i < numInputs; i++) {
        outputs[s * numOutputs + idx] += inputs[s * numInputs + i] * weights[idx * numInputs + i];
      }
			//outputs[s * numOutputs + idx] = 100; // comparison with static bias
      outputs[s * numOutputs + idx] += biases[idx];
    }
  }
}

// Kernel to perform backpropagation and update weights and biases
__global__ void backProp(float learningRate, const float *inputs, float *weights, float *biases,
                         const float *outputs, const float *targets, float *gradients,
                         const int numSamples, const int numInputs, const int numOutputs) {
  // Get the global thread index
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Calculate the gradients for each sample
  for (int s = 0; s < numSamples; s++) {
    if (idx < numOutputs) {
      // Calculate the error for the current output neuron
      float error = targets[s * numOutputs + idx] - outputs[s * numOutputs + idx] * learningRate;

      // Calculate the gradients for the weights and biases
      gradients[idx] = error * outputs[s * numOutputs + idx] * (1 - outputs[s * numOutputs + idx]) * learningRate;
      biases[idx] -= gradients[idx] * learningRate;  // updates biases
			//weights[idx] -= gradients[idx] * learningRate;
      
      // updates weights
      for (int i = 0; i < numInputs; i++) {
        gradients[idx * numInputs + i] = gradients[idx] * inputs[s * numInputs + i];
        weights[idx * numInputs + i] -= gradients[idx * numInputs + i] * learningRate;
      }
    }
  }
}

int main() {
  // Set number of layers and connections
  int numLayers = 4;
  int connections[numLayers] = {60, 30, 60, 60};
	int maxNodes = 0;
	int endNodes = connections[numLayers-1];
  int ignore;  // used to ignore returns for hip functions

	// set seed
	srand(time(0));

	// finds layer with most nodes to make reserving memory easier later
	for (int i = 0; i < numLayers; i++) {
		if (connections[i] > maxNodes) {
			maxNodes = connections[i];
		}
	}

  // declare vars for train and test sets
  int trainSamples = 1000;
  int testSamples = 500;
	int trainSize = trainSamples * sizeof(float);
	int testSize = trainSamples * sizeof(float);

  float trainX[trainSamples];
  float trainY[trainSamples];
  float testX[testSamples];
  float testY[testSamples];

  // possible keys to use for training. Currently using ascii value of numbers 0-9
	//int numKey[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int numKey[] = {(int) '0', (int) '1', (int) '2', (int) '3', (int) '4', 
                  (int) '5', (int) '6', (int) '7', (int) '8', (int) '9'};
                  
  // Generate train and test sets
  for (int i = 0; i < trainSamples; i++) {
    trainX[i] = rand() % 10;
    trainY[i] = numKey[(int) trainX[i]];
  }
  for (int i = 0; i < testSamples; i++) {
    testX[i] = rand() % 10;
    testY[i] = numKey[(int) testX[i]];
  }

	// key vars
	float trainKey[trainSamples*connections[numLayers-1]];
	float trainDat[trainSamples*connections[numLayers-1]];
	float testKey[trainSamples][connections[numLayers-1]];
	//float testDat[testSamples][connections[numLayers-1]];

	// creates training one hot notation for input data and key
	for (int i = 0; i < trainSamples; i++) {
		int index = connections[numLayers-1] * i;
		for (int j = 0; j < connections[numLayers-1]; j++) {
			if ((int)trainY[i] == j) {
				trainKey[index+j] = 1;
				trainDat[index+j] = 1;
			}
			else {
				trainKey[index+j] = 0;
				trainDat[index+j] = 0;
			}
		}
	}

	// creates test one hot notation
	for (int i = 0; i < testSamples; i++) {
		for (int j = 0; j < connections[numLayers-1]; j++) {
			if ((int)testY[i] == j) {
				testKey[i][j] = 1;
			}
			else {
				testKey[i][j] = 0;
			}
		}
	}
	
	// creates default gradient so it can be easily reset
	float defGradient[maxNodes*maxNodes];
	for (int i = 0; i < maxNodes*maxNodes; i++){
		defGradient[i] = 0.0;
	}

	// creates default output so it can be easily reset
	float defOut[trainSamples*endNodes];
	for (int i = 0; i < trainSamples*endNodes; i++){
		defOut[i] = 0.0;
	}

  // Allocate memory on GPU
  int blockSize = 1024;
  dim3 block(blockSize, blockSize);
  dim3 grid((connections[0] + blockSize - 1) / blockSize, (connections[numLayers - 1] + blockSize - 1) / blockSize);

	// reserves weight memory
  //float *layers[numLayers];
	std::vector<float*> layers;
  for (int i = 0; i < numLayers; i++) {
    // alternative memory reservation strategy, should have less waste but makes handling more difficult
		/*if (i == 0){
			ignore = hipMalloc((void **) &layers[i], 1 * connections[i] * sizeof(float));
			for (int i = 0; i < N; i++) {
				a[i] = static_cast<float>(rand()) / 100 % 10;
			}
      // results not as expected, used hipmemcpy instead
			//ignore = hipMemset((void**)layers[i], 5.0, 1 * connections[i] * sizeof(float));
		}
		else {
		  ignore = hipMalloc((void **) &layers[i], connections[i] * connections[i - 1] * sizeof(float));
			ignore = hipMemset((void**)layers[i], 5.0, connections[i] * connections[i - 1] * sizeof(float));
			for (int j = 0; j < N; j++) {
				layers[i][j] = static_cast<float>(rand()) % 100;
			}
		}*/
    
    // vector is filled layer by layer with random data in range -1 to 1
		float *layer;
    ignore = hipMalloc((void **) &layer, maxNodes * maxNodes * sizeof(float));

		// generate initial weights
		float tempWeights[maxNodes * maxNodes];
		for (int i = 0; i < maxNodes * maxNodes; i++) {
			int ran = rand() % 1000;
			tempWeights[i] = (float)(ran - 500) / 1000.0;
			//std::cout << tempWeights[i] << "  ";
		}

		ignore = hipMemcpy((void**)layer, tempWeights, maxNodes * maxNodes * sizeof(float), hipMemcpyHostToDevice);
		// ignore = hipMemcpy((void**)tempWeights, layer, maxNodes * maxNodes * sizeof(float), hipMemcpyDeviceToHost);
		layers.push_back(layer);
  }

	// reserves bias memory
	std::vector<float*> layerBias;
	for (int i = 0; i < numLayers; i++) {
		float *layer;
    ignore = hipMalloc((void**) &layer, maxNodes * maxNodes * sizeof(float));

		// generate initial biases
		float tempWeights[maxNodes * maxNodes];
		for (int i = 0; i < maxNodes * maxNodes; i++) {
			int ran = rand() % 1000;
			tempWeights[i] = (float)(ran - 500) / 10000.0;
			//std::cout << tempWeights[i] << "  ";
		}

		ignore = hipMemcpy((void**)layer, tempWeights, maxNodes * maxNodes * sizeof(float), hipMemcpyHostToDevice);
		// ignore = hipMemcpy((void**)tempWeights, layer, maxNodes * maxNodes * sizeof(float), hipMemcpyDeviceToHost);
		layerBias.push_back(layer);
  }


  // Declare hip pointers
  float *inputs;
  float *outputs;
	float *targets;
	float *gradients;
	float *loss;

	// array that stores network output to check accuracy
	float networkPredictions[connections[numLayers-1] * trainSamples];
	//float tempWeights[];

	// line for debugging
	//std::cout << "Location 1\n";

	// assigns memory for hip arrays based on what is needed
	ignore = hipMalloc((void**)&inputs, trainSamples*maxNodes*sizeof(float));
	ignore = hipMalloc((void**)&outputs, trainSamples*maxNodes*sizeof(float));
	ignore = hipMalloc((void**)&targets, trainSamples*maxNodes*sizeof(float));
	ignore = hipMalloc((void**)&gradients, maxNodes*maxNodes*sizeof(float));
  ignore = hipMalloc((void **) &loss, connections[numLayers - 1] * sizeof(float));

  int trainingEpochs = 100;

  // Train network for desired number of epochs
  for (int epoch = 0; epoch < trainingEpochs; epoch++) {
		//used to view weights in each iteration to gain insight on how they are updated
		if (0==1) {
			float weightViewer[maxNodes*maxNodes];
			ignore = hipMemcpy(weightViewer, layerBias[2], maxNodes * maxNodes * sizeof(float), hipMemcpyDeviceToHost);

			for (int view = 0; view < maxNodes; view++) {
				std::cout << weightViewer[view] << "\t";
			}
		}

		// initialize train and test variables
		float trainLoss = 0.0;
		float trainAccuracy = 0.0;
		float testLoss = 0.0;
		float testAccuracy = 0.0;

		// initialize hip array values
		ignore = hipMemcpy(inputs, trainDat, trainSize*endNodes, hipMemcpyHostToDevice);
		ignore = hipMemcpy(outputs, defOut, trainSize*endNodes, hipMemcpyHostToDevice);
		ignore = hipMemcpy(targets, trainKey, trainSize*endNodes, hipMemcpyHostToDevice);
		ignore = hipMemcpy(gradients, defGradient, maxNodes*maxNodes*sizeof(float), hipMemcpyHostToDevice);
		
	  // Perform forward propagation
	  for (int j = 0; j < numLayers - 1; j++) {
	    forwardProp<<<connections[j]*connections[j+1], blockSize>>>(inputs, layers[j], layerBias[j], outputs, 
			trainSamples, connections[j], connections[j+1]);
      
      // input for next layer is this layers output
			ignore = hipMemcpy(inputs, outputs, trainSize*maxNodes, hipMemcpyDeviceToDevice);
      
      // parameters for forwardprop kernel, kept here as a reference
			//(const float *inputs, const float *weights, const float *biases, 
      //                      float *outputs, const int numSamples, const int numInputs, const int numOutputs)
	  }

		// copies results to local array
		ignore = hipMemcpy(networkPredictions, outputs, trainSize*endNodes, hipMemcpyDeviceToHost);
		
		//std::cout << "Back Prop Started\n";

	  // Perform backward propagation at each layer
	  for (int j = numLayers - 2; j >= 0; j--) {
			backProp<<<connections[j]*connections[j+1], blockSize>>>(0.001, inputs, layers[j+1], layerBias[j+1], 
			outputs, targets, gradients, trainSamples, connections[j+1], connections[j]);

      // next layer's input is this layer's output
			ignore = hipMemcpy(targets, gradients, trainSize*maxNodes, hipMemcpyDeviceToDevice);
			ignore = hipMemcpy(inputs, outputs, trainSize*maxNodes, hipMemcpyDeviceToDevice);
		
      // parameters for backprop kernel, kept here as a reference
			/*backProp(float learningRate, const float *inputs, float *weights, float *biases,
                         const float *outputs, const float *targets, float *gradients,
                         const int numSamples, const int numInputs, const int numOutputs) */
		}

		//ignore = hipMemcpy(networkPredictions, outputs, trainSize*endNodes, hipMemcpyDeviceToHost);
		//std::cout << "Back Prop Finished\n";
			
		// Finds accuracy on train set
		for (int i = 0; i < trainSamples; i++){
			int index=i*connections[numLayers - 1];
			float max = 0.0;
			int maxIndex = 0;
      
      // Find selected output (whichever number has the highest activation
			for (int j = 0; j < connections[numLayers - 1]; j++) {
				if (networkPredictions[index + j] * networkPredictions[index + j] > max) {
					max = networkPredictions[index + j] * networkPredictions[index + j];
					maxIndex = j;
				}
        
        // keeps track of squared loss
				if (trainY[i] == (float)j){
					trainLoss += (trainY[i] - 1) * (trainY[i] - 1);
				}
				else {
					trainLoss += (trainY[i]) * (trainY[i]);
				}
			}
      
      //outputs predictions for troubleshooting purposes
			if (i < 50) {
				//std::cout << networkPredictions[index + maxIndex] << "\t";
				//std::cout << maxIndex << " ";
				//std::cout << maxIndex << " " << trainY[i] << "\n";
			}

			// tally accuracy
			if (trainY[i] == (float)maxIndex) {
				trainAccuracy += 1;
			}
		}

  // averages loss and accuracy
  trainLoss /= (trainSamples * endNodes);
  trainAccuracy /= trainSamples;

  // Print results
  std::cout << "\n\nEpoch " << epoch+1 << " Accuracy = " << trainAccuracy << " Loss: " << trainLoss;
  }

  // Free layer memory
  for (int i = 0; i < numLayers; i++) {
    ignore = hipFree(layers[i]);
		ignore = hipFree(layerBias[i]);
  }

	// free reserved memory
	ignore = hipFree(inputs);
	ignore = hipFree(outputs);
	ignore = hipFree(targets);
	ignore = hipFree(gradients);
	ignore = hipFree(loss);

  return 0;
}
