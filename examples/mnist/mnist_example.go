package main

import (
	"fmt"
	"log"
	"time"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func main() {
	// Set maximum number of samples to use
	maxSamples := 54000 // Limit to 54,000 samples
	trainRatio := 0.9   // 90% training, 10% testing

	// Load MNIST data
	data, err := LoadMNISTData(maxSamples, trainRatio)
	if err != nil {
		log.Fatalf("Failed to load MNIST data: %v", err)
	}

	// Configure the Tsetlin Machine for multiclass classification
	config := tsetlin.DefaultConfig()
	config.NumFeatures = len(data.TrainX[0])
	config.NumClasses = 1000
	config.NumClauses = 10
	config.NumLiterals = len(data.TrainX[0])
	config.Threshold = 50.0
	config.S = 10.0
	config.NStates = 100
	config.RandomSeed = 42
	config.Debug = false

	machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		log.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	for i := 0; i < 10; i++ {
		fmt.Printf("\nEpoch %d/%d\n", i+1, 10)

		// Time the training phase
		fmt.Println("Training the Tsetlin Machine...")
		startTime := time.Now()
		if err := machine.Fit(data.TrainX, data.TrainY, 1); err != nil {
			log.Fatalf("Training failed: %v", err)
		}
		trainingTime := time.Since(startTime)

		// Calculate training accuracy
		trainCorrect := 0
		for i := 0; i < len(data.TrainX); i++ {
			pred, err := machine.PredictClass(data.TrainX[i])
			if err != nil {
				log.Printf("Training prediction error: %v", err)
				continue
			}
			if pred == data.TrainY[i] {
				trainCorrect++
			}
		}
		trainAcc := float64(trainCorrect) / float64(len(data.TrainX))
		fmt.Printf("Training accuracy: %.2f%% (%d/%d) in %v\n", trainAcc*100, trainCorrect, len(data.TrainX), trainingTime)

		// Time the evaluation phase
		fmt.Println("Evaluating on test set...")
		startTime = time.Now()
		testCorrect := 0
		for i := 0; i < len(data.TestX); i++ {
			pred, err := machine.PredictClass(data.TestX[i])
			if err != nil {
				log.Printf("Test prediction error: %v", err)
				continue
			}
			if pred == data.TestY[i] {
				testCorrect++
			}
		}
		evalTime := time.Since(startTime)
		testAcc := float64(testCorrect) / float64(len(data.TestX))
		fmt.Printf("Test accuracy: %.2f%% (%d/%d) in %v\n", testAcc*100, testCorrect, len(data.TestX), evalTime)
	}

	// Show predictions for first 10 test samples
	fmt.Println("\nSample predictions from test set:")
	for i := 0; i < 10; i++ {
		pred, err := machine.PredictClass(data.TestX[i])
		if err != nil {
			log.Printf("Prediction error: %v", err)
			continue
		}
		fmt.Printf("Sample %d: True=%d, Predicted=%d\n",
			i, data.TestY[i], pred)
	}
}
