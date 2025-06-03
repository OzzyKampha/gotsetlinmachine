package mnist

import (
	"fmt"
	"log"
	"time"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// RunMNISTExample demonstrates multiclass classification using the MNIST dataset.
// It shows how to:
// 1. Configure a Tsetlin Machine for multiclass classification
// 2. Train it on the MNIST dataset
// 3. Make predictions and analyze the results
// 4. Examine the learned clauses and their states
func RunMNISTExample() {
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
	config.NumClasses = 10  // MNIST has 10 classes (digits 0-9)
	config.NumClauses = 100 // Number of clauses per class
	config.NumLiterals = len(data.TrainX[0])
	config.Threshold = 50.0
	config.S = 10.0
	config.NStates = 100
	config.RandomSeed = 42
	config.Debug = true

	// Create multiclass Tsetlin Machine
	machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		log.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Train the model
	fmt.Println("Training the model...")
	startTime := time.Now()
	err = machine.Fit(data.TrainX, data.TrainY, 10)
	if err != nil {
		log.Fatalf("Failed to train model: %v", err)
	}
	trainingTime := time.Since(startTime)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Test the model
	fmt.Println("\nTesting the model...")
	correct := 0
	total := len(data.TestX)
	for i, input := range data.TestX {
		result, err := machine.Predict(input)
		if err != nil {
			log.Fatalf("Failed to make prediction: %v", err)
		}
		if result.PredictedClass == data.TestY[i] {
			correct++
		}
		if (i+1)%1000 == 0 {
			fmt.Printf("Processed %d/%d test samples\n", i+1, total)
		}
	}

	// Print results
	accuracy := float64(correct) / float64(total) * 100
	fmt.Printf("\nTest Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, total)

	// Analyze learned clauses
	fmt.Println("\nAnalyzing learned clauses...")
	clauseInfo := machine.GetClauseInfo()
	for classIdx, classClauses := range clauseInfo {
		fmt.Printf("\nClass %d (Digit %d) Clauses:\n", classIdx, classIdx)
		activeClauses := 0
		for _, clause := range classClauses {
			activeLiterals := 0
			for _, active := range clause.Literals {
				if active {
					activeLiterals++
				}
			}
			if activeLiterals > 0 {
				activeClauses++
			}
		}
		fmt.Printf("Active Clauses: %d/%d\n", activeClauses, len(classClauses))
	}
}
