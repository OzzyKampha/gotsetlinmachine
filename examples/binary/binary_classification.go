// Package binary demonstrates binary classification using the Tsetlin Machine.
// This example shows how to use the Tsetlin Machine to solve the XOR problem,
// a classic example of a non-linearly separable classification task.
package binary

import (
	"fmt"
	"log"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// RunBinaryExample demonstrates binary classification using the XOR problem.
// It shows how to:
// 1. Configure a Tsetlin Machine for binary classification
// 2. Train it on the XOR dataset
// 3. Make predictions and analyze the results
// 4. Examine the learned clauses and their states
func RunBinaryExample() {
	// Create configuration for binary classification
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 2  // XOR has 2 inputs
	config.NumClauses = 20  // Increased number of clauses for better learning
	config.NumLiterals = 2  // Number of literals per clause (matches input features)
	config.Threshold = 10.0 // Lowered threshold for better sensitivity
	config.S = 2.5          // Adjusted specificity parameter
	config.NStates = 100    // Number of states for the automata
	config.NumClasses = 2   // Binary classification
	config.RandomSeed = 42  // For reproducibility

	// XOR training data
	X := [][]float64{
		{0, 0}, // 0
		{0, 1}, // 1
		{1, 0}, // 1
		{1, 1}, // 0
	}
	y := []int{0, 1, 1, 0}

	// Create multiclass Tsetlin Machine
	machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		log.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Train the model
	fmt.Println("Training the model...")
	for epoch := 1; epoch <= 1000; epoch++ {
		err := machine.Fit(X, y, 1)
		if err != nil {
			log.Fatalf("Failed to train model: %v", err)
		}
		// Debug: Calculate and print accuracy after each epoch
		correct := 0
		for i, input := range X {
			prediction, err := machine.PredictClass(input)
			if err != nil {
				log.Fatalf("Failed to predict: %v", err)
			}
			if prediction == y[i] {
				correct++
			}
		}
		fmt.Printf("Epoch %d: Accuracy = %.2f%%\n", epoch, float64(correct)/float64(len(X))*100)
	}

	// Test the model
	fmt.Println("\nTesting the model:")
	for i, input := range X {
		prediction, err := machine.PredictClass(input)
		if err != nil {
			log.Fatalf("Failed to predict: %v", err)
		}
		fmt.Printf("Input: %v\n", input)
		fmt.Printf("True class: %d\n", y[i])
		fmt.Printf("Predicted class: %d\n", prediction)
		// Debug: Print clause activations for this input
		for classIdx, m := range machine.GetMachines() {
			active := m.GetActiveClauses(input)
			fmt.Printf("  Class %d active clauses: %v\n", classIdx, active)
		}
		fmt.Println()
	}
}
