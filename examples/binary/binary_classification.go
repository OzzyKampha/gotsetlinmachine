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
	config.Debug = true     // Enable debug logging

	// XOR training data
	X := [][]float64{
		{0, 0}, // 0
		{0, 1}, // 1
		{1, 0}, // 1
		{1, 1}, // 0
	}
	y := []int{0, 1, 1, 0}

	// Create multiclass Tsetlin Machine (configured for binary classification)
	machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		log.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Train the model
	fmt.Println("Training the model...")
	err = machine.Fit(X, y, 100)
	if err != nil {
		log.Fatalf("Failed to train model: %v", err)
	}

	// Test the model
	fmt.Println("\nTesting the model...")
	for i, input := range X {
		result, err := machine.Predict(input)
		if err != nil {
			log.Fatalf("Failed to make prediction: %v", err)
		}
		fmt.Printf("Input: %v, Expected: %d, Predicted: %d, Confidence: %.2f\n",
			input, y[i], result.PredictedClass, result.Confidence)
	}

	// Get clause information
	fmt.Println("\nAnalyzing learned clauses...")
	clauseInfo := machine.GetClauseInfo()
	for classIdx, classClauses := range clauseInfo {
		fmt.Printf("\nClass %d Clauses:\n", classIdx)
		for clauseIdx, clause := range classClauses {
			fmt.Printf("Clause %d: ", clauseIdx)
			if clause.IsPositive {
				fmt.Print("Positive, ")
			} else {
				fmt.Print("Negative, ")
			}
			fmt.Printf("Match Score: %.2f, Momentum: %.2f\n", clause.MatchScore, clause.Momentum)
			fmt.Print("Active Literals: ")
			for i, active := range clause.Literals {
				if active {
					fmt.Printf("%d ", i)
				}
			}
			fmt.Println()
		}
	}
}
