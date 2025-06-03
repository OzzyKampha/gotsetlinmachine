// Package main demonstrates multiclass classification using the Tsetlin Machine.
// This example shows how to use the Tsetlin Machine for pattern recognition
// with multiple classes, using the one-vs-all approach.
package main

import (
	"fmt"
	"log"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func main() {
	// Create configuration for multiclass classification
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 4  // Number of input features
	config.NumClasses = 3   // Number of classes
	config.NumClauses = 20  // Number of clauses per class
	config.NumLiterals = 4  // Number of literals per clause
	config.Threshold = 10.0 // Classification threshold
	config.S = 3.9          // Specificity parameter
	config.NStates = 100    // Number of states for the automata
	config.RandomSeed = 42  // For reproducibility
	config.Debug = true     // Enable debug logging

	// Create multiclass Tsetlin Machine
	machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		log.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Training data for pattern recognition
	// Each pattern is represented by 4 binary features
	X := [][]float64{
		// Pattern 1: [1,1,0,0] -> Class 0
		{1, 1, 0, 0},
		{1, 1, 0, 1},
		{1, 1, 1, 0},
		// Pattern 2: [0,0,1,1] -> Class 1
		{0, 0, 1, 1},
		{1, 0, 1, 1},
		{0, 1, 1, 1},
		// Pattern 3: [1,0,1,0] -> Class 2
		{1, 0, 1, 0},
		{0, 1, 0, 1},
		{1, 0, 0, 1},
	}
	y := []int{0, 0, 0, 1, 1, 1, 2, 2, 2}

	// Train the model
	fmt.Println("Training the model...")
	err = machine.Fit(X, y, 10)
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

	// Analyze learned clauses
	fmt.Println("\nAnalyzing learned clauses...")
	clauseInfo := machine.GetClauseInfo()
	for classIdx, classClauses := range clauseInfo {
		fmt.Printf("\nClass %d Clauses:\n", classIdx)
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

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
