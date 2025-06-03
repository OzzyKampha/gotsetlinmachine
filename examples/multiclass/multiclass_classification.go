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

	// Train the model one epoch at a time
	epochs := 10
	for epoch := 1; epoch <= epochs; epoch++ {
		if err := machine.Fit(X, y, 1); err != nil {
			log.Fatalf("Failed to train model: %v", err)
		}

		// Compute training accuracy
		correct := 0
		for i, input := range X {
			pred := machine.Predict(input)
			if pred == y[i] {
				correct++
			}
		}
		accuracy := float64(correct) / float64(len(X)) * 100
		fmt.Printf("Epoch %d: Training accuracy = %.2f%%\n", epoch, accuracy)
	}

	// Test the model with both training and new patterns
	testPatterns := [][]float64{
		{1, 1, 0, 0}, // Should be class 0
		{0, 0, 1, 1}, // Should be class 1
		{1, 0, 1, 0}, // Should be class 2
		{1, 1, 1, 1}, // New pattern
		{0, 0, 0, 0}, // New pattern
	}

	fmt.Println("\nTesting the model:")
	for _, input := range testPatterns {
		prediction := machine.Predict(input)
		fmt.Printf("Input: %v\n", input)
		fmt.Printf("Predicted class: %d\n", prediction)
		fmt.Println()
	}
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
