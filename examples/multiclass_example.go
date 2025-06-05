package main

import (
	"fmt"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func main() {
	// Set random seed for reproducibility
	// Dummy dataset: 2 features -> simple AND logic
	X := [][]int{
		{1, 1},
		{1, 0},
		{0, 1},
		{0, 0},
	}
	Y := []int{0, 0, 0, 0} // Only {1,1} → 1

	// Create Tsetlin Machine (binary classifier)
	tm := tsetlin.NewMultiClassTM(
		2,
		500, // numClauses
		2,   // numFeatures
		15,  // T threshold (votes)
		3,   // s parameter
	)

	// Train on the dataset
	tm.Fit(X, Y, 400) // Train to recognize class=1

	// Predict on new inputs
	for _, x := range X {
		pred := tm.Predict(x)
		fmt.Printf("Input: %v → Prediction: %d\n", x, pred)
	}
}
