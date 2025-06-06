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
	Y := []int{1, 0, 0, 2} // Only {1,1} → 1

	// Create Tsetlin Machine (binary classifier)
	tm := tsetlin.NewMultiClassTM(
		3,
		500, // numClauses
		2,   // numFeatures
		250, // T threshold (votes)
		3,   // s parameter
	)

	// Train on the dataset
	tm.Fit(X, Y, 1000) // Train to recognize class=1

	// Predict on new inputs

	pred := tm.PredictBatch(X)

	fmt.Printf("Input: %v → Prediction: %d\n", X, pred)

	correct := 0
	for i := range Y {
		if Y[i] == pred[i] {
			correct++
		}
		fmt.Printf("Input: %v, Expected: %d, Predicted: %d\n",
			X, Y[i], pred[i])
	}

}
