// Package main demonstrates multiclass classification using the Tsetlin Machine.
// This example shows how to use the Tsetlin Machine for pattern recognition
// with multiple classes, using the one-vs-all approach.
package main

import (
	"fmt"
	"time"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func calculateMetrics(predictions, actual []int, numClasses int) (float64, []float64, []float64, []float64) {
	// Initialize confusion matrix
	confusion := make([][]int, numClasses)
	for i := range confusion {
		confusion[i] = make([]int, numClasses)
	}

	// Fill confusion matrix
	for i := range predictions {
		confusion[actual[i]][predictions[i]]++
	}

	// Calculate metrics for each class
	precision := make([]float64, numClasses)
	recall := make([]float64, numClasses)
	f1 := make([]float64, numClasses)

	for i := 0; i < numClasses; i++ {
		// True positives
		tp := confusion[i][i]

		// False positives (sum of column i except diagonal)
		fp := 0
		for j := 0; j < numClasses; j++ {
			if j != i {
				fp += confusion[j][i]
			}
		}

		// False negatives (sum of row i except diagonal)
		fn := 0
		for j := 0; j < numClasses; j++ {
			if j != i {
				fn += confusion[i][j]
			}
		}

		// Calculate precision
		if tp+fp > 0 {
			precision[i] = float64(tp) / float64(tp+fp)
		}

		// Calculate recall
		if tp+fn > 0 {
			recall[i] = float64(tp) / float64(tp+fn)
		}

		// Calculate F1 score
		if precision[i]+recall[i] > 0 {
			f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
		}
	}

	// Calculate macro-average F1
	macroF1 := 0.0
	for i := range f1 {
		macroF1 += f1[i]
	}
	macroF1 /= float64(numClasses)

	return macroF1, precision, recall, f1
}

func main() {
	// Training data for pattern recognition
	// Each pattern is represented by 4 binary features
	// Generate 100 distinct binary patterns for classification
	X, y := generateStratifiedSamples(1000, 10, 10)

	fmt.Printf("Loaded %d samples, each with %d bits\n", len(X), len(X[0]))
	fmt.Println("First 10 labels:", y[:10])
	// Create multiclass Tsetlin Machine
	numClasses := len(y)
	numClauses := 1000       // Number of clauses per class
	numFeatures := len(X[0]) // Number of input features
	threshold := 500         // Classification threshold
	s := 2.0                 // Specificity parameter
	//dropoutProb := 0.2 // Probability of dropping a clause during training

	machine := tsetlin.NewMultiClassTM(numClasses, numClauses, numFeatures, threshold, s)

	// Train the model with dropout
	fmt.Println("Training the model...")
	trainstart := time.Now()
	for epoch := 0; epoch < 10; epoch++ {
	for epoch := 0; epoch < 1; epoch++ {
		// Apply dropout for this epoch
		start := time.Now()
		machine.Fit(X, y, 1)
		duration := time.Since(start)
		fmt.Printf("Epoch duration: %v\n", duration)
	}
	duration := time.Since(trainstart)
	fmt.Printf("Traing duration: %v\n", duration)

	// Test the model and calculate metrics
	fmt.Println("\nTesting the model...")
	predictions := make([]int, len(X))
	correct := 0

	predstart := time.Now()
	prediction := machine.PredictBatch(X)
	predtime := time.Since(predstart)

	totalSamples := len(X)
	eps := float64(totalSamples) / predtime.Seconds()

	for i := range X {
		predictions[i] = prediction[i]
		fmt.Printf("Input: %v..., Expected: %d, Predicted: %d\n",
			X[i][:10], y[i], prediction[i])
		if prediction[i] == y[i] {
			correct++
		}
	}
	fmt.Printf("Total prediction time: %v\n", predtime)
	fmt.Printf("Events per second (EPS): %.2f\n", eps)

	accuracy := float64(correct) / float64(len(X)) * 100
	fmt.Printf("\nAccuracy: %.2f%% (%d/%d correct)\n", accuracy, correct, len(X))

	// Calculate and display detailed metrics
	macroF1, precision, recall, f1 := calculateMetrics(predictions, y, numClasses)
	fmt.Println("\nDetailed Metrics:")
	for i := 0; i < numClasses; i++ {
		fmt.Printf("Class %d:\n", i)
		fmt.Printf("  Precision: %.2f%%\n", precision[i]*100)
		fmt.Printf("  Recall: %.2f%%\n", recall[i]*100)
		fmt.Printf("  F1 Score: %.2f%%\n", f1[i]*100)
	}
	fmt.Printf("\nMacro-average F1 Score: %.2f%%\n", macroF1*100)
}
