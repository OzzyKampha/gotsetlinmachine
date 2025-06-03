// Package tsetlin implements the Tsetlin Machine, a novel machine learning algorithm
// that uses propositional logic to learn patterns from data.
package tsetlin

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// MultiClassTsetlinMachine represents a multiclass Tsetlin Machine using one-vs-all approach.
// It implements the Machine interface and provides multiclass classification capabilities
// by using multiple binary Tsetlin Machines, one for each class.
type MultiClassTsetlinMachine struct {
	config   Config
	machines []*BitPackedTsetlinMachine
	wg       sync.WaitGroup
	mu       sync.Mutex
}

// NewMultiClassTsetlinMachine creates a new multiclass Tsetlin Machine.
// It initializes one binary Tsetlin Machine per class using the one-vs-all approach.
func NewMultiClassTsetlinMachine(config Config) (*MultiClassTsetlinMachine, error) {
	if config.NumClasses < 2 {
		return nil, fmt.Errorf("number of classes must be at least 2")
	}
	if config.NumFeatures <= 0 {
		return nil, fmt.Errorf("number of features must be positive")
	}
	if config.NumClauses <= 0 {
		return nil, fmt.Errorf("number of clauses must be positive")
	}

	mctm := &MultiClassTsetlinMachine{
		config:   config,
		machines: make([]*BitPackedTsetlinMachine, config.NumClasses),
	}

	for i := 0; i < config.NumClasses; i++ {
		binaryConfig := config
		binaryConfig.NumClasses = 2
		mctm.machines[i] = NewBitPackedTsetlinMachine(binaryConfig)
	}

	return mctm, nil
}

// Fit trains the multiclass Tsetlin Machine.
// It trains each binary classifier in parallel using worker pools.
func (mctm *MultiClassTsetlinMachine) Fit(X [][]float64, y []int, epochs int) error {
	if len(X) != len(y) {
		return fmt.Errorf("X and y must have the same length")
	}
	if len(X) == 0 {
		return fmt.Errorf("empty training data")
	}
	if len(X[0]) != mctm.config.NumFeatures {
		return fmt.Errorf("input features dimension mismatch: expected %d, got %d",
			mctm.config.NumFeatures, len(X[0]))
	}

	binaryLabels := make([][]int, mctm.config.NumClasses)
	for class := 0; class < mctm.config.NumClasses; class++ {
		binaryLabels[class] = make([]int, len(y))
		for i, label := range y {
			if label == class {
				binaryLabels[class][i] = 1
			} else {
				binaryLabels[class][i] = 0
			}
		}
	}

	numWorkers := runtime.NumCPU() * 2
	workerChan := make(chan int, numWorkers)
	for i := 0; i < numWorkers; i++ {
		workerChan <- i
	}

	for class := 0; class < mctm.config.NumClasses; class++ {
		class := class
		mctm.wg.Add(1)
		go func() {
			defer mctm.wg.Done()
			<-workerChan
			defer func() { workerChan <- 0 }()

			for epoch := 0; epoch < epochs; epoch++ {
				for i, sample := range X {
					mctm.machines[class].Update(sample, binaryLabels[class][i])
				}
			}
		}()
	}
	mctm.wg.Wait()

	return nil
}

// Predict returns the predicted class for the given input pattern.
func (mctm *MultiClassTsetlinMachine) Predict(input []float64) int {
	scores := make([]int, mctm.config.NumClasses)
	for class := 0; class < mctm.config.NumClasses; class++ {
		scores[class] = mctm.machines[class].Predict(input)
	}
	maxScore := scores[0]
	predictedClass := 0
	for class := 1; class < mctm.config.NumClasses; class++ {
		if scores[class] > maxScore {
			maxScore = scores[class]
			predictedClass = class
		}
	}
	return predictedClass
}

// PredictClass returns just the predicted class for the input.
// This is a convenience method when only the class prediction is needed.
func (mctm *MultiClassTsetlinMachine) PredictClass(input []float64) (int, error) {
	result := mctm.Predict(input)
	return result, nil
}

// PredictProba returns probability estimates for each class.
// The probabilities are calculated using softmax on the voting scores.
func (mctm *MultiClassTsetlinMachine) PredictProba(input []float64) ([]float64, error) {
	scores := make([]int, mctm.config.NumClasses)
	for class := 0; class < mctm.config.NumClasses; class++ {
		scores[class] = mctm.machines[class].Predict(input)
	}

	// Convert scores to probabilities using softmax
	expScores := make([]float64, len(scores))
	var sum float64
	for i, score := range scores {
		expScores[i] = math.Exp(float64(score))
		sum += expScores[i]
	}

	probs := make([]float64, len(expScores))
	for i, expScore := range expScores {
		probs[i] = expScore / sum
	}

	return probs, nil
}
