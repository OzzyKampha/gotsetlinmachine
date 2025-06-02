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
	machines []*TsetlinMachine
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
	if config.NumLiterals <= 0 {
		return nil, fmt.Errorf("number of literals must be positive")
	}
	if config.NStates <= 0 {
		return nil, fmt.Errorf("number of states must be positive")
	}

	mctm := &MultiClassTsetlinMachine{
		config:   config,
		machines: make([]*TsetlinMachine, config.NumClasses),
	}

	// Initialize one Tsetlin Machine per class
	for i := 0; i < config.NumClasses; i++ {
		binaryConfig := config
		binaryConfig.NumClasses = 2 // Binary classification for each machine
		machine, err := NewTsetlinMachine(binaryConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create binary machine %d: %w", i, err)
		}
		mctm.machines[i] = machine
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

	// Create binary labels for each class
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

	// Train each binary classifier in parallel
	workerChan := make(chan int, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		workerChan <- i
	}

	for class := 0; class < mctm.config.NumClasses; class++ {
		class := class // Create new variable for goroutine
		mctm.wg.Add(1)
		go func() {
			defer mctm.wg.Done()
			<-workerChan                       // Get worker from pool
			defer func() { workerChan <- 0 }() // Return worker to pool

			if err := mctm.machines[class].Fit(X, binaryLabels[class], epochs); err != nil {
				// Note: In a real implementation, you might want to handle these errors differently
				fmt.Printf("warning: error training class %d: %v\n", class, err)
			}
		}()
	}
	mctm.wg.Wait()

	return nil
}

// Predict returns the prediction results for the input.
// It combines predictions from all binary classifiers to make the final prediction.
func (mctm *MultiClassTsetlinMachine) Predict(input []float64) (PredictionResult, error) {
	if len(input) != mctm.config.NumFeatures {
		return PredictionResult{}, fmt.Errorf("input features dimension mismatch: expected %d, got %d",
			mctm.config.NumFeatures, len(input))
	}

	// Get scores from each binary classifier in parallel
	scores := make([]float64, mctm.config.NumClasses)
	workerChan := make(chan int, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		workerChan <- i
	}

	for class := 0; class < mctm.config.NumClasses; class++ {
		class := class // Create new variable for goroutine
		mctm.wg.Add(1)
		<-workerChan // Get worker from pool
		go func() {
			defer mctm.wg.Done()
			defer func() { workerChan <- 0 }() // Return worker to pool

			result, err := mctm.machines[class].Predict(input)
			if err != nil {
				fmt.Printf("warning: error predicting class %d: %v\n", class, err)
				return
			}
			scores[class] = result.Votes[0] // Use positive class score
		}()
	}
	mctm.wg.Wait()

	// Find the class with the highest score
	maxScore := scores[0]
	predictedClass := 0
	secondHighestScore := float64(-1)

	for class := 1; class < mctm.config.NumClasses; class++ {
		if scores[class] > maxScore {
			secondHighestScore = maxScore
			maxScore = scores[class]
			predictedClass = class
		} else if scores[class] > secondHighestScore {
			secondHighestScore = scores[class]
		}
	}

	// Calculate margin and confidence
	margin := math.Abs(maxScore - secondHighestScore)
	maxPossibleVotes := float64(mctm.config.NumClauses)
	confidence := math.Abs(margin / maxPossibleVotes)

	return PredictionResult{
		Votes:          scores,
		PredictedClass: predictedClass,
		Margin:         margin,
		Confidence:     confidence,
	}, nil
}

// PredictClass returns just the predicted class for the input.
// This is a convenience method when only the class prediction is needed.
func (mctm *MultiClassTsetlinMachine) PredictClass(input []float64) (int, error) {
	result, err := mctm.Predict(input)
	if err != nil {
		return 0, err
	}
	return result.PredictedClass, nil
}

// PredictProba returns probability estimates for each class.
// The probabilities are calculated using softmax on the voting scores.
func (mctm *MultiClassTsetlinMachine) PredictProba(input []float64) ([]float64, error) {
	result, err := mctm.Predict(input)
	if err != nil {
		return nil, err
	}

	// Convert scores to probabilities using softmax
	expScores := make([]float64, len(result.Votes))
	var sum float64
	for i, score := range result.Votes {
		expScores[i] = math.Exp(score)
		sum += expScores[i]
	}

	probs := make([]float64, len(expScores))
	for i, expScore := range expScores {
		probs[i] = expScore / sum
	}

	return probs, nil
}

// GetClauseInfo returns information about the clauses in the machine.
// This is useful for analyzing the learned patterns and model interpretability.
func (mctm *MultiClassTsetlinMachine) GetClauseInfo() [][]ClauseInfo {
	mctm.mu.Lock()
	defer mctm.mu.Unlock()

	info := make([][]ClauseInfo, mctm.config.NumClasses)
	for i, machine := range mctm.machines {
		info[i] = machine.GetClauseInfo()[0]
	}
	return info
}

// GetActiveClauses returns information about the active clauses for a given input.
// This helps understand which clauses contributed to the prediction.
func (mctm *MultiClassTsetlinMachine) GetActiveClauses(input []float64) [][]ClauseInfo {
	if len(input) != mctm.config.NumFeatures {
		return nil
	}

	info := make([][]ClauseInfo, mctm.config.NumClasses)
	workerChan := make(chan int, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		workerChan <- i
	}

	for class := 0; class < mctm.config.NumClasses; class++ {
		class := class
		mctm.wg.Add(1)
		go func() {
			defer mctm.wg.Done()
			<-workerChan
			defer func() { workerChan <- 0 }()

			info[class] = mctm.machines[class].GetActiveClauses(input)[0]
		}()
	}
	mctm.wg.Wait()

	return info
}
