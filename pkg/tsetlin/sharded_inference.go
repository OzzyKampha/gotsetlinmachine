package tsetlin

import (
	"math"
	"runtime"
	"sync"
)

// InferenceResult holds the results from a single worker's computation
type InferenceResult struct {
	Votes        []int
	MatchScores  []float64
	ClauseCounts []int
}

// ShardedInference represents a sharded inference system for multiclass Tsetlin Machines.
type ShardedInference struct {
	machines []*BitPackedTsetlinMachine
	config   Config
}

// NewShardedInference creates a new ShardedInference system.
func NewShardedInference(machines []*BitPackedTsetlinMachine, config Config) *ShardedInference {
	return &ShardedInference{
		machines: machines,
		config:   config,
	}
}

// Predict returns the predicted class for the given input pattern.
func (si *ShardedInference) Predict(input []float64) int {
	bitInput := FromFloat64Slice(input)
	scores := make([]int, len(si.machines))
	for i, machine := range si.machines {
		scores[i] = machine.PredictBitVec(bitInput)
	}
	maxScore := scores[0]
	predictedClass := 0
	for class := 1; class < len(scores); class++ {
		if scores[class] > maxScore {
			maxScore = scores[class]
			predictedClass = class
		}
	}
	return predictedClass
}

// PredictBatch returns the predicted classes for a batch of input patterns.
func (si *ShardedInference) PredictBatch(X [][]float64) []int {
	predictions := make([]int, len(X))
	for i, input := range X {
		predictions[i] = si.Predict(input)
	}
	return predictions
}

// PredictBatchParallel returns the predicted classes for a batch of input patterns in parallel.
func (si *ShardedInference) PredictBatchParallel(X [][]float64) []int {
	predictions := make([]int, len(X))
	numWorkers := runtime.NumCPU() * 2
	workerChan := make(chan int, numWorkers)
	for i := 0; i < numWorkers; i++ {
		workerChan <- i
	}

	var wg sync.WaitGroup
	for i := 0; i < len(X); i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-workerChan
			defer func() { workerChan <- 0 }()
			predictions[i] = si.Predict(X[i])
		}()
	}
	wg.Wait()
	return predictions
}

// PredictBatchParallelWithCallback returns the predicted classes for a batch of input patterns in parallel,
// and calls the callback function for each prediction.
func (si *ShardedInference) PredictBatchParallelWithCallback(X [][]float64, callback func(int, *BitPackedTsetlinMachine)) []int {
	predictions := make([]int, len(X))
	numWorkers := runtime.NumCPU() * 2
	workerChan := make(chan int, numWorkers)
	for i := 0; i < numWorkers; i++ {
		workerChan <- i
	}

	var wg sync.WaitGroup
	for i := 0; i < len(X); i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-workerChan
			defer func() { workerChan <- 0 }()
			predictions[i] = si.Predict(X[i])
			callback(i, si.machines[predictions[i]])
		}()
	}
	wg.Wait()
	return predictions
}

// PredictBatchParallelWithCallbackAndError returns the predicted classes for a batch of input patterns in parallel,
// and calls the callback function for each prediction, with error handling.
func (si *ShardedInference) PredictBatchParallelWithCallbackAndError(X [][]float64, callback func(int, *BitPackedTsetlinMachine) error) ([]int, error) {
	predictions := make([]int, len(X))
	numWorkers := runtime.NumCPU() * 2
	workerChan := make(chan int, numWorkers)
	for i := 0; i < numWorkers; i++ {
		workerChan <- i
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	var err error
	for i := 0; i < len(X); i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-workerChan
			defer func() { workerChan <- 0 }()
			predictions[i] = si.Predict(X[i])
			if err := callback(i, si.machines[predictions[i]]); err != nil {
				mu.Lock()
				if err == nil {
					err = err
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	return predictions, err
}

// computeConfidence calculates the confidence score for the prediction
func computeConfidence(votes []float64, matchScores []float64, clauseCounts []int, predictedClass int) float64 {
	// Find second highest vote count
	secondMax := math.Inf(-1)
	maxVotes := votes[predictedClass]
	for i, v := range votes {
		if i != predictedClass && v > secondMax {
			secondMax = v
		}
	}

	// Calculate margin
	margin := maxVotes - secondMax
	maxPossibleVotes := float64(clauseCounts[predictedClass])
	if maxPossibleVotes == 0 {
		return 0.0
	}

	// Calculate average match score
	avgScore := 0.0
	if clauseCounts[predictedClass] > 0 {
		avgScore = matchScores[predictedClass] / float64(clauseCounts[predictedClass])
	}

	// Combine margin and average score with proper weighting
	confidence := 0.7*(margin/maxPossibleVotes) + 0.3*avgScore

	// Clamp between 0 and 1
	return math.Max(0.0, math.Min(1.0, confidence))
}
