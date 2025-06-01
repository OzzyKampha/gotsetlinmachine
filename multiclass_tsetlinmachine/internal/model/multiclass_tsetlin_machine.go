package model

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"
)

// PredictionResult holds the prediction results including votes for each class and the predicted class
type PredictionResult struct {
	Votes          []float64 // Votes/scores for each class
	PredictedClass int       // The predicted class
	Margin         float64   // Difference between highest and second highest votes
	Confidence     float64   // Normalized margin (margin / max_possible_votes)
}

// String returns a formatted string representation of the prediction result
func (pr PredictionResult) String() string {
	var votesStr string
	for i, votes := range pr.Votes {
		if i > 0 {
			votesStr += ", "
		}
		votesStr += fmt.Sprintf("Class %d: %d votes", i, int(votes))
	}
	return fmt.Sprintf("Votes: [%s], Predicted Class: %d, Margin: %.2f, Confidence: %.2f",
		votesStr, pr.PredictedClass, pr.Margin, pr.Confidence)
}

// MultiClassTsetlinMachine represents a multiclass Tsetlin Machine using one-vs-all approach
type MultiClassTsetlinMachine struct {
	// Number of classes
	numClasses int
	// Number of features
	numFeatures int
	// Number of clauses per class
	numClauses int
	// Number of literals per clause
	numLiterals int
	// Threshold for classification
	threshold float64
	// Specificity parameter
	s float64
	// Array of binary Tsetlin Machines (one per class)
	machines []*TsetlinMachine
	// WaitGroup for synchronization
	wg sync.WaitGroup
	// Mutex for thread-safe operations
	mu sync.Mutex
	// Debug flag for logging
	debug bool
	// Number of workers for parallel processing
	numWorkers int
}

// NewMultiClassTsetlinMachine creates a new multiclass Tsetlin Machine
func NewMultiClassTsetlinMachine(numClasses, numFeatures, numClauses, numLiterals int, threshold, s float64, nStates int) *MultiClassTsetlinMachine {
	mctm := &MultiClassTsetlinMachine{
		numClasses:  numClasses,
		numFeatures: numFeatures,
		numClauses:  numClauses,
		numLiterals: numLiterals,
		threshold:   threshold,
		s:           s,
		debug:       false,            // Debug logging off by default
		numWorkers:  runtime.NumCPU(), // Use number of CPU cores
	}

	// Initialize one Tsetlin Machine per class
	mctm.machines = make([]*TsetlinMachine, numClasses)
	for i := 0; i < numClasses; i++ {
		// For binary classification, we only need one class
		mctm.machines[i] = NewTsetlinMachine(2, numFeatures, numClauses, numLiterals, threshold, s)
		mctm.machines[i].SetNStates(nStates)
	}

	return mctm
}

// SetRandomState sets the random seed for all machines
func (mctm *MultiClassTsetlinMachine) SetRandomState(seed int64) {
	for _, machine := range mctm.machines {
		machine.SetRandomState(seed)
	}
}

// SetNStates sets the number of states for all machines
func (mctm *MultiClassTsetlinMachine) SetNStates(nStates int) {
	for _, machine := range mctm.machines {
		machine.SetNStates(nStates)
	}
}

// SetS sets the specificity parameter for all machines
func (mctm *MultiClassTsetlinMachine) SetS(s float64) {
	mctm.s = s
	for _, machine := range mctm.machines {
		machine.SetS(s)
	}
}

// SetDebug sets the debug flag for all machines
func (mctm *MultiClassTsetlinMachine) SetDebug(debug bool) {
	mctm.debug = debug
	for _, machine := range mctm.machines {
		machine.SetDebug(debug)
	}
}

// Fit trains the multiclass Tsetlin Machine
func (mctm *MultiClassTsetlinMachine) Fit(X [][]float64, y []int, epochs int) {
	if len(X) != len(y) {
		panic("X and y must have the same length")
	}

	// Create binary labels for each class
	binaryLabels := make([][]int, mctm.numClasses)
	for class := 0; class < mctm.numClasses; class++ {
		binaryLabels[class] = make([]int, len(y))
		for i, label := range y {
			if label == class {
				binaryLabels[class][i] = 1
			} else {
				binaryLabels[class][i] = 0
			}
		}
	}

	// Train each binary classifier in parallel using global worker pool
	for class := 0; class < mctm.numClasses; class++ {
		class := class // Create new variable for goroutine
		globalWorkerPool.Submit(func() {
			mctm.machines[class].Fit(X, binaryLabels[class], epochs)
		})
	}
	globalWorkerPool.Wait()
}

// Predict returns the prediction results including votes for each class and the predicted class
func (mctm *MultiClassTsetlinMachine) Predict(input []float64) PredictionResult {
	// Get scores from each binary classifier in parallel using worker pool
	scores := make([]float64, mctm.numClasses)
	var wg sync.WaitGroup
	wg.Add(mctm.numClasses)

	// Create a pool of workers for processing classes
	workerChan := make(chan int, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		workerChan <- i
	}

	// Process each class in parallel
	for class := 0; class < mctm.numClasses; class++ {
		class := class // Create new variable for goroutine
		<-workerChan   // Get worker from pool
		go func() {
			defer wg.Done()
			defer func() { workerChan <- 0 }() // Return worker to pool

			// Calculate score for this class
			scores[class] = mctm.machines[class].calculateScore(input, 1)
		}()
	}
	wg.Wait()

	// Find the class with the highest score
	maxScore := scores[0]
	predictedClass := 0
	secondHighestScore := float64(-1)

	for class := 1; class < mctm.numClasses; class++ {
		if scores[class] > maxScore {
			secondHighestScore = maxScore
			maxScore = scores[class]
			predictedClass = class
		} else if scores[class] > secondHighestScore {
			secondHighestScore = scores[class]
		}
	}

	// Calculate margin and confidence using absolute values
	margin := math.Abs(maxScore - secondHighestScore)
	maxPossibleVotes := float64(mctm.numClauses) / float64(mctm.numClasses)
	confidence := math.Abs(margin / maxPossibleVotes)

	if mctm.debug {
		log.Printf("Prediction results:")
		log.Printf("  Predicted class: %d", predictedClass)
		log.Printf("  Margin: %.2f", margin)
		log.Printf("  Confidence: %.2f", confidence)
	}

	return PredictionResult{
		Votes:          scores,
		PredictedClass: predictedClass,
		Margin:         margin,
		Confidence:     confidence,
	}
}

// PredictClass returns only the predicted class without calculating full prediction results
func (mctm *MultiClassTsetlinMachine) PredictClass(input []float64) int {
	// Get scores from each binary classifier in parallel using worker pool
	scores := make([]float64, mctm.numClasses)
	var wg sync.WaitGroup
	wg.Add(mctm.numClasses)

	// Create a pool of workers for processing classes
	workerChan := make(chan int, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		workerChan <- i
	}

	// Process each class in parallel
	for class := 0; class < mctm.numClasses; class++ {
		class := class // Create new variable for goroutine
		<-workerChan   // Get worker from pool
		go func() {
			defer wg.Done()
			defer func() { workerChan <- 0 }() // Return worker to pool

			// Calculate score for this class
			scores[class] = mctm.machines[class].calculateScore(input, 1)
		}()
	}
	wg.Wait()

	// Find the class with the highest score using SIMD-like operations
	maxScore := scores[0]
	predictedClass := 0

	// Process scores in pairs for better cache utilization
	for class := 1; class < mctm.numClasses; class += 2 {
		if class+1 < mctm.numClasses {
			// Compare two classes at once
			if scores[class] > maxScore {
				maxScore = scores[class]
				predictedClass = class
			}
			if scores[class+1] > maxScore {
				maxScore = scores[class+1]
				predictedClass = class + 1
			}
		} else {
			// Handle last class if odd number of classes
			if scores[class] > maxScore {
				maxScore = scores[class]
				predictedClass = class
			}
		}
	}

	return predictedClass
}

// PredictProba returns the probability scores for each class
func (mctm *MultiClassTsetlinMachine) PredictProba(input []float64) []float64 {
	// Get scores from each binary classifier in parallel using global worker pool
	scores := make([]float64, mctm.numClasses)
	var mu sync.Mutex

	// Process each class in parallel
	for class := 0; class < mctm.numClasses; class++ {
		class := class // Create new variable for goroutine
		globalWorkerPool.Submit(func() {
			// Calculate score for this class
			score := mctm.machines[class].calculateScore(input, 1)

			mu.Lock()
			scores[class] = score
			if mctm.debug {
				log.Printf("Class %d score: %.2f", class, score)
			}
			mu.Unlock()
		})
	}
	globalWorkerPool.Wait()

	// Normalize scores to get probabilities
	sum := 0.0
	for _, score := range scores {
		sum += score
	}

	if sum > 0 {
		for i := range scores {
			scores[i] /= sum
		}
	} else {
		// If all scores are zero, return uniform probabilities
		for i := range scores {
			scores[i] = 1.0 / float64(mctm.numClasses)
		}
	}

	if mctm.debug {
		log.Printf("Probability scores:")
		for class, prob := range scores {
			log.Printf("  Class %d: %.4f", class, prob)
		}
	}

	return scores
}

// GetClauseInfo returns information about all clauses for all classes
func (mctm *MultiClassTsetlinMachine) GetClauseInfo() [][]ClauseInfo {
	allClauses := make([][]ClauseInfo, mctm.numClasses)
	for class := 0; class < mctm.numClasses; class++ {
		// Get clauses for both classes in the binary classifier
		posClauses := mctm.machines[class].GetClauseInfo(1) // Positive class
		negClauses := mctm.machines[class].GetClauseInfo(0) // Negative class

		// Combine both sets of clauses
		allClauses[class] = append(posClauses, negClauses...)
	}
	return allClauses
}

// GetActiveClauses returns information about currently active clauses for a given input
func (mctm *MultiClassTsetlinMachine) GetActiveClauses(input []float64) [][]ClauseInfo {
	activeClauses := make([][]ClauseInfo, mctm.numClasses)
	for class := 0; class < mctm.numClasses; class++ {
		// Get active clauses for both classes in the binary classifier
		posClauses := mctm.machines[class].GetActiveClauses(input, 1) // Positive class
		negClauses := mctm.machines[class].GetActiveClauses(input, 0) // Negative class

		// Combine both sets of active clauses
		activeClauses[class] = append(posClauses, negClauses...)
	}
	return activeClauses
}
