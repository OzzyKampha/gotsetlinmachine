package tsetlin

import (
	"fmt"
	"runtime"
	"sync"
)

// MultiClassTM represents a multiclass Tsetlin Machine
type MultiClassTM struct {
	Classes []*TsetlinMachine
	workers int
	jobChan chan int
	wg      sync.WaitGroup
}

// NewMultiClassTM creates a new multiclass Tsetlin Machine.
// numClasses: number of classes to classify
// numClauses: number of clauses per class
// numFeatures: number of input features
// threshold: vote threshold for each binary classifier
// s: specificity parameter
func NewMultiClassTM(numClasses, numClauses, numFeatures, threshold int, s float64) *MultiClassTM {
	m := &MultiClassTM{
		Classes: make([]*TsetlinMachine, numClasses),
		workers: runtime.NumCPU(),
		jobChan: make(chan int, 1000), // Buffer size of 1000 jobs
	}
	for i := 0; i < numClasses; i++ {
		m.Classes[i] = NewTsetlinMachine(numClauses, numFeatures, threshold, s)
	}
	return m
}

// PredictBatch performs batch prediction using a worker pool.
// It processes multiple inputs in parallel and returns their predictions.
func (m *MultiClassTM) PredictBatch(inputs [][]int) []int {
	numSamples := len(inputs)
	results := make([]int, numSamples)

	// Start workers if not already running
	if m.jobChan == nil {
		m.jobChan = make(chan int, 1000)
		m.startWorkers()
	}

	// Create a wait group for this batch
	var batchWg sync.WaitGroup
	batchWg.Add(numSamples)

	// Process each input
	for i := range inputs {
		go func(idx int) {
			defer batchWg.Done()
			results[idx] = m.Predict(inputs[idx])
		}(i)
	}

	// Wait for all predictions to complete
	batchWg.Wait()
	return results
}

// startWorkers initializes the worker pool
func (m *MultiClassTM) startWorkers() {
	for w := 0; w < m.workers; w++ {
		m.wg.Add(1)
		go func() {
			defer m.wg.Done()
			for range m.jobChan {
				// Workers are now used for batch processing
			}
		}()
	}
}

// Close stops all workers and cleans up resources
func (m *MultiClassTM) Close() {
	if m.jobChan != nil {
		close(m.jobChan)
		m.wg.Wait()
		m.jobChan = nil
	}
}

// Fit trains the multiclass Tsetlin Machine on the provided data.
// Training is performed in parallel for each class.
func (m *MultiClassTM) Fit(X [][]int, Y []int, epochs int) {
	var wg sync.WaitGroup
	for class := 0; class < len(m.Classes); class++ {
		wg.Add(1)
		go func(cls int) {
			defer wg.Done()
			m.Classes[cls].Fit(X, Y, cls, epochs)
		}(class)
	}
	wg.Wait()
}

// Predict makes a multiclass prediction for the input.
// Returns the class with the highest score.
func (m *MultiClassTM) Predict(X []int) int {
	bestClass := 0
	bestScore := m.Classes[0].Score(X)
	for class := 1; class < len(m.Classes); class++ {
		score := m.Classes[class].Score(X)
		if score > bestScore {
			bestScore = score
			bestClass = class
		}
	}
	fmt.Println(bestScore)
	return bestClass
}
