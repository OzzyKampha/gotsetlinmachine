package tsetlin

import (
	"runtime"
	"sync"
)

// BatchProcessor handles parallel batch processing of predictions
type BatchProcessor struct {
	numWorkers int
	jobChan    chan int
	wg         sync.WaitGroup
}

// NewBatchProcessor creates a new batch processor with the specified number of workers
func NewBatchProcessor() *BatchProcessor {
	return &BatchProcessor{
		numWorkers: runtime.NumCPU(),
		jobChan:    make(chan int, 1000),
	}
}

// ProcessBatch processes a batch of inputs in parallel using the provided prediction function
func (bp *BatchProcessor) ProcessBatch(inputs [][]int, predictFn func([]int) int) []int {
	numSamples := len(inputs)
	results := make([]int, numSamples)

	// Create a wait group for this batch
	var batchWg sync.WaitGroup
	batchWg.Add(numSamples)

	// Process each input
	for i := range inputs {
		go func(idx int) {
			defer batchWg.Done()
			results[idx] = predictFn(inputs[idx])
		}(i)
	}

	// Wait for all predictions to complete
	batchWg.Wait()
	return results
}

// Close stops all workers and cleans up resources
func (bp *BatchProcessor) Close() {
	if bp.jobChan != nil {
		close(bp.jobChan)
		bp.wg.Wait()
		bp.jobChan = nil
	}
}
