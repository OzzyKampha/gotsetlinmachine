package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ozzykampha/multiclass_tsetlinmachine/internal/model"
)

// generateTestData creates synthetic test data with 1000 features
func generateTestData(numExamples int) ([][]float64, []int) {
	X := make([][]float64, numExamples)
	y := make([]int, numExamples)

	for i := 0; i < numExamples; i++ {
		// Generate 1000 features
		features := make([]float64, 1000)
		for j := 0; j < 1000; j++ {
			features[j] = rand.Float64()
		}
		X[i] = features

		// Assign class (0-9) based on feature patterns
		y[i] = i % 10
	}

	return X, y
}

// benchmarkInference measures the inference performance
func benchmarkInference(mctm *model.MultiClassTsetlinMachine, numEvents, batchSize int) {
	fmt.Printf("\nBenchmarking inference with %d events (batch size: %d)...\n", numEvents, batchSize)

	// Generate test data
	testData, testLabels := generateTestData(numEvents)

	// Pre-allocate result slice
	predictions := make([]int, numEvents)

	// Warm up
	fmt.Println("Warming up...")
	for i := 0; i < min(100, numEvents); i++ {
		mctm.PredictClass(testData[i])
	}

	// Benchmark
	fmt.Println("Starting benchmark...")
	startTime := time.Now()

	var processedEvents int64
	batches := (numEvents + batchSize - 1) / batchSize

	// Process batches in parallel
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, runtime.NumCPU())

	// Pre-allocate batch results
	batchResults := make([][]int, runtime.NumCPU())
	for i := range batchResults {
		batchResults[i] = make([]int, batchSize)
	}

	for b := 0; b < batches; b++ {
		start := b * batchSize
		end := min((b+1)*batchSize, numEvents)

		wg.Add(1)
		semaphore <- struct{}{} // Acquire semaphore

		go func(batchStart, batchEnd int) {
			defer wg.Done()
			defer func() { <-semaphore }() // Release semaphore

			// Get worker ID
			workerID := runtime.NumGoroutine() % runtime.NumCPU()
			batchResult := batchResults[workerID]

			// Process batch
			for i := batchStart; i < batchEnd; i++ {
				batchResult[i-batchStart] = mctm.PredictClass(testData[i])
				atomic.AddInt64(&processedEvents, 1)
			}

			// Copy results back to main predictions slice
			copy(predictions[batchStart:batchEnd], batchResult[:batchEnd-batchStart])
		}(start, end)

		// Progress update
		if (b+1)%10 == 0 {
			elapsed := time.Since(startTime)
			eps := float64(processedEvents) / elapsed.Seconds()
			fmt.Printf("Processed %d/%d events (%.1f EPS)\n",
				processedEvents, numEvents, eps)
		}
	}

	wg.Wait()
	totalTime := time.Since(startTime)
	eps := float64(processedEvents) / totalTime.Seconds()
	avgLatency := totalTime / time.Duration(processedEvents)

	// Calculate accuracy
	correct := 0
	for i := 0; i < numEvents; i++ {
		if predictions[i] == testLabels[i] {
			correct++
		}
	}
	accuracy := float64(correct) / float64(numEvents) * 100

	fmt.Printf("\nBenchmark Results:\n")
	fmt.Printf("Total events processed: %d\n", processedEvents)
	fmt.Printf("Total time: %v\n", totalTime)
	fmt.Printf("Events per second: %.2f\n", eps)
	fmt.Printf("Average latency: %v\n", avgLatency)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy)
}

func main() {
	// Command line flags
	numWorkers := flag.Int("workers", runtime.NumCPU(), "Number of worker goroutines")
	numEvents := flag.Int("events", 1000000, "Number of events for benchmarking")
	batchSize := flag.Int("batch", 10000, "Batch size for processing") // Increased batch size
	debug := flag.Bool("debug", false, "Enable debug logging")
	flag.Parse()

	// Initialize global worker pool
	fmt.Printf("Initializing global worker pool with %d workers...\n", *numWorkers)
	model.InitGlobalPool(*numWorkers)
	defer model.CloseGlobalPool()

	// Create multiclass Tsetlin Machine with 1000 features
	fmt.Println("Creating multiclass Tsetlin Machine...")
	mctm := model.NewMultiClassTsetlinMachine(
		10,   // numClasses
		1000, // numFeatures
		100,  // numClauses
		32,   // numLiterals
		15.0, // threshold
		3.9,  // s
		100,  // nStates
	)
	mctm.SetDebug(*debug)

	// Generate training data
	fmt.Println("Generating training data...")
	X, y := generateTestData(1000) // 1000 examples

	// Train the model
	fmt.Println("Training model...")
	startTime := time.Now()
	mctm.Fit(X, y, 5)
	trainingTime := time.Since(startTime)
	fmt.Printf("Training completed in %v\n", trainingTime)

	// Generate test examples
	fmt.Println("\nGenerating test examples...")
	testExamples, testLabels := generateTestData(1000) // 1000 test examples

	// Measure inference time for all test examples
	fmt.Println("Running inference on test examples...")
	startTime = time.Now()

	correct := 0
	for i, example := range testExamples {
		prediction := mctm.PredictClass(example)
		if prediction == testLabels[i] {
			correct++
		}
	}

	inferenceTime := time.Since(startTime)
	accuracy := float64(correct) / float64(len(testExamples)) * 100

	fmt.Printf("\nTest Results:\n")
	fmt.Printf("Accuracy: %.2f%%\n", accuracy)
	fmt.Printf("Total inference time for %d examples: %v\n", len(testExamples), inferenceTime)
	fmt.Printf("Average inference time per example: %v\n", inferenceTime/time.Duration(len(testExamples)))

	// Run benchmark
	benchmarkInference(mctm, *numEvents, *batchSize)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
