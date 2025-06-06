package tests

import (
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func BenchmarkMultiClassTMPredict(b *testing.B) {
	// Create a multiclass TM with 10 classes, 100 clauses per class, 100 features
	tm := tsetlin.NewMultiClassTM(10, 100, 100, 50, 3)

	// Create a sample input
	input := make([]int, 100)
	for i := range input {
		input[i] = i % 2 // Alternating 0s and 1s
	}

	// Warm up
	for i := 0; i < 100; i++ {
		tm.Predict(input)
	}

	// Reset timer and run benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tm.Predict(input)
	}
}

func TestMultiClassTMPredictPerformance(t *testing.T) {
	// Create a multiclass TM with 10 classes, 100 clauses per class, 100 features
	tm := tsetlin.NewMultiClassTM(10, 100, 100, 50, 3)

	// Create a sample input
	input := make([]int, 100)
	for i := range input {
		input[i] = i % 2 // Alternating 0s and 1s
	}

	// Warm up
	for i := 0; i < 100; i++ {
		tm.Predict(input)
	}

	// Measure parallel implementation
	start := time.Now()
	for i := 0; i < 1000; i++ {
		tm.Predict(input)
	}
	parallelDuration := time.Since(start)

	t.Logf("Parallel implementation took %v for 1000 predictions", parallelDuration)
	t.Logf("Average time per prediction: %v", parallelDuration/1000)
}

func TestMultiClassTMPredictionsPerSecond(t *testing.T) {
	// Create a multiclass TM with 10 classes, 100 clauses per class, 100 features
	tm := tsetlin.NewMultiClassTM(10, 100, 100, 50, 3)

	// Create a sample input
	input := make([]int, 100)
	for i := range input {
		input[i] = i % 2 // Alternating 0s and 1s
	}

	// Warm up
	for i := 0; i < 100; i++ {
		tm.Predict(input)
	}

	// Measure performance over 1 second
	iterations := 0
	start := time.Now()
	endTime := start.Add(time.Second)

	for time.Now().Before(endTime) {
		tm.Predict(input)
		iterations++
	}
	duration := time.Since(start)

	// Calculate metrics
	predictionsPerSecond := float64(iterations) / duration.Seconds()
	avgTimePerPrediction := duration / time.Duration(iterations)

	t.Logf("Performance metrics:")
	t.Logf("Total predictions in 1 second: %d", iterations)
	t.Logf("Predictions per second: %.2f", predictionsPerSecond)
	t.Logf("Average time per prediction: %v", avgTimePerPrediction)
	t.Logf("Total duration: %v", duration)
}

func TestMultiClassTMInference1M(t *testing.T) {
	// Create a multiclass TM with 10 classes, 100 clauses per class, 100 features
	tm := tsetlin.NewMultiClassTM(10, 100, 100, 50, 3)

	// Create a batch of 1M inputs
	const numSamples = 1_000_000
	inputs := make([][]int, numSamples)
	for i := range inputs {
		inputs[i] = make([]int, 100)
		for j := range inputs[i] {
			inputs[i][j] = j % 2 // Alternating 0s and 1s
		}
	}

	// Warm up
	for i := 0; i < 100; i++ {
		tm.Predict(inputs[0])
	}

	// Method 1: Batch Processing with Worker Pool
	t.Log("Starting batch processing with worker pool...")
	start := time.Now()

	numWorkers := runtime.NumCPU()
	results := make([]int, numSamples)
	jobs := make(chan int, numSamples)
	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = tm.Predict(inputs[i])
			}
		}()
	}

	// Send jobs
	for i := 0; i < numSamples; i++ {
		jobs <- i
	}
	close(jobs)

	// Wait for completion
	wg.Wait()
	batchDuration := time.Since(start)

	// Method 2: Streaming Processing
	t.Log("Starting streaming processing...")
	start = time.Now()

	streamResults := make([]int, numSamples)
	for i := 0; i < numSamples; i++ {
		streamResults[i] = tm.Predict(inputs[i])
	}
	streamDuration := time.Since(start)

	// Report results
	t.Logf("\nPerformance Results for 1M Samples:")
	t.Logf("Batch Processing (Worker Pool):")
	t.Logf("  Total time: %v", batchDuration)
	t.Logf("  Samples per second: %.2f", float64(numSamples)/batchDuration.Seconds())
	t.Logf("  Time per sample: %v", batchDuration/time.Duration(numSamples))
	t.Logf("\nStreaming Processing:")
	t.Logf("  Total time: %v", streamDuration)
	t.Logf("  Samples per second: %.2f", float64(numSamples)/streamDuration.Seconds())
	t.Logf("  Time per sample: %v", streamDuration/time.Duration(numSamples))
	t.Logf("\nSpeedup: %.2fx", float64(streamDuration)/float64(batchDuration))
}

func TestMultiClassTMBatchProcessing(t *testing.T) {
	// Create a multiclass TM with 10 classes, 100 clauses per class, 100 features
	tm := tsetlin.NewMultiClassTM(10, 100, 100, 50, 3)

	// Create a batch processor
	bp := tsetlin.NewBatchProcessor()
	defer bp.Close()

	// Create a batch of 1000 inputs
	const numSamples = 1000
	inputs := make([][]int, numSamples)
	for i := range inputs {
		inputs[i] = make([]int, 100)
		for j := range inputs[i] {
			inputs[i][j] = j % 2 // Alternating 0s and 1s
		}
	}

	// Warm up
	for i := 0; i < 100; i++ {
		tm.Predict(inputs[0])
	}

	// Method 1: Using batch processor
	t.Log("Starting batch processing...")
	start := time.Now()
	results := bp.ProcessBatch(inputs, tm.Predict)
	_ = results // Use results to avoid unused variable error
	batchDuration := time.Since(start)

	// Method 2: Sequential processing
	t.Log("Starting sequential processing...")
	start = time.Now()

	seqResults := make([]int, numSamples)
	for i := 0; i < numSamples; i++ {
		seqResults[i] = tm.Predict(inputs[i])
	}
	seqDuration := time.Since(start)

	// Report results
	t.Logf("\nPerformance Results for %d Samples:", numSamples)
	t.Logf("Batch Processing:")
	t.Logf("  Total time: %v", batchDuration)
	t.Logf("  Samples per second: %.2f", float64(numSamples)/batchDuration.Seconds())
	t.Logf("  Time per sample: %v", batchDuration/time.Duration(numSamples))
	t.Logf("\nSequential Processing:")
	t.Logf("  Total time: %v", seqDuration)
	t.Logf("  Samples per second: %.2f", float64(numSamples)/seqDuration.Seconds())
	t.Logf("  Time per sample: %v", seqDuration/time.Duration(numSamples))
	t.Logf("\nSpeedup: %.2fx", float64(seqDuration)/float64(batchDuration))
}
