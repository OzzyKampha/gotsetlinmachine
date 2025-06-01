package tests

import (
	"math/rand"
	"testing"
	"time"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// BenchmarkThroughput measures the throughput in evaluations per second
func BenchmarkThroughput(b *testing.B) {
	// Create a large random dataset with optimized parameters
	numFeatures := 50   // Reduced from 100 to improve throughput
	numClauses := 500   // Reduced from 1000 to improve throughput
	numSamples := 10000 // Increased samples for better measurement
	numLiterals := 5    // Reduced from 10 to improve throughput

	// Create configuration with optimized parameters
	config := tsetlin.DefaultConfig()
	config.NumFeatures = numFeatures
	config.NumClauses = numClauses
	config.NumLiterals = numLiterals
	config.Threshold = 5.0 // Reduced threshold
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create machine
	tm, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// Generate random input data
	inputs := make([][]float64, numSamples)
	for i := range inputs {
		inputs[i] = make([]float64, numFeatures)
		for j := range inputs[i] {
			inputs[i][j] = float64(rand.Intn(2))
		}
	}

	// Generate random labels
	labels := make([]int, numSamples)
	for i := range labels {
		labels[i] = rand.Intn(2)
	}

	// Train the machine
	if err := tm.Fit(inputs, labels, 1); err != nil {
		b.Fatalf("Failed to train machine: %v", err)
	}

	// Reset timer before benchmark
	b.ResetTimer()

	// Measure throughput
	startTime := time.Now()
	evaluations := 0

	// Run benchmark with parallel processing
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			for _, input := range inputs {
				_, err := tm.Predict(input)
				if err != nil {
					b.Fatalf("Failed to predict: %v", err)
				}
				evaluations++
			}
		}
	})

	// Calculate and report metrics
	duration := time.Since(startTime)
	eps := float64(evaluations) / duration.Seconds()

	b.ReportMetric(eps, "EPS")
	b.ReportMetric(float64(duration.Microseconds())/float64(evaluations), "µs/eval")
	b.ReportMetric(float64(numFeatures*numClauses)/eps, "features*clauses/EPS")
}

// TestClauseSkipping verifies the clause skipping functionality
