package tests

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// generateRandomData creates random training data for benchmarking
func generateRandomData(numSamples, numFeatures int) ([][]float64, []int) {
	X := make([][]float64, numSamples)
	y := make([]int, numSamples)
	r := rand.New(rand.NewSource(42))

	for i := range X {
		X[i] = make([]float64, numFeatures)
		for j := range X[i] {
			X[i][j] = float64(r.Intn(2))
		}
		y[i] = r.Intn(2) // Binary classification
	}
	return X, y
}

// generateMulticlassData creates random multiclass training data
func generateMulticlassData(numSamples, numFeatures, numClasses int) ([][]float64, []int) {
	X := make([][]float64, numSamples)
	y := make([]int, numSamples)
	r := rand.New(rand.NewSource(42))

	for i := range X {
		X[i] = make([]float64, numFeatures)
		for j := range X[i] {
			X[i][j] = float64(r.Intn(2))
		}
		y[i] = r.Intn(numClasses)
	}
	return X, y
}

func BenchmarkTraining(b *testing.B) {
	// Test configurations
	configs := []struct {
		name        string
		numFeatures int
		numClauses  int
		numLiterals int
		numSamples  int
	}{
		{"Small", 10, 20, 4, 1000},
		{"Medium", 50, 100, 8, 5000},
		{"Large", 100, 200, 16, 10000},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			// Create configuration
			config := tsetlin.DefaultConfig()
			config.NumFeatures = cfg.numFeatures
			config.NumClauses = cfg.numClauses
			config.NumLiterals = cfg.numLiterals
			config.Threshold = float64(cfg.numClauses) / 2
			config.S = 3.9
			config.NStates = 100
			config.RandomSeed = 42

			// Generate random data
			X, y := generateRandomData(cfg.numSamples, cfg.numFeatures)

			// Create machine
			machine, err := tsetlin.NewTsetlinMachine(config)
			if err != nil {
				b.Fatalf("Failed to create machine: %v", err)
			}

			// Reset timer and run benchmark
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := machine.Fit(X, y, 1); err != nil {
					b.Fatalf("Training failed: %v", err)
				}
			}

			// Calculate and report metrics
			eps := float64(cfg.numSamples) / (float64(b.Elapsed().Nanoseconds()) / 1e9)
			b.ReportMetric(eps, "EPS")
		})
	}
}

func BenchmarkMulticlassTraining(b *testing.B) {
	// Test configurations
	configs := []struct {
		name        string
		numFeatures int
		numClasses  int
		numClauses  int
		numLiterals int
		numSamples  int
	}{
		{"Small", 10, 3, 20, 4, 1000},
		{"Medium", 50, 5, 100, 8, 5000},
		{"Large", 100, 10, 200, 16, 10000},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			// Create configuration
			config := tsetlin.DefaultConfig()
			config.NumFeatures = cfg.numFeatures
			config.NumClasses = cfg.numClasses
			config.NumClauses = cfg.numClauses
			config.NumLiterals = cfg.numLiterals
			config.Threshold = float64(cfg.numClauses) / 2
			config.S = 3.9
			config.NStates = 100
			config.RandomSeed = 42

			// Generate random data
			X, y := generateMulticlassData(cfg.numSamples, cfg.numFeatures, cfg.numClasses)

			// Create machine
			machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
			if err != nil {
				b.Fatalf("Failed to create machine: %v", err)
			}

			// Reset timer and run benchmark
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := machine.Fit(X, y, 1); err != nil {
					b.Fatalf("Training failed: %v", err)
				}
			}

			// Calculate and report metrics
			eps := float64(cfg.numSamples) / (float64(b.Elapsed().Nanoseconds()) / 1e9)
			b.ReportMetric(eps, "EPS")
		})
	}
}

func BenchmarkTrainingEpochs(b *testing.B) {
	// Fixed configuration
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 50
	config.NumClauses = 100
	config.NumLiterals = 8
	config.Threshold = 50
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Generate random data
	X, y := generateRandomData(1000, config.NumFeatures)

	// Create machine
	machine, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create machine: %v", err)
	}

	// Test different numbers of epochs
	epochs := []int{1, 5, 10, 50, 100}

	for _, numEpochs := range epochs {
		b.Run(fmt.Sprintf("Epochs_%d", numEpochs), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := machine.Fit(X, y, numEpochs); err != nil {
					b.Fatalf("Training failed: %v", err)
				}
			}

			// Calculate and report metrics
			totalSamples := float64(len(X) * numEpochs)
			eps := totalSamples / (float64(b.Elapsed().Nanoseconds()) / 1e9)
			b.ReportMetric(eps, "EPS")
		})
	}
}

func BenchmarkTrainingConvergence(b *testing.B) {
	// Fixed configuration
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 50
	config.NumClauses = 100
	config.NumLiterals = 8
	config.Threshold = 50
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Generate random data
	X, y := generateRandomData(1000, config.NumFeatures)

	// Create machine
	machine, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create machine: %v", err)
	}

	// Measure convergence time
	b.ResetTimer()
	startTime := time.Now()

	// Train until convergence or max epochs
	maxEpochs := 1000
	converged := false
	lastAccuracy := 0.0

	for epoch := 0; epoch < maxEpochs && !converged; epoch++ {
		if err := machine.Fit(X, y, 1); err != nil {
			b.Fatalf("Training failed: %v", err)
		}

		// Calculate accuracy
		correct := 0
		for i, input := range X {
			pred, err := machine.PredictClass(input)
			if err != nil {
				b.Fatalf("Prediction failed: %v", err)
			}
			if pred == y[i] {
				correct++
			}
		}
		accuracy := float64(correct) / float64(len(X))

		// Check for convergence
		if epoch > 0 && math.Abs(accuracy-lastAccuracy) < 0.001 {
			converged = true
		}
		lastAccuracy = accuracy
	}

	duration := time.Since(startTime)
	eps := float64(len(X)) / (float64(duration.Nanoseconds()) / 1e9)
	b.ReportMetric(eps, "EPS")
	b.ReportMetric(float64(duration.Milliseconds()), "ms/conv")
}
