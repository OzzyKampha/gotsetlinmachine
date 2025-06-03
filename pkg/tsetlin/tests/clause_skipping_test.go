package tests

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// TestClauseSkipping verifies the clause skipping functionality
func TestClauseSkipping(t *testing.T) {
	// Create configuration
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 10
	config.NumClauses = 5
	config.NumLiterals = 3
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create machine
	tm := tsetlin.NewBitPackedTsetlinMachine(config)

	// Print initial clause information for debugging
	t.Log("Initial clause information:")
	for i, clause := range tm.Clauses {
		t.Logf("Clause %d: include_mask=%v, exclude_mask=%v",
			i, clause.IncludeMask, clause.ExcludeMask)
	}

	// Create test cases
	testCases := []struct {
		name          string
		input         []float64
		expectedScore int
		description   string
	}{
		{
			name:          "All features zero",
			input:         []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			expectedScore: 0,
			description:   "Input with all zeros should have zero score",
		},
		{
			name:          "Single feature one",
			input:         []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
			expectedScore: 0,
			description:   "Input with single one should have appropriate score",
		},
		{
			name:          "All features one",
			input:         []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			expectedScore: 0,
			description:   "Input with all ones should have appropriate score",
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Convert input to bit pattern
			bitInput := tsetlin.FromFloat64Slice(tc.input)

			// Calculate score
			score := 0
			for _, clause := range tm.Clauses {
				if clause.Match(bitInput) {
					score++
				}
			}

			// Verify score
			if score != tc.expectedScore {
				t.Errorf("Expected score %d, got %d", tc.expectedScore, score)
			}

			// Print detailed information for debugging
			t.Logf("Test case: %s", tc.description)
			t.Logf("Input features: %v", tc.input)
			t.Logf("Final score: %d", score)

			// Print clause evaluation details
			for i, clause := range tm.Clauses {
				matches := clause.Match(bitInput)
				t.Logf("Clause %d: matches=%v, include_mask=%v, exclude_mask=%v",
					i, matches, clause.IncludeMask, clause.ExcludeMask)
			}
		})
	}
}

// BenchmarkClauseSkipping measures the performance impact of clause matching
func BenchmarkClauseSkipping(b *testing.B) {
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 500
	config.NumClauses = 1000
	config.NumLiterals = 10
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	tm := tsetlin.NewBitPackedTsetlinMachine(config)

	input := make([]float64, config.NumFeatures)
	for i := 0; i < len(input); i++ {
		if i%100 == 0 {
			input[i] = 1
		}
	}
	bitInput := tsetlin.FromFloat64Slice(input)

	numCPU := runtime.NumCPU()
	var processed int64
	var wg sync.WaitGroup
	b.ResetTimer()
	start := time.Now()
	eventChan := make(chan struct{}, numCPU*2)

	for i := 0; i < numCPU; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range eventChan {
				score := 0
				for _, clause := range tm.Clauses {
					if clause.Match(bitInput) {
						score++
					}
				}
				_ = score
				atomic.AddInt64(&processed, 1)
			}
		}()
	}

	for i := 0; i < b.N; i++ {
		eventChan <- struct{}{}
	}
	close(eventChan)
	wg.Wait()
	dur := time.Since(start)
	eps := float64(processed) / dur.Seconds()
	b.ReportMetric(eps, "EPS")
	b.ReportMetric(float64(dur.Microseconds())/float64(processed), "µs/eval")
}

// BenchmarkLargeTsetlinMachine measures performance with a very large TM
func BenchmarkLargeTsetlinMachine(b *testing.B) {
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 500
	config.NumClauses = 1000
	config.NumLiterals = 20
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	tm := tsetlin.NewBitPackedTsetlinMachine(config)

	inputs := []struct {
		name     string
		density  float64
		features []float64
	}{
		{
			name:    "Very Sparse",
			density: 0.001,
			features: func() []float64 {
				input := make([]float64, config.NumFeatures)
				for i := 0; i < len(input); i++ {
					if i%1000 == 0 {
						input[i] = 1
					}
				}
				return input
			}(),
		},
		{
			name:    "Sparse",
			density: 0.01,
			features: func() []float64 {
				input := make([]float64, config.NumFeatures)
				for i := 0; i < len(input); i++ {
					if i%100 == 0 {
						input[i] = 1
					}
				}
				return input
			}(),
		},
		{
			name:    "Dense",
			density: 0.5,
			features: func() []float64 {
				input := make([]float64, config.NumFeatures)
				for i := 0; i < len(input); i++ {
					if i%2 == 0 {
						input[i] = 1
					}
				}
				return input
			}(),
		},
	}

	for _, input := range inputs {
		b.Run(input.name, func(b *testing.B) {
			bitInput := tsetlin.FromFloat64Slice(input.features)
			numCPU := runtime.NumCPU()
			var processed int64
			var wg sync.WaitGroup
			b.ResetTimer()
			start := time.Now()
			eventChan := make(chan struct{}, numCPU*2)

			for i := 0; i < numCPU; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for range eventChan {
						score := 0
						for _, clause := range tm.Clauses {
							if clause.Match(bitInput) {
								score++
							}
						}
						_ = score
						atomic.AddInt64(&processed, 1)
					}
				}()
			}

			for i := 0; i < b.N; i++ {
				eventChan <- struct{}{}
			}
			close(eventChan)
			wg.Wait()
			dur := time.Since(start)
			eps := float64(processed) / dur.Seconds()
			b.ReportMetric(eps, "EPS")
			b.ReportMetric(float64(dur.Microseconds())/float64(processed), "µs/eval")
		})
	}
}
