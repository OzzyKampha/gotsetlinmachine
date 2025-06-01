package tests

import (
	"testing"

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
	tm, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		t.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// Print initial clause information for debugging
	t.Log("Initial clause information:")
	for i, clause := range tm.Clauses() {
		t.Logf("Clause %d: literals=%v, interested_features=%v",
			i, clause, tm.InterestedFeatures(i))
	}

	// Create test cases
	testCases := []struct {
		name          string
		input         []float64
		expectedSkips int
		expectedScore float64
		description   string
	}{
		{
			name:          "All features zero",
			input:         []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			expectedSkips: 5, // Should skip all clauses since no features match
			expectedScore: 0,
			description:   "Input with all zeros should skip all clauses",
		},
		{
			name:          "Single feature one",
			input:         []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
			expectedSkips: 3, // Should skip clauses that don't use feature 2
			expectedScore: 0,
			description:   "Input with single one should skip clauses not using that feature",
		},
		{
			name:          "All features one",
			input:         []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			expectedSkips: 0,  // Should evaluate all clauses
			expectedScore: -1, // Score depends on clause states
			description:   "Input with all ones should evaluate all clauses",
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Count skipped clauses
			skippedClauses := 0
			inputFeatureSet := make(map[int]struct{})
			for i, val := range tc.input {
				if val == 1 {
					inputFeatureSet[i] = struct{}{}
				}
			}

			// Check each clause
			for i := range tm.Clauses() {
				if tm.CanSkipClause(i, inputFeatureSet) {
					skippedClauses++
				}
			}

			// Verify skipped clauses count
			if skippedClauses != tc.expectedSkips {
				t.Errorf("Expected %d skipped clauses, got %d", tc.expectedSkips, skippedClauses)
			}

			// Verify score calculation
			score := tm.CalculateScore(tc.input, 0)
			if score != tc.expectedScore {
				t.Errorf("Expected score %f, got %f", tc.expectedScore, score)
			}

			// Print detailed information for debugging
			t.Logf("Test case: %s", tc.description)
			t.Logf("Input features: %v", tc.input)
			t.Logf("Skipped clauses: %d/%d", skippedClauses, config.NumClauses)
			t.Logf("Final score: %f", score)

			// Print clause evaluation details
			for i, clause := range tm.Clauses() {
				skipped := tm.CanSkipClause(i, inputFeatureSet)
				output := 0
				if !skipped {
					output = tm.EvaluateClause(tc.input, clause)
				}
				t.Logf("Clause %d: skipped=%v, output=%d, literals=%v",
					i, skipped, output, clause)
			}
		})
	}
}

// BenchmarkClauseSkipping measures the performance impact of clause skipping
func BenchmarkClauseSkipping(b *testing.B) {
	// Create configuration
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 500
	config.NumClauses = 1000
	config.NumLiterals = 10
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create machine
	tm, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// Create sparse input (only 1% of features are 1)
	input := make([]float64, config.NumFeatures)
	for i := 0; i < len(input); i++ {
		if i%100 == 0 { // 1% of features are 1
			input[i] = 1
		}
	}

	// Reset timer before benchmark
	b.ResetTimer()

	// Run benchmark
	for i := 0; i < b.N; i++ {
		score := tm.CalculateScore(input, 0)
		_ = score // Prevent compiler optimization
	}

	// Calculate and report metrics
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "EPS")
	b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N), "µs/eval")
}

// BenchmarkLargeTsetlinMachine measures performance with a very large TM
func BenchmarkLargeTsetlinMachine(b *testing.B) {
	// Create configuration for a very large TM
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 500 // 500 features
	config.NumClauses = 1000 // 1000 clauses
	config.NumLiterals = 20  // 20 literals per clause
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create machine
	tm, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// Create different types of inputs for testing
	inputs := []struct {
		name     string
		density  float64 // percentage of 1s
		features []float64
	}{
		{
			name:    "Very Sparse",
			density: 0.001, // 0.1% ones
			features: func() []float64 {
				input := make([]float64, config.NumFeatures)
				for i := 0; i < len(input); i++ {
					if i%1000 == 0 { // 0.1% of features are 1
						input[i] = 1
					}
				}
				return input
			}(),
		},
		{
			name:    "Sparse",
			density: 0.01, // 1% ones
			features: func() []float64 {
				input := make([]float64, config.NumFeatures)
				for i := 0; i < len(input); i++ {
					if i%100 == 0 { // 1% of features are 1
						input[i] = 1
					}
				}
				return input
			}(),
		},
		{
			name:    "Dense",
			density: 0.5, // 50% ones
			features: func() []float64 {
				input := make([]float64, config.NumFeatures)
				for i := 0; i < len(input); i++ {
					if i%2 == 0 { // 50% of features are 1
						input[i] = 1
					}
				}
				return input
			}(),
		},
	}

	// Run benchmarks for each input type
	for _, input := range inputs {
		b.Run(input.name, func(b *testing.B) {
			// Reset timer before benchmark
			b.ResetTimer()

			// Run benchmark
			for i := 0; i < b.N; i++ {
				score := tm.CalculateScore(input.features, 0)
				_ = score // Prevent compiler optimization
			}

			// Calculate metrics
			eps := float64(b.N) / b.Elapsed().Seconds()
			featuresClausesPerSecond := float64(config.NumFeatures*config.NumClauses) / eps
			activeFeatures := 0
			for _, val := range input.features {
				if val == 1 {
					activeFeatures++
				}
			}
			activeFeaturesPercent := float64(activeFeatures) / float64(config.NumFeatures) * 100
			estimatedClausesEvaluated := float64(config.NumClauses) * (activeFeaturesPercent / 100)
			actualFeaturesClausesPerSecond := float64(activeFeatures*config.NumClauses) / eps

			// Report standard metrics
			b.ReportMetric(eps, "EPS")
			b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N), "µs/eval")
			b.ReportMetric(featuresClausesPerSecond, "features*clauses/EPS")
			b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N)/float64(config.NumClauses), "µs/clause")

			// Report detailed analysis
			b.Logf("\nDetailed Analysis for %s input:", input.name)
			b.Logf("Total Features: %d", config.NumFeatures)
			b.Logf("Total Clauses: %d", config.NumClauses)
			b.Logf("Active Features: %d (%.2f%%)", activeFeatures, activeFeaturesPercent)
			b.Logf("Estimated Clauses Evaluated: %.0f (%.2f%%)", estimatedClausesEvaluated, activeFeaturesPercent)
			b.Logf("Raw Throughput: %.2f features*clauses/EPS", featuresClausesPerSecond)
			b.Logf("Effective Throughput: %.2f active_features*clauses/EPS", actualFeaturesClausesPerSecond)
			b.Logf("Theoretical Maximum: %d features*clauses", config.NumFeatures*config.NumClauses)
			b.Logf("Efficiency: %.2f%%", (actualFeaturesClausesPerSecond/float64(config.NumFeatures*config.NumClauses))*100)
		})
	}
}
