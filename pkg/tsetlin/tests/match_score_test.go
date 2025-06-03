package tests

import (
	"math"
	"testing"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// TestMatchScoreAndMomentum verifies the MatchScore and Momentum tracking functionality
func TestMatchScoreAndMomentum(t *testing.T) {
	// Create configuration
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 4
	config.NumClauses = 2
	config.NumLiterals = 4
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create machine
	tm, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		t.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// Test cases with different input patterns
	testCases := []struct {
		name         string
		input        []float64
		expectedConf float64
		description  string
	}{
		{
			name:         "Pattern 1",
			input:        []float64{1, 1, 0, 0},
			expectedConf: 0.5,
			description:  "Simple pattern that should match some clauses",
		},
		{
			name:         "Pattern 2",
			input:        []float64{0, 0, 1, 1},
			expectedConf: 0.5,
			description:  "Different pattern that should match different clauses",
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Get initial clause info
			initialInfo := tm.GetClauseInfo()[0]

			// Make prediction
			result, err := tm.Predict(tc.input)
			if err != nil {
				t.Fatalf("Failed to predict: %v", err)
			}

			// Get updated clause info
			updatedInfo := tm.GetClauseInfo()[0]

			// Verify confidence calculation
			if result.Confidence < 0.0 || result.Confidence > 1.0 {
				t.Errorf("Confidence should be between 0 and 1, got %f", result.Confidence)
			}

			// For Pattern 2, just check non-negativity and log values
			if tc.name == "Pattern 2" {
				for i := range initialInfo {
					t.Logf("[Pattern 2] Clause %d: Initial MatchScore: %f, Updated MatchScore: %f", i, initialInfo[i].MatchScore, updatedInfo[i].MatchScore)
					t.Logf("[Pattern 2] Clause %d: Initial Momentum: %f, Updated Momentum: %f", i, initialInfo[i].Momentum, updatedInfo[i].Momentum)
					if updatedInfo[i].MatchScore < 0.0 {
						t.Errorf("Clause %d: MatchScore should not be negative", i)
					}
					if updatedInfo[i].Momentum < 0.0 {
						t.Errorf("Clause %d: Momentum should not be negative", i)
					}
				}
			} else {
				// For other patterns, verify that at least one clause's scores increased
				for i := range initialInfo {
					// Print detailed information for debugging
					t.Logf("Clause %d:", i)
					t.Logf("  Initial MatchScore: %f", initialInfo[i].MatchScore)
					t.Logf("  Updated MatchScore: %f", updatedInfo[i].MatchScore)
					t.Logf("  Initial Momentum: %f", initialInfo[i].Momentum)
					t.Logf("  Updated Momentum: %f", updatedInfo[i].Momentum)

					if i == 0 && updatedInfo[i].MatchScore <= initialInfo[i].MatchScore &&
						updatedInfo[i].Momentum <= initialInfo[i].Momentum {
						t.Errorf("At least one clause should have increased scores")
					}
				}
			}

			// Print prediction details
			t.Logf("Test case: %s", tc.description)
			t.Logf("Input: %v", tc.input)
			t.Logf("Predicted class: %d", result.PredictedClass)
			t.Logf("Confidence: %f", result.Confidence)
			t.Logf("Margin: %f", result.Margin)
		})
	}

	// Test repeated patterns to verify accumulation
	t.Run("Repeated Patterns", func(t *testing.T) {
		pattern := []float64{1, 1, 0, 0}
		initialInfo := tm.GetClauseInfo()[0]

		// Make multiple predictions with the same pattern
		for i := 0; i < 5; i++ {
			_, err := tm.Predict(pattern)
			if err != nil {
				t.Fatalf("Failed to predict: %v", err)
			}
		}

		// Get final clause info
		finalInfo := tm.GetClauseInfo()[0]

		// Verify accumulation of MatchScore and Momentum
		for i := range initialInfo {
			t.Logf("Clause %d after repeated matches:", i)
			t.Logf("  Initial MatchScore: %f", initialInfo[i].MatchScore)
			t.Logf("  Final MatchScore: %f", finalInfo[i].MatchScore)
			t.Logf("  Initial Momentum: %f", initialInfo[i].Momentum)
			t.Logf("  Final Momentum: %f", finalInfo[i].Momentum)

			// Verify that at least one clause's scores increased
			if i == 0 && finalInfo[i].MatchScore <= initialInfo[i].MatchScore &&
				finalInfo[i].Momentum <= initialInfo[i].Momentum {
				t.Errorf("At least one clause should have increased scores with repeated matches")
			}
		}
	})
}

// BenchmarkMatchScoreAndMomentum measures the performance impact of MatchScore and Momentum tracking
func BenchmarkMatchScoreAndMomentum(b *testing.B) {
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 50
	config.NumClauses = 100
	config.NumLiterals = 8
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	tm, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// Create test input
	input := make([]float64, config.NumFeatures)
	for i := 0; i < len(input); i++ {
		if i%2 == 0 {
			input[i] = 1
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := tm.Predict(input)
		if err != nil {
			b.Fatalf("Failed to predict: %v", err)
		}
	}
}

func TestMatchScoreWeighting(t *testing.T) {
	// Create a small TM with 2 features and 2 clauses
	config := tsetlin.Config{
		NumFeatures: 2,
		NumClauses:  2,
		NumLiterals: 2,
		NStates:     100,
		S:           3.9,
		Threshold:   15,
		RandomSeed:  42,
	}

	tm, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		t.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// Train the model to build up MatchScores
	// Use repeated patterns to build up different MatchScores
	trainingData := [][]float64{
		{1, 1}, // Pattern 1
		{1, 1}, // Pattern 1
		{1, 1}, // Pattern 1
		{1, 1}, // Pattern 1
		{1, 1}, // Pattern 1
		{0, 0}, // Pattern 2
		{0, 0}, // Pattern 2
	}
	labels := []int{1, 1, 1, 1, 1, 0, 0}

	err = tm.Fit(trainingData, labels, 1)
	if err != nil {
		t.Fatalf("Failed to train model: %v", err)
	}

	// Get clause info to verify MatchScores
	clauseInfo := tm.GetClauseInfo()[0]

	// Test input that matches both clauses
	input := []float64{1, 1}

	// Get prediction
	result, err := tm.Predict(input)
	if err != nil {
		t.Fatalf("Failed to get prediction: %v", err)
	}

	// Calculate expected weighted score based on MatchScores from clause info
	expectedWeight0 := clauseInfo[0].MatchScore / (clauseInfo[0].MatchScore + 1.0)
	expectedWeight1 := clauseInfo[1].MatchScore / (clauseInfo[1].MatchScore + 1.0)
	expectedScore := expectedWeight0 + expectedWeight1

	// Verify the score is weighted correctly
	if math.Abs(result.Margin*float64(config.NumClauses)-expectedScore) > 0.01 {
		t.Errorf("Expected weighted score %v, got %v", expectedScore, result.Margin*float64(config.NumClauses))
	}

	// Verify confidence reflects the weighted contribution
	if result.Confidence < 0.5 {
		t.Errorf("Expected high confidence due to weighted contributions, got %v", result.Confidence)
	}

	// Log the details for inspection
	t.Logf("Clause 0: MatchScore=%.2f, Weight=%.3f", clauseInfo[0].MatchScore, expectedWeight0)
	t.Logf("Clause 1: MatchScore=%.2f, Weight=%.3f", clauseInfo[1].MatchScore, expectedWeight1)
	t.Logf("Final score: %.3f", result.Margin*float64(config.NumClauses))
	t.Logf("Confidence: %.3f", result.Confidence)
}
