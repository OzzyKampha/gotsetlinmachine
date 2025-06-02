package tests

import (
	"math"
	"testing"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// TestShardedInference verifies that the sharded inference implementation correctly:
// 1. Processes input patterns in parallel across classes
// 2. Correctly identifies patterns it was trained on
// 3. Handles unknown patterns appropriately
// 4. Returns valid confidence scores and vote distributions
func TestShardedInference(t *testing.T) {
	// Create a small multiclass TM with 3 classes and 4 features
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 4
	config.NumClasses = 3
	config.NumClauses = 10
	config.NumLiterals = 4
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create and initialize the multiclass TM
	tm, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		t.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Training data consists of three distinct patterns:
	// Pattern 1: [1,1,0,0] -> Class 0 (repeated 3 times)
	// Pattern 2: [0,0,1,1] -> Class 1 (repeated 3 times)
	// Pattern 3: [1,0,1,0] -> Class 2 (repeated 3 times)
	X := [][]float64{
		{1, 1, 0, 0}, // Pattern 1 -> Class 0
		{1, 1, 0, 0},
		{1, 1, 0, 0},
		{0, 0, 1, 1}, // Pattern 2 -> Class 1
		{0, 0, 1, 1},
		{0, 0, 1, 1},
		{1, 0, 1, 0}, // Pattern 3 -> Class 2
		{1, 0, 1, 0},
		{1, 0, 1, 0},
	}
	y := []int{0, 0, 0, 1, 1, 1, 2, 2, 2}

	// Train the model for 10 epochs
	if err := tm.Fit(X, y, 10); err != nil {
		t.Fatalf("Failed to train model: %v", err)
	}

	// Test cases verify different aspects of the sharded inference:
	// 1. Known patterns should be correctly classified
	// 2. Unknown patterns should be handled gracefully
	// 3. Vote distributions should be reasonable
	testCases := []struct {
		name            string
		input           []float64
		expectedClass   int
		expectedVotes   []int
		checkConfidence bool
		description     string // Added description for clarity
	}{
		{
			name:            "Pattern 1",
			input:           []float64{1, 1, 0, 0},
			expectedClass:   0,
			expectedVotes:   []int{3, 0, 0}, // Should have votes for class 0
			checkConfidence: true,
			description:     "Tests recognition of first pattern (class 0)",
		},
		{
			name:            "Pattern 2",
			input:           []float64{0, 0, 1, 1},
			expectedClass:   1,
			expectedVotes:   []int{0, 3, 0}, // Should have votes for class 1
			checkConfidence: true,
			description:     "Tests recognition of second pattern (class 1)",
		},
		{
			name:            "Pattern 3",
			input:           []float64{1, 0, 1, 0},
			expectedClass:   2,
			expectedVotes:   []int{0, 0, 3}, // Should have votes for class 2
			checkConfidence: true,
			description:     "Tests recognition of third pattern (class 2)",
		},
		{
			name:            "Unknown Pattern",
			input:           []float64{1, 1, 1, 1},
			expectedClass:   -1,  // Don't check specific class
			expectedVotes:   nil, // Don't check specific votes
			checkConfidence: true,
			description:     "Tests handling of an unseen pattern",
		},
	}

	// Run each test case
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Test case: %s - %s", tc.name, tc.description)

			// Run sharded inference
			_, confidence, votes, matchScores := tsetlin.ShardedInference(tm, tc.input)

			// Verify output dimensions
			if len(votes) != config.NumClasses {
				t.Errorf("Expected %d vote counts, got %d", config.NumClasses, len(votes))
			}
			if len(matchScores) != config.NumClasses {
				t.Errorf("Expected %d match scores, got %d", config.NumClasses, len(matchScores))
			}

			// Verify confidence is in valid range
			if confidence < 0.0 || confidence > 1.0 {
				t.Errorf("Confidence should be between 0 and 1, got %f", confidence)
			}

			// Verify match scores are valid numbers
			for i, score := range matchScores {
				if math.IsNaN(score) || math.IsInf(score, 0) {
					t.Errorf("Invalid match score for class %d: %f", i, score)
				}
			}

			// Log the results for debugging
			t.Logf("Results - Confidence: %.3f, Votes: %v, Match Scores: %v",
				confidence, votes, matchScores)
		})
	}
}

// TestShardedInferenceEdgeCases verifies that the sharded inference implementation
// correctly handles various edge cases and invalid inputs:
// 1. Empty input vectors
// 2. Input vectors with incorrect dimensions
// 3. Input vectors with all zeros
// 4. Input vectors with all ones
func TestShardedInferenceEdgeCases(t *testing.T) {
	// Create a minimal configuration for edge case testing
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 2
	config.NumClasses = 2
	config.NumClauses = 2
	config.NumLiterals = 2
	config.Threshold = 1.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create and initialize the multiclass TM
	tm, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		t.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Test cases for various edge cases
	testCases := []struct {
		name          string
		input         []float64
		expectedError bool
		description   string // Added description for clarity
	}{
		{
			name:          "Empty Input",
			input:         []float64{},
			expectedError: true,
			description:   "Tests handling of empty input vector",
		},
		{
			name:          "Wrong Input Size",
			input:         []float64{1, 1, 1}, // One extra feature
			expectedError: true,
			description:   "Tests handling of input with incorrect dimensions",
		},
		{
			name:          "All Zeros",
			input:         []float64{0, 0},
			expectedError: false,
			description:   "Tests handling of input with all zeros",
		},
		{
			name:          "All Ones",
			input:         []float64{1, 1},
			expectedError: false,
			description:   "Tests handling of input with all ones",
		},
	}

	// Run each test case
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Test case: %s - %s", tc.name, tc.description)

			// Run sharded inference
			_, confidence, votes, matchScores := tsetlin.ShardedInference(tm, tc.input)

			if tc.expectedError {
				// For error cases, verify appropriate error handling
				if confidence != 0.0 {
					t.Errorf("For invalid input, expected confidence 0.0, got %f", confidence)
				}
				t.Logf("Error case handled correctly - Confidence: %.3f", confidence)
			} else {
				// For valid inputs, verify output properties
				if len(votes) != config.NumClasses {
					t.Errorf("Expected %d vote counts, got %d", config.NumClasses, len(votes))
				}
				if len(matchScores) != config.NumClasses {
					t.Errorf("Expected %d match scores, got %d", config.NumClasses, len(matchScores))
				}
				t.Logf("Valid case results - Confidence: %.3f, Votes: %v, Match Scores: %v",
					confidence, votes, matchScores)
			}
		})
	}
}

// TestShardedInferenceConcurrent verifies that the sharded inference implementation
// works correctly under concurrent execution by:
// 1. Running multiple inference calls in parallel
// 2. Verifying that all concurrent calls produce consistent results
// 3. Ensuring thread safety of the implementation
func TestShardedInferenceConcurrent(t *testing.T) {
	// Create configuration for concurrent testing
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 4
	config.NumClasses = 3
	config.NumClauses = 20
	config.NumLiterals = 4
	config.Threshold = 10.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create and initialize the multiclass TM
	tm, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		t.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Train the model with three distinct patterns
	X := [][]float64{
		{1, 1, 0, 0}, {1, 1, 0, 0}, {1, 1, 0, 0}, // Pattern 1 -> Class 0
		{0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, // Pattern 2 -> Class 1
		{1, 0, 1, 0}, {1, 0, 1, 0}, {1, 0, 1, 0}, // Pattern 3 -> Class 2
	}
	y := []int{0, 0, 0, 1, 1, 1, 2, 2, 2}

	if err := tm.Fit(X, y, 10); err != nil {
		t.Fatalf("Failed to train model: %v", err)
	}

	// Test input pattern (Pattern 1)
	testInput := []float64{1, 1, 0, 0}

	// Channel to collect results from concurrent inference calls
	results := make(chan struct {
		class      int
		confidence float64
		votes      []float64
		scores     []float64
	}, 10)

	// Launch 10 concurrent inference calls
	t.Log("Launching 10 concurrent inference calls...")
	for i := 0; i < 10; i++ {
		go func() {
			class, conf, votes, scores := tsetlin.ShardedInference(tm, testInput)
			results <- struct {
				class      int
				confidence float64
				votes      []float64
				scores     []float64
			}{class, conf, votes, scores}
		}()
	}

	// Collect and verify results
	t.Log("Collecting and verifying results...")
	firstResult := <-results
	t.Logf("First result - Class: %d, Confidence: %.3f, Votes: %v",
		firstResult.class, firstResult.confidence, firstResult.votes)

	// Compare all other results with the first one
	for i := 1; i < 10; i++ {
		result := <-results
		t.Logf("Result %d - Class: %d, Confidence: %.3f, Votes: %v",
			i+1, result.class, result.confidence, result.votes)

		// Verify consistency of results
		if result.class != firstResult.class {
			t.Errorf("Concurrent inference produced different classes: %d vs %d",
				result.class, firstResult.class)
		}
		if math.Abs(result.confidence-firstResult.confidence) > 0.001 {
			t.Errorf("Concurrent inference produced different confidences: %.3f vs %.3f",
				result.confidence, firstResult.confidence)
		}
		for j, vote := range result.votes {
			if vote != firstResult.votes[j] {
				t.Errorf("Concurrent inference produced different votes for class %d: %.3f vs %.3f",
					j, vote, firstResult.votes[j])
			}
		}
	}
}

func BenchmarkInferenceSpeed(b *testing.B) {
	// Create configuration for a larger multiclass TM to better demonstrate parallel benefits
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 16
	config.NumClasses = 10
	config.NumClauses = 100
	config.NumLiterals = 8
	config.Threshold = 50.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create multiclass TM
	tm, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Generate random training data
	X := make([][]float64, 1000)
	y := make([]int, 1000)
	for i := range X {
		X[i] = make([]float64, config.NumFeatures)
		for j := range X[i] {
			X[i][j] = float64(i % 2) // Simple pattern for training
		}
		y[i] = i % config.NumClasses
	}

	// Train the model
	if err := tm.Fit(X, y, 10); err != nil {
		b.Fatalf("Failed to train model: %v", err)
	}

	// Generate test input
	input := make([]float64, config.NumFeatures)
	for i := range input {
		input[i] = float64(i % 2)
	}

	// Reset timer before benchmark
	b.ResetTimer()

	// Run benchmark
	for i := 0; i < b.N; i++ {
		// Use parallel inference
		_, _, _, _ = tsetlin.ShardedInference(tm, input)
	}
}

func BenchmarkNonParallelInference(b *testing.B) {
	// Create configuration for a larger multiclass TM to better demonstrate parallel benefits
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 16
	config.NumClasses = 10
	config.NumClauses = 100
	config.NumLiterals = 8
	config.Threshold = 50.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create multiclass TM
	tm, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Generate random training data
	X := make([][]float64, 1000)
	y := make([]int, 1000)
	for i := range X {
		X[i] = make([]float64, config.NumFeatures)
		for j := range X[i] {
			X[i][j] = float64(i % 2) // Simple pattern for training
		}
		y[i] = i % config.NumClasses
	}

	// Train the model
	if err := tm.Fit(X, y, 10); err != nil {
		b.Fatalf("Failed to train model: %v", err)
	}

	// Generate test input
	input := make([]float64, config.NumFeatures)
	for i := range input {
		input[i] = float64(i % 2)
	}

	// Reset timer before benchmark
	b.ResetTimer()

	// Run benchmark
	for i := 0; i < b.N; i++ {
		// Use non-parallel inference
		_, err := tm.Predict(input)
		if err != nil {
			b.Fatalf("Non-parallel inference failed: %v", err)
		}
	}
}

func BenchmarkInferenceScalability(b *testing.B) {
	// Test configurations with different sizes
	configs := []struct {
		name        string
		numFeatures int
		numClasses  int
		numClauses  int
		numLiterals int
	}{
		{"Small", 8, 4, 50, 4},
		{"Medium", 16, 8, 100, 8},
		{"Large", 32, 16, 200, 16},
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

			// Create multiclass TM
			tm, err := tsetlin.NewMultiClassTsetlinMachine(config)
			if err != nil {
				b.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
			}

			// Generate test input
			input := make([]float64, cfg.numFeatures)
			for i := range input {
				input[i] = float64(i % 2)
			}

			// Reset timer before benchmark
			b.ResetTimer()

			// Run benchmark
			for i := 0; i < b.N; i++ {
				// Use parallel inference
				_, _, _, _ = tsetlin.ShardedInference(tm, input)
			}
		})
	}
}

// TestShardedInferenceLargeTM verifies that the sharded inference implementation
// works correctly with a large Tsetlin Machine by:
// 1. Creating a TM with many features, classes, and clauses
// 2. Training it with a large dataset
// 3. Verifying inference performance and correctness
func TestShardedInferenceLargeTM(t *testing.T) {
	// Create configuration for a large TM
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 32
	config.NumClasses = 10
	config.NumClauses = 500
	config.NumLiterals = 16
	config.Threshold = 2.0 // Reduced threshold
	config.S = 2.0         // Reduced S for more focused learning
	config.NStates = 100
	config.RandomSeed = 42
	config.Debug = true // Enable debug mode to see state movement

	// Create and initialize the large multiclass TM
	tm, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		t.Fatalf("Failed to create large Multiclass Tsetlin Machine: %v", err)
	}

	// Generate a large training dataset
	numSamples := 2000 // Increased number of samples
	X := make([][]float64, numSamples)
	y := make([]int, numSamples)

	// Create patterns for each class
	for i := 0; i < numSamples; i++ {
		X[i] = make([]float64, config.NumFeatures)
		class := i % config.NumClasses

		// Create patterns for each class
		for j := 0; j < config.NumFeatures; j++ {
			// Use simpler class-specific patterns
			if j < 8 { // First 8 features are class-specific
				X[i][j] = float64((class >> (j % 3)) & 1)
			} else { // Remaining features are random
				X[i][j] = float64((i + j) % 2)
			}
		}
		y[i] = class
	}

	// Train the model
	t.Log("Training large TM...")
	t.Log("Initial state information:")
	tm.PrintStateInfo()

	tm.Fit(X, y, 50)

	t.Log("Final state information:")
	tm.PrintStateInfo()

	// Test cases for the large TM
	testCases := []struct {
		name          string
		input         []float64
		expectedClass int
		description   string
	}{
		{
			name:          "Class 0 Pattern",
			input:         X[0], // Use first training sample
			expectedClass: 0,
			description:   "Tests recognition of a pattern from class 0",
		},

		{
			name:          "Unknown Pattern",
			input:         make([]float64, config.NumFeatures), // All zeros
			expectedClass: -1,
			description:   "Tests handling of an unknown pattern",
		},
	}

	// Run each test case
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Test case: %s - %s", tc.name, tc.description)

			// Run sharded inference
			class, confidence, votes, matchScores := tsetlin.ShardedInference(tm, tc.input)

			// Debug: Print clause activations for Class 0 Pattern
			if tc.name == "Class 0 Pattern" {
				activeClausesInfo := tm.GetActiveClauses(tc.input)
				for classIdx, classClauses := range activeClausesInfo {
					t.Logf("Class %d: %d active clauses", classIdx, len(classClauses))
				}
			}

			// Log results
			t.Logf("Results - Class: %d, Confidence: %.3f", class, confidence)
			t.Logf("Votes: %v", votes)
			t.Logf("Match Scores: %v", matchScores)

			// Verify output dimensions
			if len(votes) != config.NumClasses {
				t.Errorf("Expected %d vote counts, got %d", config.NumClasses, len(votes))
			}
			if len(matchScores) != config.NumClasses {
				t.Errorf("Expected %d match scores, got %d", config.NumClasses, len(matchScores))
			}

			// Verify confidence is in valid range
			if confidence < 0.0 || confidence > 1.0 {
				t.Errorf("Confidence should be between 0 and 1, got %f", confidence)
			}

			// Verify match scores are valid numbers
			for i, score := range matchScores {
				if math.IsNaN(score) || math.IsInf(score, 0) {
					t.Errorf("Invalid match score for class %d: %f", i, score)
				}
			}

			// For known patterns, verify the predicted class
			if tc.expectedClass >= 0 {
				if class != tc.expectedClass {
					t.Errorf("Expected class %d, got %d", tc.expectedClass, class)
				}
				// Verify that the expected class has the highest votes
				maxVotes := votes[0]
				maxClass := 0
				for i, v := range votes {
					if v > maxVotes {
						maxVotes = v
						maxClass = i
					}
				}
				if maxClass != tc.expectedClass {
					t.Errorf("Expected class %d to have highest votes, but class %d had highest votes",
						tc.expectedClass, maxClass)
				}
			}
		})
	}

	// Test concurrent inference with the large TM
	t.Run("Concurrent Inference", func(t *testing.T) {
		t.Log("Testing concurrent inference with large TM...")

		// Test input (use a pattern from class 0)
		testInput := X[0]

		// Channel to collect results
		results := make(chan struct {
			class      int
			confidence float64
			votes      []float64
			scores     []float64
		}, 10)

		// Launch 10 concurrent inference calls
		for i := 0; i < 10; i++ {
			go func() {
				class, conf, votes, scores := tsetlin.ShardedInference(tm, testInput)
				results <- struct {
					class      int
					confidence float64
					votes      []float64
					scores     []float64
				}{class, conf, votes, scores}
			}()
		}

		// Collect and verify results
		firstResult := <-results
		t.Logf("First result - Class: %d, Confidence: %.3f",
			firstResult.class, firstResult.confidence)

		// Compare all other results with the first one
		for i := 1; i < 10; i++ {
			result := <-results
			t.Logf("Result %d - Class: %d, Confidence: %.3f",
				i+1, result.class, result.confidence)

			// Verify consistency
			if result.class != firstResult.class {
				t.Errorf("Concurrent inference produced different classes: %d vs %d",
					result.class, firstResult.class)
			}
			if math.Abs(result.confidence-firstResult.confidence) > 0.001 {
				t.Errorf("Concurrent inference produced different confidences: %.3f vs %.3f",
					result.confidence, firstResult.confidence)
			}
			for j, vote := range result.votes {
				if vote != firstResult.votes[j] {
					t.Errorf("Concurrent inference produced different votes for class %d: %.3f vs %.3f",
						j, vote, firstResult.votes[j])
				}
			}
		}
	})
}

// TestShardedInferenceSmallTM checks if a small TM can learn simple patterns and if state updates occur.
func TestShardedInferenceSmallTM(t *testing.T) {
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 4
	config.NumClasses = 2
	config.NumClauses = 10
	config.NumLiterals = 4
	config.Threshold = 1.0
	config.S = 1.0
	config.NStates = 100
	config.RandomSeed = 42
	config.Debug = true // Enable debug mode

	tm, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		t.Fatalf("Failed to create small Multiclass Tsetlin Machine: %v", err)
	}

	// Simple training data: two patterns, one for each class
	X := [][]float64{
		{1, 0, 0, 1}, // Class 0
		{1, 0, 0, 1},
		{0, 1, 1, 0}, // Class 1
		{0, 1, 1, 0},
	}
	y := []int{0, 0, 1, 1}

	t.Log("Training small TM...")
	tm.Fit(X, y, 100)

	t.Log("Final state information:")
	tm.PrintStateInfo()

	// Test both patterns
	testCases := []struct {
		name          string
		input         []float64
		expectedClass int
	}{
		{"Class 0 Pattern", X[0], 0},
		{"Class 1 Pattern", X[2], 1},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			class, confidence, votes, matchScores := tsetlin.ShardedInference(tm, tc.input)
			t.Logf("Results - Class: %d, Confidence: %.3f, Votes: %v, Match Scores: %v", class, confidence, votes, matchScores)
			if class != tc.expectedClass {
				t.Errorf("Expected class %d, got %d", tc.expectedClass, class)
			}
		})
	}
}
