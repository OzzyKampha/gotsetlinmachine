package tests

import (
	"math/rand"
	"os"
	"testing"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func TestMain(m *testing.M) {
	rand.Seed(42)
	os.Exit(m.Run())
}

func TestNewTsetlinMachine(t *testing.T) {
	tests := []struct {
		name          string
		numClauses    int
		numFeatures   int
		voteThreshold int
		s             int
	}{
		{"default threshold", 10, 5, -1, 3},
		{"custom threshold", 20, 10, 15, 5},
		{"zero threshold", 30, 15, 0, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tm := tsetlin.NewTsetlinMachine(tt.numClauses, tt.numFeatures, tt.voteThreshold, tt.s)
			if len(tm.Clauses) != tt.numClauses {
				t.Errorf("NewTsetlinMachine() clauses = %d, want %d", len(tm.Clauses), tt.numClauses)
			}
			if tm.NumFeatures != tt.numFeatures {
				t.Errorf("NewTsetlinMachine() features = %d, want %d", tm.NumFeatures, tt.numFeatures)
			}
			if tt.voteThreshold == -1 {
				if tm.VoteThreshold != tt.numClauses/2 {
					t.Errorf("NewTsetlinMachine() threshold = %d, want %d", tm.VoteThreshold, tt.numClauses/2)
				}
			} else {
				if tm.VoteThreshold != tt.voteThreshold {
					t.Errorf("NewTsetlinMachine() threshold = %d, want %d", tm.VoteThreshold, tt.voteThreshold)
				}
			}
			if tm.S != tt.s {
				t.Errorf("NewTsetlinMachine() s = %d, want %d", tm.S, tt.s)
			}
		})
	}
}

func TestEvaluateClause(t *testing.T) {
	tests := []struct {
		name     string
		clause   tsetlin.Clause
		input    []int
		expected bool
	}{
		{
			"empty clause",
			tsetlin.Clause{},
			[]int{},
			true,
		},
		{
			"simple match",
			tsetlin.Clause{
				Include: tsetlin.NewPackedStates(2),
				Exclude: tsetlin.NewPackedStates(2),
			},
			[]int{1, 0},
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bv := tsetlin.PackInputVector(tt.input)
			if got := tsetlin.EvaluateClause(tt.clause, bv); got != tt.expected {
				t.Errorf("EvaluateClause() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestTsetlinMachinePredict(t *testing.T) {
	tm := tsetlin.NewTsetlinMachine(10, 5, 5, 3)
	tests := []struct {
		name     string
		input    []int
		expected int
	}{
		{"empty input", []int{}, 0},
		{"all zeros", []int{0, 0, 0, 0, 0}, 0},
		{"all ones", []int{1, 1, 1, 1, 1}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tm.Predict(tt.input, 0).(int); got != tt.expected {
				t.Errorf("Predict() = %d, want %d", got, tt.expected)
			}
		})
	}
}

func TestTsetlinMachineBatchPredict(t *testing.T) {
	tm := tsetlin.NewTsetlinMachine(10, 5, 5, 3)
	tests := []struct {
		name     string
		inputs   [][]int
		expected []int
	}{
		{
			"multiple inputs",
			[][]int{
				{0, 0, 0, 0, 0},
				{1, 1, 1, 1, 1},
				{1, 0, 1, 0, 1},
			},
			[]int{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tm.Predict(tt.inputs, 0).([]int)
			if len(got) != len(tt.expected) {
				t.Errorf("Predict() returned %d results, want %d", len(got), len(tt.expected))
				return
			}
			for i := range got {
				if got[i] != tt.expected[i] {
					t.Errorf("Predict()[%d] = %d, want %d", i, got[i], tt.expected[i])
				}
			}
		})
	}
}

func TestTsetlinMachineFit(t *testing.T) {
	tm := tsetlin.NewTsetlinMachine(10, 5, 5, 3)
	X := [][]int{
		{1, 0, 1, 0, 1},
		{0, 1, 0, 1, 0},
	}
	Y := []int{1, 0}

	// Test that Fit doesn't panic
	t.Run("fit no panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Fit() panicked: %v", r)
			}
		}()
		tm.Fit(X, Y, 1, 1)
	})
}

func TestNewMultiClassTM(t *testing.T) {
	tests := []struct {
		name        string
		numClasses  int
		numClauses  int
		numFeatures int
		threshold   int
		s           int
	}{
		{"binary", 2, 10, 5, 5, 3},
		{"multiclass", 3, 20, 10, 10, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tsetlin.NewMultiClassTM(tt.numClasses, tt.numClauses, tt.numFeatures, tt.threshold, tt.s)
			if len(m.Classes) != tt.numClasses {
				t.Errorf("NewMultiClassTM() classes = %d, want %d", len(m.Classes), tt.numClasses)
			}
			for i, tm := range m.Classes {
				if len(tm.Clauses) != tt.numClauses {
					t.Errorf("NewMultiClassTM() class %d clauses = %d, want %d", i, len(tm.Clauses), tt.numClauses)
				}
				if tm.NumFeatures != tt.numFeatures {
					t.Errorf("NewMultiClassTM() class %d features = %d, want %d", i, tm.NumFeatures, tt.numFeatures)
				}
				if tm.VoteThreshold != tt.threshold {
					t.Errorf("NewMultiClassTM() class %d threshold = %d, want %d", i, tm.VoteThreshold, tt.threshold)
				}
				if tm.S != tt.s {
					t.Errorf("NewMultiClassTM() class %d s = %d, want %d", i, tm.S, tt.s)
				}
			}
		})
	}
}

func TestMultiClassTMPredict(t *testing.T) {
	m := tsetlin.NewMultiClassTM(3, 10, 5, 5, 3)
	tests := []struct {
		name     string
		input    []int
		expected int
	}{
		{"empty input", []int{}, 0},
		{"all zeros", []int{0, 0, 0, 0, 0}, 0},
		{"all ones", []int{1, 1, 1, 1, 1}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := m.Predict(tt.input); got != tt.expected {
				t.Errorf("Predict() = %d, want %d", got, tt.expected)
			}
		})
	}
}

func TestMultiClassTMFit(t *testing.T) {
	m := tsetlin.NewMultiClassTM(3, 10, 5, 5, 3)
	X := [][]int{
		{1, 0, 1, 0, 1},
		{0, 1, 0, 1, 0},
	}
	Y := []int{0, 1}

	// Test that Fit doesn't panic
	t.Run("fit no panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Fit() panicked: %v", r)
			}
		}()
		m.Fit(X, Y, 1)
	})
}
