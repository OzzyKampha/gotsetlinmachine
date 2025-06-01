package model

import (
	"testing"
)

func TestNewTsetlinMachine(t *testing.T) {
	tests := []struct {
		name        string
		numClasses  int
		numFeatures int
		numClauses  int
		numLiterals int
		threshold   float64
		s           float64
		wantErr     bool
	}{
		{
			name:        "Valid parameters",
			numClasses:  2,
			numFeatures: 4,
			numClauses:  10,
			numLiterals: 4,
			threshold:   0.5,
			s:           3.9,
			wantErr:     false,
		},
		{
			name:        "Invalid numClasses",
			numClasses:  0,
			numFeatures: 4,
			numClauses:  10,
			numLiterals: 4,
			threshold:   0.5,
			s:           3.9,
			wantErr:     true,
		},
		{
			name:        "Invalid numFeatures",
			numClasses:  2,
			numFeatures: 0,
			numClauses:  10,
			numLiterals: 4,
			threshold:   0.5,
			s:           3.9,
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tm := NewTsetlinMachine(tt.numClasses, tt.numFeatures, tt.numClauses, tt.numLiterals, tt.threshold, tt.s)
			if tm == nil && !tt.wantErr {
				t.Errorf("NewTsetlinMachine() returned nil for valid parameters")
			}
			if tm != nil && tt.wantErr {
				t.Errorf("NewTsetlinMachine() returned non-nil for invalid parameters")
			}
		})
	}
}

func TestTsetlinAutomaton(t *testing.T) {
	ta := TsetlinAutomaton{
		state:   5,
		nStates: 10,
	}

	// Test reward
	ta.reward()
	if ta.state != 6 {
		t.Errorf("reward() = %v, want %v", ta.state, 6)
	}

	// Test penalty
	ta.penalty()
	if ta.state != 5 {
		t.Errorf("penalty() = %v, want %v", ta.state, 5)
	}

	// Test getAction
	action := ta.getAction()
	if action != 1 {
		t.Errorf("getAction() = %v, want %v", action, 1)
	}

	// Test state bounds
	ta.state = 0
	ta.penalty()
	if ta.state != 0 {
		t.Errorf("penalty() at state 0 = %v, want %v", ta.state, 0)
	}

	ta.state = 10
	ta.reward()
	if ta.state != 10 {
		t.Errorf("reward() at max state = %v, want %v", ta.state, 10)
	}
}

func TestMultiClassTsetlinMachine(t *testing.T) {
	mctm := NewMultiClassTsetlinMachine(3, 4, 10, 4, 0.5, 3.9, 20)

	// Test initialization
	if mctm.numClasses != 3 {
		t.Errorf("numClasses = %v, want %v", mctm.numClasses, 3)
	}
	if mctm.numFeatures != 4 {
		t.Errorf("numFeatures = %v, want %v", mctm.numFeatures, 4)
	}
	if len(mctm.machines) != 3 {
		t.Errorf("len(machines) = %v, want %v", len(mctm.machines), 3)
	}

	// Test SetRandomState
	mctm.SetRandomState(42)
	for _, machine := range mctm.machines {
		if machine.rng == nil {
			t.Error("SetRandomState() did not initialize random number generator")
		}
	}

	// Test SetDebug
	mctm.SetDebug(true)
	for _, machine := range mctm.machines {
		if !machine.debug {
			t.Error("SetDebug() did not set debug flag")
		}
	}
}

func BenchmarkTsetlinMachine(b *testing.B) {
	mctm := NewMultiClassTsetlinMachine(3, 4, 10, 4, 0.5, 3.9, 20)
	input := []float64{0.1, 0.2, 0.3, 0.4}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mctm.PredictClass(input)
	}
}
