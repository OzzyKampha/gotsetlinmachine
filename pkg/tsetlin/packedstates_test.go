package tsetlin

import (
	"testing"
)

func TestNewPackedStates(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		expected int
	}{
		{"small size", 10, 3},
		{"exact multiple of 4", 12, 3},
		{"multiple words", 100, 25},
		{"zero size", 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ps := NewPackedStates(tt.size)
			if len(ps) != tt.expected {
				t.Errorf("NewPackedStates(%d) = %d words, want %d", tt.size, len(ps), tt.expected)
			}
		})
	}
}

func TestPackedStatesGetSet(t *testing.T) {
	ps := NewPackedStates(100)

	// Test setting and getting values
	testValues := []struct {
		idx int
		val uint16
	}{
		{0, 42},
		{1, 100},
		{3, 255},
		{4, 0},
		{99, 65535},
	}

	for _, tv := range testValues {
		ps.Set(tv.idx, tv.val)
		if got := ps.Get(tv.idx); got != tv.val {
			t.Errorf("Get(%d) = %d, want %d after Set", tv.idx, got, tv.val)
		}
	}

	// Test values that weren't set
	unsetIndices := []int{2, 5, 98}
	for _, idx := range unsetIndices {
		if got := ps.Get(idx); got != 0 {
			t.Errorf("Get(%d) = %d, want 0 (unset value)", idx, got)
		}
	}
}

func TestPackedStatesIncDec(t *testing.T) {
	ps := NewPackedStates(100)

	// Test increment
	ps.Set(0, 0)
	ps.Inc(0)
	if got := ps.Get(0); got != 1 {
		t.Errorf("Inc(0) from 0 = %d, want 1", got)
	}

	// Test increment at max
	ps.Set(1, stateMax)
	ps.Inc(1)
	if got := ps.Get(1); got != stateMax {
		t.Errorf("Inc(1) at max = %d, want %d", got, stateMax)
	}

	// Test decrement
	ps.Set(2, 1)
	ps.Dec(2)
	if got := ps.Get(2); got != 0 {
		t.Errorf("Dec(2) from 1 = %d, want 0", got)
	}

	// Test decrement at zero
	ps.Set(3, 0)
	ps.Dec(3)
	if got := ps.Get(3); got != 0 {
		t.Errorf("Dec(3) at zero = %d, want 0", got)
	}
}

func TestPackedStatesWordBoundaries(t *testing.T) {
	ps := NewPackedStates(100)

	// Test values at word boundaries
	boundaryTests := []struct {
		idx int
		val uint16
	}{
		{3, 42},  // Last value in first word
		{4, 100}, // First value in second word
		{7, 255}, // Last value in second word
		{8, 0},   // First value in third word
	}

	for _, bt := range boundaryTests {
		ps.Set(bt.idx, bt.val)
		if got := ps.Get(bt.idx); got != bt.val {
			t.Errorf("Get(%d) = %d, want %d (word boundary test)", bt.idx, got, bt.val)
		}
	}
}
