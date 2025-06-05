package tests

import (
	"testing"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func TestNewPackedStates(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		expected int
	}{
		{"zero size", 0, 0},
		{"small size", 10, 3},
		{"exact word size", 64, 16},
		{"multiple words", 100, 25},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ps := tsetlin.NewPackedStates(tt.size)
			if len(ps) != tt.expected {
				t.Errorf("NewPackedStates(%d) = %d words, want %d", tt.size, len(ps), tt.expected)
			}
		})
	}
}

func TestPackedStatesGetSet(t *testing.T) {
	ps := tsetlin.NewPackedStates(100)
	tests := []struct {
		name     string
		index    int
		value    uint16
		expected uint16
	}{
		{"set first", 0, 10, 10},
		{"set last", 99, 20, 20},
		{"update first", 0, 15, 15},
		{"update last", 99, 25, 25},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ps.Set(tt.index, tt.value)
			if got := ps.Get(tt.index); got != tt.expected {
				t.Errorf("Get(%d) = %d, want %d", tt.index, got, tt.expected)
			}
		})
	}
}

func TestPackedStatesIncDec(t *testing.T) {
	ps := tsetlin.NewPackedStates(100)
	tests := []struct {
		name     string
		index    int
		initial  uint16
		inc      bool
		expected uint16
	}{
		{"increment from zero", 0, 0, true, 1},
		{"increment from max", 1, 0xFFFF, true, 0xFFFF},
		{"decrement from one", 2, 1, false, 0},
		{"decrement from zero", 3, 0, false, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ps.Set(tt.index, tt.initial)
			if tt.inc {
				ps.Inc(tt.index)
			} else {
				ps.Dec(tt.index)
			}
			if got := ps.Get(tt.index); got != tt.expected {
				t.Errorf("Get(%d) = %d, want %d", tt.index, got, tt.expected)
			}
		})
	}
}

func TestPackedStatesWordBoundaries(t *testing.T) {
	ps := tsetlin.NewPackedStates(100)

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
