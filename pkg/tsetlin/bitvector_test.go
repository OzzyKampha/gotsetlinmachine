package tsetlin

import (
	"testing"
)

func TestNewBitVector(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		expected int
	}{
		{"zero size", 0, 0},
		{"small size", 10, 1},
		{"exact word size", 64, 1},
		{"multiple words", 100, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bv := NewBitVector(tt.size)
			if len(bv) != tt.expected {
				t.Errorf("NewBitVector(%d) = %d words, want %d", tt.size, len(bv), tt.expected)
			}
		})
	}
}

func TestBitVectorSetGet(t *testing.T) {
	bv := NewBitVector(100)
	tests := []struct {
		name     string
		index    int
		expected bool
	}{
		{"set first bit", 0, true},
		{"set middle bit", 50, true},
		{"set last bit", 99, true},
		{"unset bit", 25, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.expected {
				bv.Set(tt.index)
			}
			if got := bv.Get(tt.index); got != tt.expected {
				t.Errorf("Get(%d) = %v, want %v", tt.index, got, tt.expected)
			}
		})
	}
}

func TestBitVectorIsSubset(t *testing.T) {
	tests := []struct {
		name     string
		a        []int
		b        []int
		expected bool
	}{
		{"empty vectors", []int{}, []int{}, true},
		{"identical vectors", []int{1, 0, 1}, []int{1, 0, 1}, true},
		{"subset", []int{1, 0, 0}, []int{1, 1, 1}, true},
		{"not subset", []int{1, 1, 0}, []int{1, 0, 1}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := PackInputVector(tt.a)
			b := PackInputVector(tt.b)
			if got := a.IsSubset(b); got != tt.expected {
				t.Errorf("IsSubset() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestPackInputVector(t *testing.T) {
	tests := []struct {
		name     string
		input    []int
		expected []int
	}{
		{"empty", []int{}, []int{}},
		{"all zeros", []int{0, 0, 0}, []int{0, 0, 0}},
		{"all ones", []int{1, 1, 1}, []int{1, 1, 1}},
		{"mixed", []int{1, 0, 1}, []int{1, 0, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bv := PackInputVector(tt.input)
			for i, v := range tt.expected {
				if got := bv.Get(i); got != (v == 1) {
					t.Errorf("Get(%d) = %v, want %v", i, got, v == 1)
				}
			}
		})
	}
}
