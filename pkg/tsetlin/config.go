package tsetlin

import "fmt"

// Config holds all the configuration parameters for the Tsetlin Machine
type Config struct {
	// Core parameters
	NumClauses    int `json:"num_clauses"`    // Number of clauses in the Tsetlin Machine
	NumFeatures   int `json:"num_features"`   // Number of input features
	VoteThreshold int `json:"vote_threshold"` // Threshold for prediction (default: numClauses/2)
	S             int `json:"s"`              // S parameter for feedback probability (1/s)

	// For multi-class classification
	NumClasses int `json:"num_classes"` // Number of classes for multi-class classification

	// Training parameters
	Epochs int `json:"epochs"` // Number of training epochs

	// Constants (these are typically not changed)
	T        int `json:"t"`         // Used for prediction decision (default: 100)
	StateMax int `json:"state_max"` // Defines automaton state range [0, stateMax] (default: 199)
}

// DefaultConfig returns a configuration with default values
func DefaultConfig() *Config {
	return &Config{
		NumClauses:    200, // Increased from 100 to 200 for better pattern recognition
		NumFeatures:   10,
		VoteThreshold: -1, // Will be set to numClauses/2 in NewTsetlinMachine
		S:             5,  // Increased from 3 to 5 for more stable learning
		NumClasses:    2,
		Epochs:        200, // Increased from 100 to 200 for better convergence
		T:             100,
		StateMax:      199,
	}
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.NumClauses <= 0 {
		return fmt.Errorf("num_clauses must be positive")
	}
	if c.NumFeatures <= 0 {
		return fmt.Errorf("num_features must be positive")
	}
	if c.S <= 0 {
		return fmt.Errorf("s must be positive")
	}
	if c.NumClasses < 2 {
		return fmt.Errorf("num_classes must be at least 2")
	}
	if c.Epochs <= 0 {
		return fmt.Errorf("epochs must be positive")
	}
	return nil
}
