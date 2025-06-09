// Package tsetlin implements the Tsetlin Machine algorithm for binary and multiclass classification.
// The Tsetlin Machine is a novel machine learning algorithm that uses propositional logic
// to learn patterns from data. It is particularly effective for binary and multiclass
// classification tasks, offering interpretable results and efficient training.
package tsetlin

// Constants for the Tsetlin Machine implementation
const (
	// wordSize defines the number of bits in a uint64 word, used for bit packing
	wordSize = 64

	// T is the threshold for prediction decisions in the Tsetlin Machine

	// stateMax defines the maximum value for automaton states
	stateMax = 100

	// ActivationThreshold is the threshold value that determines when a Tsetlin automaton
	// activates its associated literal
	ActivationThreshold = (stateMax / 2) - 1
)

// BitVector represents a vector of bits packed into uint64 words for efficient storage
// and operations. Each bit in the vector can be either 0 or 1.
type BitVector []uint64

// PackedStates represents a vector of 16-bit states packed into uint64 words.
// Each state value can range from 0 to stateMax, representing the state of a
// Tsetlin automaton.
type PackedStates []uint64

// Clause represents a single clause in the Tsetlin Machine.
// A clause is a conjunction of literals (features or their negations) that
// contributes to the final prediction.
type Clause struct {
	// include contains the states of automata for included literals
	Include PackedStates

	// exclude contains the states of automata for excluded literals
	Exclude PackedStates

	// Vote determines whether the clause votes for or against the positive class
	// (1 for positive, -1 for negative)
	Vote int

	// Weight represents the importance of this clause in the final prediction
	Weight float64

	// dropoutProb is the probability of dropping out this clause during training
	// to prevent overfitting
	DropoutProb float32

	FeatureMask BitVector
}

// TsetlinMachine represents a single Tsetlin Machine classifier.
// It consists of multiple clauses that work together to make predictions.
type TsetlinMachine struct {
	// Clauses is the set of clauses that make up the Tsetlin Machine
	Clauses []Clause

	// NumFeatures is the number of input features
	NumFeatures int

	// VoteThreshold is the minimum number of votes required for a positive prediction
	VoteThreshold int

	// S is the specificity parameter that controls the probability of type I feedback
	S float64
}

// MultiClassTM represents a multiclass Tsetlin Machine classifier.
// It consists of multiple binary Tsetlin Machines, one for each class.
// type MultiClassTM struct {
// 	// Classes contains one Tsetlin Machine for each class
// 	Classes []*TsetlinMachine
// }

type ClauseUpdateTask struct {
	ClauseIndex int       // Index of the clause in the clause list
	Clause      *Clause   // Pointer to the clause being updated
	Feedback    int       // Feedback signal (+1 or -1)
	Input       BitVector // Input vector for this training sample
	S           float64   // Specificity parameter for feedback control
}
