package tests

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// BenchmarkDCEvents measures performance for DC-like event processing
func BenchmarkDCEvents(b *testing.B) {
	// Create configuration optimized for DC events
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 32 // Reduced feature set for DC events
	config.NumClauses = 100 // Optimized for DC event patterns
	config.NumLiterals = 8  // Balanced for DC event complexity
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create machine
	tm, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// Create realistic DC event patterns
	events := []struct {
		name     string
		features []float64
	}{
		{
			name: "Authentication Success",
			features: []float64{
				1, 0, 0, 0, 1, 0, 0, 0, // Event ID 4624 pattern
				0, 1, 0, 0, 0, 0, 0, 0, // User context
				0, 0, 1, 0, 0, 0, 0, 0, // Process info
				0, 0, 0, 1, 0, 0, 0, 0, // Network info
			},
		},
		{
			name: "Authentication Failure",
			features: []float64{
				0, 1, 0, 0, 1, 0, 0, 0, // Event ID 4625 pattern
				0, 1, 0, 0, 0, 0, 0, 0, // User context
				0, 0, 1, 0, 0, 0, 0, 0, // Process info
				0, 0, 0, 1, 0, 0, 0, 0, // Network info
			},
		},
		{
			name: "Account Management",
			features: []float64{
				0, 0, 1, 0, 0, 1, 0, 0, // Event ID 4720 pattern
				0, 1, 0, 0, 0, 0, 0, 0, // User context
				0, 0, 1, 0, 0, 0, 0, 0, // Process info
				0, 0, 0, 0, 1, 0, 0, 0, // Account info
			},
		},
	}

	// Reset timer before benchmark
	b.ResetTimer()

	// Run benchmark
	startTime := time.Now()
	processedEvents := 0

	for i := 0; i < b.N; i++ {
		// Process each event type
		for _, event := range events {
			score := tm.CalculateScore(event.features, 0)
			_ = score // Prevent compiler optimization
			processedEvents++
		}
	}

	// Calculate metrics
	duration := time.Since(startTime)
	eps := float64(processedEvents) / duration.Seconds()
	usPerEvent := float64(duration.Microseconds()) / float64(processedEvents)

	// Report metrics
	b.ReportMetric(eps, "EPS")
	b.ReportMetric(usPerEvent, "µs/event")
	b.ReportMetric(float64(config.NumFeatures*config.NumClauses)/eps, "features*clauses/EPS")

	// Print detailed analysis
	b.Logf("\nDetailed Analysis for DC Event Processing (Single Thread):")
	b.Logf("Total Events Processed: %d", processedEvents)
	b.Logf("Duration: %v", duration)
	b.Logf("Events Per Second: %.2f", eps)
	b.Logf("Microseconds Per Event: %.2f", usPerEvent)
	b.Logf("Features*Clauses Per Second: %.2f", float64(config.NumFeatures*config.NumClauses)/eps)
}

// BenchmarkDCEventsParallel measures parallel performance for DC-like event processing
func BenchmarkDCEventsParallel(b *testing.B) {
	// Create configuration optimized for DC events
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 32 // Reduced feature set for DC events
	config.NumClauses = 100 // Optimized for DC event patterns
	config.NumLiterals = 8  // Balanced for DC event complexity
	config.Threshold = 5.0
	config.S = 3.9
	config.NStates = 100
	config.RandomSeed = 42

	// Create machine
	tm, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		b.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// Create realistic DC event patterns
	events := []struct {
		name     string
		features []float64
	}{
		{
			name: "Authentication Success",
			features: []float64{
				1, 0, 0, 0, 1, 0, 0, 0, // Event ID 4624 pattern
				0, 1, 0, 0, 0, 0, 0, 0, // User context
				0, 0, 1, 0, 0, 0, 0, 0, // Process info
				0, 0, 0, 1, 0, 0, 0, 0, // Network info
			},
		},
		{
			name: "Authentication Failure",
			features: []float64{
				0, 1, 0, 0, 1, 0, 0, 0, // Event ID 4625 pattern
				0, 1, 0, 0, 0, 0, 0, 0, // User context
				0, 0, 1, 0, 0, 0, 0, 0, // Process info
				0, 0, 0, 1, 0, 0, 0, 0, // Network info
			},
		},
		{
			name: "Account Management",
			features: []float64{
				0, 0, 1, 0, 0, 1, 0, 0, // Event ID 4720 pattern
				0, 1, 0, 0, 0, 0, 0, 0, // User context
				0, 0, 1, 0, 0, 0, 0, 0, // Process info
				0, 0, 0, 0, 1, 0, 0, 0, // Account info
			},
		},
	}

	// Get number of CPU cores
	numCPU := runtime.NumCPU()
	b.Logf("Running parallel benchmark with %d CPU cores", numCPU)

	// Reset timer before benchmark
	b.ResetTimer()

	// Run benchmark
	startTime := time.Now()
	var processedEvents int64
	var wg sync.WaitGroup

	// Create a channel to distribute work
	eventChan := make(chan struct {
		name     string
		features []float64
	}, numCPU*2)

	// Start worker goroutines
	for i := 0; i < numCPU; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for event := range eventChan {
				score := tm.CalculateScore(event.features, 0)
				_ = score // Prevent compiler optimization
				atomic.AddInt64(&processedEvents, 1)
			}
		}()
	}

	// Feed events to workers
	for i := 0; i < b.N; i++ {
		for _, event := range events {
			eventChan <- event
		}
	}
	close(eventChan)

	// Wait for all workers to finish
	wg.Wait()

	// Calculate metrics
	duration := time.Since(startTime)
	eps := float64(processedEvents) / duration.Seconds()
	usPerEvent := float64(duration.Microseconds()) / float64(processedEvents)

	// Report metrics
	b.ReportMetric(eps, "EPS")
	b.ReportMetric(usPerEvent, "µs/event")
	b.ReportMetric(float64(config.NumFeatures*config.NumClauses)/eps, "features*clauses/EPS")

	// Print detailed analysis
	b.Logf("\nDetailed Analysis for DC Event Processing (Parallel):")
	b.Logf("Total Events Processed: %d", processedEvents)
	b.Logf("Duration: %v", duration)
	b.Logf("Events Per Second: %.2f", eps)
	b.Logf("Microseconds Per Event: %.2f", usPerEvent)
	b.Logf("Features*Clauses Per Second: %.2f", float64(config.NumFeatures*config.NumClauses)/eps)
	b.Logf("CPU Cores Used: %d", numCPU)
}
