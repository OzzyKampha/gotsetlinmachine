package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// loadIDXImages loads MNIST images from an IDX file and returns a slice of binarized images
func loadIDXImages(path string, maxSamples int) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, numImages, numRows, numCols int32
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &numImages); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &numRows); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &numCols); err != nil {
		return nil, err
	}

	if maxSamples > 0 && int32(maxSamples) < numImages {
		numImages = int32(maxSamples)
	}

	images := make([][]float64, numImages)
	buf := make([]byte, numRows*numCols)
	for i := int32(0); i < numImages; i++ {
		if _, err := io.ReadFull(f, buf); err != nil {
			return nil, err
		}
		img := make([]float64, numRows*numCols)
		for j, px := range buf {
			if px > 127 {
				img[j] = 1.0
			} else {
				img[j] = 0.0
			}
		}
		images[i] = img
	}
	return images, nil
}

// loadIDXLabels loads MNIST labels from an IDX file
func loadIDXLabels(path string, maxSamples int) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, numLabels int32
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &numLabels); err != nil {
		return nil, err
	}

	if maxSamples > 0 && int32(maxSamples) < numLabels {
		numLabels = int32(maxSamples)
	}

	labels := make([]int, numLabels)
	buf := make([]byte, numLabels)
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, err
	}
	for i, b := range buf {
		labels[i] = int(b)
	}
	return labels, nil
}

func downloadFile(url, dest string) error {
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return nil
		},
	}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept-Encoding", "gzip")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func decompressGzip(gzPath, destPath string) error {
	gzFile, err := os.Open(gzPath)
	if err != nil {
		return err
	}
	defer gzFile.Close()

	gzReader, err := gzip.NewReader(gzFile)
	if err != nil {
		return err
	}
	defer gzReader.Close()

	outFile, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	_, err = io.Copy(outFile, gzReader)
	return err
}

// shuffleAndSplit shuffles the data and splits it into training and testing sets
func shuffleAndSplit(X [][]float64, y []int, trainRatio float64) ([][]float64, []int, [][]float64, []int) {
	// Create indices array and shuffle it
	indices := make([]int, len(X))
	for i := range indices {
		indices[i] = i
	}
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Calculate split point
	splitPoint := int(float64(len(X)) * trainRatio)

	// Split the data
	trainX := make([][]float64, splitPoint)
	trainY := make([]int, splitPoint)
	testX := make([][]float64, len(X)-splitPoint)
	testY := make([]int, len(X)-splitPoint)

	for i, idx := range indices[:splitPoint] {
		trainX[i] = X[idx]
		trainY[i] = y[idx]
	}
	for i, idx := range indices[splitPoint:] {
		testX[i] = X[idx]
		testY[i] = y[idx]
	}

	return trainX, trainY, testX, testY
}

func main() {
	// Define paths for MNIST files
	dataDir := "data"
	imagePath := filepath.Join(dataDir, "train-images-idx3-ubyte")
	labelPath := filepath.Join(dataDir, "train-labels-idx1-ubyte")
	imageGzPath := imagePath + ".gz"
	labelGzPath := labelPath + ".gz"

	// Create data directory if it doesn't exist
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		log.Fatalf("Failed to create data directory: %v", err)
	}

	// Download and decompress MNIST image file if needed
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		fmt.Println("Downloading MNIST images...")
		if _, err := os.Stat(imageGzPath); os.IsNotExist(err) {
			if err := downloadFile("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", imageGzPath); err != nil {
				log.Fatalf("Failed to download images: %v", err)
			}
		}
		fmt.Println("Decompressing MNIST images...")
		if err := decompressGzip(imageGzPath, imagePath); err != nil {
			log.Fatalf("Failed to decompress images: %v", err)
		}
		os.Remove(imageGzPath)
	}

	// Download and decompress MNIST label file if needed
	if _, err := os.Stat(labelPath); os.IsNotExist(err) {
		fmt.Println("Downloading MNIST labels...")
		if _, err := os.Stat(labelGzPath); os.IsNotExist(err) {
			if err := downloadFile("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", labelGzPath); err != nil {
				log.Fatalf("Failed to download labels: %v", err)
			}
		}
		fmt.Println("Decompressing MNIST labels...")
		if err := decompressGzip(labelGzPath, labelPath); err != nil {
			log.Fatalf("Failed to decompress labels: %v", err)
		}
		os.Remove(labelGzPath)
	}

	fmt.Println("Loading MNIST data...")
	X, err := loadIDXImages(imagePath, 0) // Load all samples
	if err != nil {
		log.Fatalf("Failed to load images: %v", err)
	}
	y, err := loadIDXLabels(labelPath, 0) // Load all samples
	if err != nil {
		log.Fatalf("Failed to load labels: %v", err)
	}

	fmt.Printf("Loaded %d samples with %d features each.\n", len(X), len(X[0]))

	// Set random seed for reproducibility
	rand.Seed(42)

	// Split data into training (90%) and testing (10%) sets
	trainX, trainY, testX, testY := shuffleAndSplit(X, y, 0.9)
	fmt.Printf("Training set size: %d samples\n", len(trainX))
	fmt.Printf("Testing set size: %d samples\n", len(testX))

	// Configure the Tsetlin Machine for multiclass classification
	config := tsetlin.DefaultConfig()
	config.NumFeatures = len(X[0])
	config.NumClasses = 10
	config.NumClauses = 2000
	config.NumLiterals = len(X[0])
	config.Threshold = 50.0
	config.S = 10.0
	config.NStates = 100
	config.RandomSeed = 42
	config.Debug = false

	machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		log.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}
	for i := 0; i < 10; i++ {
		fmt.Printf("\nEpoch %d/%d\n", i+1, 10)
		fmt.Println("Training the Tsetlin Machine...")
		if err := machine.Fit(trainX, trainY, 1); err != nil {
			log.Fatalf("Training failed: %v", err)
		}

		// Calculate training accuracy
		trainCorrect := 0
		for i := 0; i < len(trainX); i++ {
			pred, err := machine.PredictClass(trainX[i])
			if err != nil {
				log.Printf("Training prediction error: %v", err)
				continue
			}
			if pred == trainY[i] {
				trainCorrect++
			}
		}
		trainAcc := float64(trainCorrect) / float64(len(trainX))
		fmt.Printf("Training accuracy: %.2f%% (%d/%d)\n", trainAcc*100, trainCorrect, len(trainX))

		// Calculate test accuracy
		fmt.Println("Evaluating on test set...")
		testCorrect := 0
		for i := 0; i < len(testX); i++ {
			pred, err := machine.PredictClass(testX[i])
			if err != nil {
				log.Printf("Test prediction error: %v", err)
				continue
			}
			if pred == testY[i] {
				testCorrect++
			}
		}
		testAcc := float64(testCorrect) / float64(len(testX))
		fmt.Printf("Test accuracy: %.2f%% (%d/%d)\n", testAcc*100, testCorrect, len(testX))
	}

	// Show predictions for first 10 test samples
	fmt.Println("\nSample predictions from test set:")
	for i := 0; i < 10; i++ {
		result, _ := machine.Predict(testX[i])
		fmt.Printf("Sample %d: True=%d, Predicted=%d, Confidence=%.2f, Votes=%v\n",
			i, testY[i], result.PredictedClass, result.Confidence, result.Votes)
	}
}
