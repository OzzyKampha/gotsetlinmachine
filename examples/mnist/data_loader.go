package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
)

const (
	mnistBaseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
	trainImages  = "train-images-idx3-ubyte.gz"
	trainLabels  = "train-labels-idx1-ubyte.gz"
)

// downloadMNISTData downloads the MNIST dataset if it doesn't exist
func downloadMNISTData(baseDir string) error {
	// Create data directory if it doesn't exist
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	// Download and extract training images
	imagesPath := filepath.Join(baseDir, "train-images-idx3-ubyte")
	if _, err := os.Stat(imagesPath); os.IsNotExist(err) {
		if err := downloadAndExtract(mnistBaseURL+trainImages, imagesPath); err != nil {
			return fmt.Errorf("failed to download training images: %v", err)
		}
	}

	// Download and extract training labels
	labelsPath := filepath.Join(baseDir, "train-labels-idx1-ubyte")
	if _, err := os.Stat(labelsPath); os.IsNotExist(err) {
		if err := downloadAndExtract(mnistBaseURL+trainLabels, labelsPath); err != nil {
			return fmt.Errorf("failed to download training labels: %v", err)
		}
	}

	return nil
}

// downloadAndExtract downloads a gzipped file and extracts it
func downloadAndExtract(url, outputPath string) error {
	// Download the file
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Create a gzip reader
	gzReader, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %v", err)
	}
	defer gzReader.Close()

	// Create output file
	out, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Copy the decompressed data to the output file
	_, err = io.Copy(out, gzReader)
	return err
}

// MNISTData holds the training and test data for MNIST
type MNISTData struct {
	TrainX [][]float64
	TrainY []int
	TestX  [][]float64
	TestY  []int
}

// LoadMNISTData loads the MNIST dataset and returns training and test data
func LoadMNISTData(maxSamples int, trainRatio float64) (*MNISTData, error) {
	// Define paths to MNIST files
	baseDir := "examples/mnist/data"

	// Download data if it doesn't exist
	if err := downloadMNISTData(baseDir); err != nil {
		return nil, fmt.Errorf("failed to download MNIST data: %v", err)
	}

	imagesFile := filepath.Join(baseDir, "train-images-idx3-ubyte")
	labelsFile := filepath.Join(baseDir, "train-labels-idx1-ubyte")

	// Load images
	images, err := loadImages(imagesFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %v", err)
	}

	// Load labels
	labels, err := loadLabels(labelsFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load labels: %v", err)
	}

	// Limit number of samples if needed
	if maxSamples > 0 && maxSamples < len(images) {
		images = images[:maxSamples]
		labels = labels[:maxSamples]
	}

	// Split into training and test sets
	trainSize := int(float64(len(images)) * trainRatio)

	// Shuffle the data
	indices := make([]int, len(images))
	for i := range indices {
		indices[i] = i
	}
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Create training and test sets
	trainX := make([][]float64, trainSize)
	trainY := make([]int, trainSize)
	testX := make([][]float64, len(images)-trainSize)
	testY := make([]int, len(images)-trainSize)

	for i, idx := range indices {
		if i < trainSize {
			trainX[i] = images[idx]
			trainY[i] = labels[idx]
		} else {
			testX[i-trainSize] = images[idx]
			testY[i-trainSize] = labels[idx]
		}
	}

	return &MNISTData{
		TrainX: trainX,
		TrainY: trainY,
		TestX:  testX,
		TestY:  testY,
	}, nil
}

// loadImages loads MNIST images from the given file
func loadImages(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read magic number
	var magic uint32
	if err := binary.Read(file, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 2051 {
		return nil, fmt.Errorf("invalid magic number for images file: %d", magic)
	}

	// Read number of images
	var numImages uint32
	if err := binary.Read(file, binary.BigEndian, &numImages); err != nil {
		return nil, err
	}

	// Read image dimensions
	var rows, cols uint32
	if err := binary.Read(file, binary.BigEndian, &rows); err != nil {
		return nil, err
	}
	if err := binary.Read(file, binary.BigEndian, &cols); err != nil {
		return nil, err
	}

	// Read images
	images := make([][]float64, numImages)
	for i := uint32(0); i < numImages; i++ {
		image := make([]float64, rows*cols)
		for j := uint32(0); j < rows*cols; j++ {
			var pixel uint8
			if err := binary.Read(file, binary.BigEndian, &pixel); err != nil {
				return nil, err
			}
			image[j] = float64(pixel) / 255.0 // Normalize to [0,1]
		}
		images[i] = image
	}

	return images, nil
}

// loadLabels loads MNIST labels from the given file
func loadLabels(filename string) ([]int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read magic number
	var magic uint32
	if err := binary.Read(file, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 2049 {
		return nil, fmt.Errorf("invalid magic number for labels file: %d", magic)
	}

	// Read number of labels
	var numLabels uint32
	if err := binary.Read(file, binary.BigEndian, &numLabels); err != nil {
		return nil, err
	}

	// Read labels
	labels := make([]int, numLabels)
	for i := uint32(0); i < numLabels; i++ {
		var label uint8
		if err := binary.Read(file, binary.BigEndian, &label); err != nil {
			return nil, err
		}
		labels[i] = int(label)
	}

	return labels, nil
}
