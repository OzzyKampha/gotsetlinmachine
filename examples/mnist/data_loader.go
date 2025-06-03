package main

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

// MNISTData represents the loaded MNIST dataset
type MNISTData struct {
	TrainX [][]float64
	TrainY []int
	TestX  [][]float64
	TestY  []int
}

// LoadMNISTData loads and prepares the MNIST dataset
func LoadMNISTData(maxSamples int, trainRatio float64) (*MNISTData, error) {
	// Define paths for MNIST files
	dataDir := "data"
	imagePath := filepath.Join(dataDir, "train-images-idx3-ubyte")
	labelPath := filepath.Join(dataDir, "train-labels-idx1-ubyte")
	imageGzPath := imagePath + ".gz"
	labelGzPath := labelPath + ".gz"

	// Create data directory if it doesn't exist
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %v", err)
	}

	// Download and decompress MNIST image file if needed
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		fmt.Println("Downloading MNIST images...")
		if _, err := os.Stat(imageGzPath); os.IsNotExist(err) {
			if err := downloadFile("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", imageGzPath); err != nil {
				return nil, fmt.Errorf("failed to download images: %v", err)
			}
		}
		fmt.Println("Decompressing MNIST images...")
		if err := decompressGzip(imageGzPath, imagePath); err != nil {
			return nil, fmt.Errorf("failed to decompress images: %v", err)
		}
		os.Remove(imageGzPath)
	}

	// Download and decompress MNIST label file if needed
	if _, err := os.Stat(labelPath); os.IsNotExist(err) {
		fmt.Println("Downloading MNIST labels...")
		if _, err := os.Stat(labelGzPath); os.IsNotExist(err) {
			if err := downloadFile("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", labelGzPath); err != nil {
				return nil, fmt.Errorf("failed to download labels: %v", err)
			}
		}
		fmt.Println("Decompressing MNIST labels...")
		if err := decompressGzip(labelGzPath, labelPath); err != nil {
			return nil, fmt.Errorf("failed to decompress labels: %v", err)
		}
		os.Remove(labelGzPath)
	}

	fmt.Println("Loading MNIST data...")
	X, err := loadIDXImages(imagePath, maxSamples)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %v", err)
	}
	y, err := loadIDXLabels(labelPath, maxSamples)
	if err != nil {
		return nil, fmt.Errorf("failed to load labels: %v", err)
	}

	fmt.Printf("Loaded %d samples with %d features each.\n", len(X), len(X[0]))

	// Set random seed for reproducibility
	rand.Seed(42)

	// Split data into training and testing sets
	trainX, trainY, testX, testY := shuffleAndSplit(X, y, trainRatio)
	fmt.Printf("Training set size: %d samples\n", len(trainX))
	fmt.Printf("Testing set size: %d samples\n", len(testX))

	return &MNISTData{
		TrainX: trainX,
		TrainY: trainY,
		TestX:  testX,
		TestY:  testY,
	}, nil
}

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
