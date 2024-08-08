package data

import (
	"encoding/json"
	mnist1d "github.com/mdcfrancis/gomnist1d"
	"os"
)

func ParseMNISTIMageFile(b []byte) (*mnist1d.DataSet, error) {
	ds := mnist1d.DataSet{}
	err := json.Unmarshal(b, &ds)
	if err != nil {
		return nil, err
	}
	return &ds, nil
}

func LoadMNISTImageFile(path string) (*mnist1d.DataSet, error) {
	// open the file and convert from json
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseMNISTIMageFile(b)
}
