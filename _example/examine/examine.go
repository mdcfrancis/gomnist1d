package main

import (
	"fmt"
	"github.com/mdcfrancis/gomnist1d"
)

func main() {
	m := mnist1d.NewDefault()

	ds := m.MakeDataset()
	for i := 0; i < 10; i++ {
		fmt.Println(ds.Y[i])
		fmt.Println(mnist1d.ShowAsImage(ds.X[i]))
		fmt.Println(ds.X[i])
	}
}
