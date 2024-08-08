package main

import (
	"fmt"
	mnist1d "github.com/mdcfrancis/gomnist1d"
)

func main() {
	m := mnist1d.NewDefault()
	//m.IidNoiseScale = 0
	//m.ShearScale = 0

	temp := m.Templates
	x := temp.X
	y := temp.Y
	m.Rnd.Seed(m.Seed)
	for i, l := range y {
		xs, _ := m.Transform(x[i], y)
		fmt.Println("Label", l)
		fmt.Println(mnist1d.ShowAsImage(xs))
		fmt.Println(xs)
	}
}
