package main

import (
	"bufio"
	"fmt"

	//"bufio"
	//"fmt"
	//"log"
	//"os"
	"strings"
)

func main() {

	in := `
2000
5 3
1000x1000
1000x1500
640x930
640x1500
3000x1000
`

	//var w int64
	//_, err := fmt.Scan(&w)

	//fmt.Println(w)

	scanner := bufio.NewScanner(strings.NewReader(in))
	scanner.Scan()
	fmt.Printf(scanner.Text())
	//err := scanner.Err()
}
