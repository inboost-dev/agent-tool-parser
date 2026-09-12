package main

import (
	"fmt"
	"log"

	agenttoolparser "github.com/inboost-dev/agent-tool-parser/bindings/go/agenttoolparser"
)

func main() {
	sample := `<｜tool calls begin｜><｜tool call begin｜>function=search<｜tool sep｜>{"pattern": "parse_tool_call", "path": "."}<｜tool call end｜><｜tool calls end｜>`

	call, err := agenttoolparser.ParseToolCall(sample)
	if err != nil {
		log.Fatalf("Failed to parse tool call: %v", err)
	}

	fmt.Printf("Tool Name: %s\n", call.Name)
	fmt.Printf("Arguments: %v\n", call.Args)
	fmt.Printf("Raw Source: %s\n", call.RawSource)
}
