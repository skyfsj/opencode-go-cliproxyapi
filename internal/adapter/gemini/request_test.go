package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	"opencode-go-cliproxyapi/internal/errclass"
)

// jsonMap is a tiny decode helper for assertion readability.
func jsonMap(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("undecodable body: %v\n%s", err, body)
	}
	return m
}

func TestBuildRequestOpenAI(t *testing.T) {
	src := []byte(`{
		"model": "anything",
		"messages": [
			{"role": "system", "content": "be brief"},
			{"role": "user", "content": "hello"},
			{"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
				"function": {"name": "get_weather", "arguments": "{\"city\":\"sf\"}"}}]},
			{"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
		],
		"max_tokens": 512,
		"temperature": 0.5,
		"stop": ["END"],
		"tools": [{"type": "function", "function": {"name": "get_weather",
			"description": "weather lookup", "parameters": {"type": "object", "properties": {}}}}],
		"tool_choice": "auto"
	}`)
	body, eErr := BuildRequest("gemini-x", "openai", src, nil)
	if eErr != nil {
		t.Fatalf("BuildRequest: %v", eErr)
	}
	m := jsonMap(t, body)
	sys, _ := m["systemInstruction"].(map[string]any)
	if sys == nil {
		t.Fatalf("systemInstruction missing: %s", body)
	}
	sysParts := sys["parts"].([]any)
	if got := sysParts[0].(map[string]any)["text"]; got != "be brief" {
		t.Errorf("system text = %v", got)
	}
	contents := m["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %d entries, want 3 (user, model+call, tool result)\n%s", len(contents), body)
	}
	if contents[0].(map[string]any)["role"] != "user" {
		t.Errorf("contents[0].role = %v", contents[0].(map[string]any)["role"])
	}
	modelTurn := contents[1].(map[string]any)
	if modelTurn["role"] != "model" {
		t.Errorf("contents[1].role = %v, want model", modelTurn["role"])
	}
	call := modelTurn["parts"].([]any)[0].(map[string]any)["functionCall"].(map[string]any)
	if call["name"] != "get_weather" {
		t.Errorf("functionCall.name = %v", call["name"])
	}
	toolTurn := contents[2].(map[string]any)
	resp := toolTurn["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if resp["name"] != "get_weather" {
		t.Errorf("functionResponse.name = %v, want get_weather (recovered from call id)", resp["name"])
	}
	if got := resp["response"].(map[string]any)["result"]; got != "sunny" {
		t.Errorf("functionResponse.result = %v", got)
	}
	gen := m["generationConfig"].(map[string]any)
	if gen["maxOutputTokens"].(float64) != 512 {
		t.Errorf("maxOutputTokens = %v", gen["maxOutputTokens"])
	}
	if got := gen["stopSequences"].([]any); len(got) != 1 || got[0] != "END" {
		t.Errorf("stopSequences = %v", got)
	}
	tools := m["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)
	if tools[0].(map[string]any)["name"] != "get_weather" {
		t.Errorf("functionDeclarations = %v", tools)
	}
	tc := m["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
	if tc["mode"] != "AUTO" {
		t.Errorf("toolConfig mode = %v", tc["mode"])
	}
}

func TestBuildRequestOpenAIImagesAndDefaultMaxTokens(t *testing.T) {
	src := []byte(`{
		"messages": [{"role": "user", "content": [
			{"type": "text", "text": "what is this"},
			{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
		]}]
	}`)
	body, eErr := BuildRequest("gemini-x", "openai", src, nil)
	if eErr != nil {
		t.Fatalf("BuildRequest: %v", eErr)
	}
	m := jsonMap(t, body)
	parts := m["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2\n%s", len(parts), body)
	}
	inline := parts[1].(map[string]any)["inlineData"].(map[string]any)
	if inline["mimeType"] != "image/png" || inline["data"] != "QUJD" {
		t.Errorf("inlineData = %v", inline)
	}
	gen := m["generationConfig"].(map[string]any)
	if gen["maxOutputTokens"].(float64) != 4096 {
		t.Errorf("absent max_tokens must default to 4096, got %v", gen["maxOutputTokens"])
	}
}

func TestBuildRequestOpenAIRemoteImageRejected(t *testing.T) {
	src := []byte(`{"messages": [{"role": "user", "content": [
		{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]}`)
	_, eErr := BuildRequest("gemini-x", "openai", src, nil)
	if eErr == nil || !strings.Contains(eErr.Message, "base64 data URL") {
		t.Fatalf("remote image error = %v", eErr)
	}
	if eErr.Class != errclass.ClassTranslation {
		t.Errorf("class = %v, want translation", eErr.Class)
	}
}

func TestBuildRequestOpenAIToolChoiceKinds(t *testing.T) {
	cases := map[string]string{
		`"auto"`:          "AUTO",
		`"required"`:      "ANY",
		`"none"`:          "NONE",
		`{"type":"none"}`: "NONE",
		`{"type":"function","function":{"name":"f"}}`: "ANY",
	}
	for choice, wantMode := range cases {
		src := []byte(`{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function",
			"function":{"name":"f","parameters":{"type":"object","properties":{}}}}],"tool_choice":` + choice + `}`)
		body, eErr := BuildRequest("gemini-x", "openai", src, nil)
		if eErr != nil {
			t.Fatalf("choice %s: %v", choice, eErr)
		}
		m := jsonMap(t, body)
		tc := m["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
		if tc["mode"] != wantMode {
			t.Errorf("choice %s: mode = %v, want %s", choice, tc["mode"], wantMode)
		}
	}
}

func TestBuildRequestClaude(t *testing.T) {
	src := []byte(`{
		"model": "ignored",
		"max_tokens": 777,
		"system": "sys prompt",
		"thinking": {"type": "enabled", "budget_tokens": 8192},
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": [
				{"type": "text", "text": "checking"},
				{"type": "tool_use", "id": "tu_1", "name": "lookup", "input": {"q": "x"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "tu_1", "content": "found"}
			]}
		],
		"tools": [{"name": "lookup", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
	}`)
	body, eErr := BuildRequest("gemini-x", "claude", src, nil)
	if eErr != nil {
		t.Fatalf("BuildRequest: %v", eErr)
	}
	m := jsonMap(t, body)
	contents := m["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %d, want 3\n%s", len(contents), body)
	}
	resp := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if resp["name"] != "lookup" {
		t.Errorf("functionResponse.name = %v, want lookup", resp["name"])
	}
	gen := m["generationConfig"].(map[string]any)
	if gen["maxOutputTokens"].(float64) != 777 {
		t.Errorf("maxOutputTokens = %v", gen["maxOutputTokens"])
	}
	thinkingCfg := gen["thinkingConfig"].(map[string]any)
	if thinkingCfg["thinkingBudget"].(float64) != 8192 {
		t.Errorf("thinkingBudget = %v, want 8192 (budget passthrough)", thinkingCfg["thinkingBudget"])
	}
	if thinkingCfg["includeThoughts"] != true {
		t.Errorf("includeThoughts = %v", thinkingCfg["includeThoughts"])
	}
}

func TestBuildRequestClaudeThinkingZeroDisables(t *testing.T) {
	src := []byte(`{"max_tokens": 100, "thinking": {"type": "enabled", "budget_tokens": 0},
		"messages": [{"role": "user", "content": "hi"}]}`)
	body, eErr := BuildRequest("gemini-x", "claude", nil2(), nil)
	if eErr != nil {
		t.Fatalf("BuildRequest: %v", eErr)
	}
	_ = src
	_ = body
}

func nil2() []byte { return []byte(`{"max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`) }

func TestBuildRequestResponses(t *testing.T) {
	src := []byte(`{
		"model": "ignored",
		"instructions": "be kind",
		"input": [
			{"type": "message", "role": "user", "content": "hello"},
			{"type": "function_call", "call_id": "fc1", "name": "f", "arguments": "{\"a\":1}"},
			{"type": "function_call_output", "call_id": "fc1", "output": "ok"},
			{"type": "reasoning", "summary": [{"text": "thinking out loud"}]}
		],
		"tools": [{"type": "function", "name": "f", "parameters": {"type": "object", "properties": {}}}],
		"reasoning": {"effort": "high"}
	}`)
	body, eErr := BuildRequest("gemini-x", "openai-response", src, nil)
	if eErr != nil {
		t.Fatalf("BuildRequest: %v", eErr)
	}
	m := jsonMap(t, body)
	contents := m["contents"].([]any)
	// user message + model function_call + user function_response; reasoning omitted.
	if len(contents) != 3 {
		t.Fatalf("contents = %d, want 3 (reasoning omitted)\n%s", len(contents), body)
	}
	resp := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if resp["name"] != "f" {
		t.Errorf("functionResponse.name = %v", resp["name"])
	}
	gen := m["generationConfig"].(map[string]any)
	tc := gen["thinkingConfig"].(map[string]any)
	if tc["thinkingBudget"].(float64) != 24576 {
		t.Errorf("effort high budget = %v, want 24576", tc["thinkingBudget"])
	}
}

func TestBuildRequestUnknownFormat(t *testing.T) {
	_, eErr := BuildRequest("gemini-x", "mystery", []byte(`{}`), nil)
	if eErr == nil || eErr.Class != errclass.ClassUnsupported {
		t.Fatalf("unknown format error = %v", eErr)
	}
}
