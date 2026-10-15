package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	"opencode-go-cliproxyapi/internal/errclass"
)

func gemResultBody() []byte {
	return []byte(`{
		"candidates": [{
			"content": {"parts": [
				{"text": "Answer part one. "},
				{"thought": true, "text": "secret reasoning"},
				{"functionCall": {"name": "get_weather", "args": {"city": "sf"}}}
			]},
			"finishReason": "STOP"
		}],
		"usageMetadata": {
			"promptTokenCount": 100,
			"candidatesTokenCount": 50,
			"thoughtsTokenCount": 20,
			"cachedContentTokenCount": 10
		},
		"modelVersion": "gemini-test-1"
	}`)
}

func TestConvertToChat(t *testing.T) {
	body, eErr := ConvertNonStreamResponse("openai", 200, gemResultBody())
	if eErr != nil {
		t.Fatalf("ConvertNonStreamResponse: %v", eErr)
	}
	m := jsonMap(t, body)
	if m["model"] != "gemini-test-1" {
		t.Errorf("model = %v", m["model"])
	}
	choice := m["choices"].([]any)[0].(map[string]any)
	// Observed tool calls outrank the status-derived reason (shared
	// TerminalReason semantics, FR-006).
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v, want tool_calls", choice["finish_reason"])
	}
	msg := choice["message"].(map[string]any)
	if got := msg["content"]; got != "Answer part one. " {
		t.Errorf("content = %v, thought text must be omitted", got)
	}
	calls := msg["tool_calls"].([]any)
	tc := calls[0].(map[string]any)
	if tc["id"] != "get_weather#0" || tc["type"] != "function" {
		t.Errorf("tool_call = %v", tc)
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("function.name = %v", fn["name"])
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(fn["arguments"].(string)), &args); err != nil {
		t.Fatalf("arguments not JSON: %v", err)
	}
	if args["city"] != "sf" {
		t.Errorf("arguments = %v", args)
	}
	usage := m["usage"].(map[string]any)
	if usage["prompt_tokens"].(float64) != 100 {
		t.Errorf("prompt_tokens = %v", usage["prompt_tokens"])
	}
	// output span includes thinking tokens (50 + 20).
	if usage["completion_tokens"].(float64) != 70 {
		t.Errorf("completion_tokens = %v, want 70", usage["completion_tokens"])
	}
	if usage["total_tokens"].(float64) != 170 {
		t.Errorf("total_tokens = %v, want 170", usage["total_tokens"])
	}
}

func TestConvertToClaude(t *testing.T) {
	body, eErr := ConvertNonStreamResponse("claude", 200, gemResultBody())
	if eErr != nil {
		t.Fatalf("ConvertNonStreamResponse: %v", eErr)
	}
	m := jsonMap(t, body)
	if m["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use (tool calls observed)", m["stop_reason"])
	}
	content := m["content"].([]any)
	// One text block (thought omitted) + one tool_use block.
	if len(content) != 2 {
		t.Fatalf("content blocks = %d, want 2\n%s", len(content), body)
	}
	if content[0].(map[string]any)["type"] != "text" {
		t.Errorf("block[0] = %v", content[0])
	}
	tu := content[1].(map[string]any)
	if tu["type"] != "tool_use" || tu["name"] != "get_weather" || tu["id"] != "get_weather#0" {
		t.Errorf("tool_use block = %v", tu)
	}
	usage := m["usage"].(map[string]any)
	// cache-read tokens subtract from input; thinking adds to output.
	if usage["input_tokens"].(float64) != 90 {
		t.Errorf("input_tokens = %v, want 90", usage["input_tokens"])
	}
	if usage["output_tokens"].(float64) != 70 {
		t.Errorf("output_tokens = %v, want 70", usage["output_tokens"])
	}
	if usage["cache_read_input_tokens"].(float64) != 10 {
		t.Errorf("cache_read_input_tokens = %v", usage["cache_read_input_tokens"])
	}
}

func TestConvertToResponses(t *testing.T) {
	body, eErr := ConvertNonStreamResponse("openai-response", 200, gemResultBody())
	if eErr != nil {
		t.Fatalf("ConvertNonStreamResponse: %v", eErr)
	}
	m := jsonMap(t, body)
	if m["status"] != "completed" {
		t.Errorf("status = %v", m["status"])
	}
	output := m["output"].([]any)
	// CC-route placement: the message leads, then the function_call item.
	if len(output) != 2 {
		t.Fatalf("output = %d items, want 2\n%s", len(output), body)
	}
	msg := output[0].(map[string]any)
	if msg["type"] != "message" {
		t.Errorf("output[0] = %v", msg)
	}
	call := output[1].(map[string]any)
	if call["type"] != "function_call" || call["name"] != "get_weather" {
		t.Errorf("output[1] = %v", call)
	}
	usage := m["usage"].(map[string]any)
	if usage["output_tokens"].(float64) != 70 {
		t.Errorf("output_tokens = %v, want 70", usage["output_tokens"])
	}
}

func TestConvertMaxTokensTruncation(t *testing.T) {
	body := []byte(`{
		"candidates": [{"content": {"parts": [{"text": "cut off"}]}, "finishReason": "MAX_TOKENS"}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
		"modelVersion": "gemini-test-1"
	}`)
	chat, eErr := ConvertNonStreamResponse("openai", 200, body)
	if eErr != nil {
		t.Fatalf("openai: %v", eErr)
	}
	m := jsonMap(t, chat)
	if m["choices"].([]any)[0].(map[string]any)["finish_reason"] != "length" {
		t.Errorf("finish_reason = %v, want length", m)
	}
	claude, eErr := ConvertNonStreamResponse("claude", 200, body)
	if eErr != nil {
		t.Fatalf("claude: %v", eErr)
	}
	if jsonMap(t, claude)["stop_reason"] != "max_tokens" {
		t.Errorf("stop_reason = %v, want max_tokens", claude)
	}
	resp, eErr := ConvertNonStreamResponse("openai-response", 200, body)
	if eErr != nil {
		t.Fatalf("responses: %v", eErr)
	}
	if jsonMap(t, resp)["status"] != "incomplete" {
		t.Errorf("status = %v, want incomplete", resp)
	}
}

func TestConvertSafetyRefusal(t *testing.T) {
	body := []byte(`{
		"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}],
		"modelVersion": "gemini-test-1"
	}`)
	claude, eErr := ConvertNonStreamResponse("claude", 200, body)
	if eErr != nil {
		t.Fatalf("claude: %v", eErr)
	}
	if jsonMap(t, claude)["stop_reason"] != "refusal" {
		t.Errorf("SAFETY stop_reason = %v, want refusal", claude)
	}
}

func TestConvertNoCandidatesBlocked(t *testing.T) {
	body := []byte(`{"promptFeedback": {"blockReason": "SAFETY"}}`)
	_, eErr := ConvertNonStreamResponse("openai", 200, body)
	if eErr == nil || !strings.Contains(eErr.Message, "blocked: SAFETY") {
		t.Fatalf("blocked error = %v", eErr)
	}
}

func TestConvertUnknownFormat(t *testing.T) {
	_, eErr := ConvertNonStreamResponse("mystery", 200, []byte(`{}`))
	if eErr == nil || eErr.Class != errclass.ClassUnsupported {
		t.Fatalf("unknown format error = %v", eErr)
	}
}
