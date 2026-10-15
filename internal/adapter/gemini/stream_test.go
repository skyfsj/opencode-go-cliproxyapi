package gemini

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// feedAll pushes byte-split chunks through the converter exactly as the
// executor's pump would, then Flushes the close-without-terminal residual.
func feedAll(t *testing.T, sc *StreamConverter, chunks []string) [][]byte {
	t.Helper()
	var events [][]byte
	for _, c := range chunks {
		ev, _, eErr := sc.Feed([]byte(c))
		if eErr != nil {
			t.Fatalf("Feed: %v", eErr)
		}
		events = append(events, ev...)
	}
	events = append(events, sc.Flush()...)
	return events
}

func TestStreamTextOpenAI(t *testing.T) {
	sc := NewStreamConverter("openai")
	frames := []string{
		`data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}],"modelVersion":"gemini-test-1"}`,
		`data: {"candidates":[{"content":{"parts":[{"text":"lo!"},{"thought":true,"text":"hm"}]}}],"modelVersion":"gemini-test-1"}`,
		`data: {"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":3,"thoughtsTokenCount":5},"modelVersion":"gemini-test-1"}`,
		`data: {"candidates":[],"modelVersion":"gemini-test-1"}`,
	}
	// split bytes at awkward boundaries to exercise framer buffering
	chunks := []string{
		frames[0] + "\n\n" + frames[1],
		"\n\ndata: {\"candi", // partial frame
	}
	chunks = append(chunks, "dates\":[],\"modelVersion\":\"gemini-test-1\"}\n\n")
	chunks = append(chunks, frames[2]+"\n\n")

	events := feedAll(t, sc, chunks)

	var buf []byte
	for _, e := range events {
		buf = append(buf, e...)
	}
	var chunksOut []struct {
		Choices []struct {
			Delta        json.RawMessage `json:"delta"`
			FinishReason any             `json:"finish_reason"`
		} `json:"choices"`
	}
	dec := json.NewDecoder(bytes.NewReader(buf))
	for {
		var c struct {
			Choices []struct {
				Delta        json.RawMessage `json:"delta"`
				FinishReason any             `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := dec.Decode(&c); err != nil {
			break
		}
		chunksOut = append(chunksOut, c)
	}
	// role chunk + 2 text deltas (thought dropped) + finish chunk
	if len(chunksOut) != 4 {
		t.Fatalf("emitted %d chunks, want 4\n%s", len(chunksOut), string(buf))
	}
	var roleDelta struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(chunksOut[0].Choices[0].Delta, &roleDelta); err != nil {
		t.Fatal(err)
	}
	if roleDelta.Role != "assistant" {
		t.Errorf("first chunk delta = %v", chunksOut[0].Choices[0].Delta)
	}
	finishChunk := chunksOut[3]
	if finishChunk.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %v", finishChunk.Choices[0].FinishReason)
	}
	// finish chunk carries the usage from the terminal frame (top-level
	// "usage" member of chat.completion.chunk); decode each chunk into a
	// map so no key is dropped by re-marshaling.
	var lastChunk map[string]any
	decAll := json.NewDecoder(bytes.NewReader(buf))
	for {
		var m map[string]any
		if err := decAll.Decode(&m); err != nil {
			break
		}
		lastChunk = m
	}
	if lastChunk == nil {
		t.Fatal("no chunks decoded")
	}
	usage, _ := lastChunk["usage"].(map[string]any)
	if usage == nil || usage["prompt_tokens"].(float64) != 12 {
		t.Errorf("terminal usage = %v, want prompt_tokens 12", usage)
	}
	if usage != nil && usage["completion_tokens"].(float64) != 8 { // 3 + 5 thinking
		t.Errorf("terminal completion tokens = %v, want 8", usage["completion_tokens"])
	}
}

func TestStreamToolCallOpenAI(t *testing.T) {
	sc := NewStreamConverter("openai")
	events := feedAll(t, sc, []string{
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"sf\"}}}]}}],\"modelVersion\":\"gemini-test-1\"}\n\n",
		"data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":8,\"candidatesTokenCount\":4},\"modelVersion\":\"gemini-test-1\"}\n\n",
	})
	if len(events) != 3 {
		t.Fatalf("events = %d, want role + tool_call + finish\n%s", len(events), events)
	}
	var tcEntry struct {
		Choices []struct {
			Delta struct {
				ToolCalls []struct {
					Index    int    `json:"index"`
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(events[1], &tcEntry); err != nil {
		t.Fatalf("tool chunk: %v", err)
	}
	tc := tcEntry.Choices[0].Delta.ToolCalls[0]
	if tc.ID != "get_weather#0" || tc.Type != "function" || tc.Function.Name != "get_weather" {
		t.Fatalf("tool call entry = %+v", tc)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatalf("arguments not JSON: %v", args)
	}
	if args["city"] != "sf" {
		t.Errorf("arguments = %v", args)
	}
	var finish struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage map[string]any `json:"usage"`
	}
	if err := json.Unmarshal(events[2], &finish); err != nil {
		t.Fatal(err)
	}
	if finish.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish = %v", finish.Choices[0].FinishReason)
	}
	if finish.Usage == nil || finish.Usage["prompt_tokens"].(float64) != 8 {
		t.Errorf("usage missing or wrong: %v", finish.Usage)
	}
}

func TestStreamClaude(t *testing.T) {
	sc := NewStreamConverter("claude")
	events := feedAll(t, sc, []string{
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}],\"modelVersion\":\"gemini-test-1\"}\n\n",
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo!\"}]}}],\"modelVersion\":\"gemini-test-1\"}\n\n",
		"data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":12,\"candidatesTokenCount\":5},\"modelVersion\":\"gemini-test-1\"}\n\n",
	})
	// message_start, text block start, 2 deltas, block stop, message_delta, message_stop
	if len(events) != 7 {
		t.Fatalf("events = %d, want 7\n%s", len(events), events)
	}
	if !strings.Contains(string(events[0]), `"type":"message_start"`) &&
		!strings.Contains(string(events[0]), `"type": "message_start"`) {
		t.Errorf("first event = %s", events[0])
	}
	joined := make([]string, len(events))
	for i, e := range events {
		joined[i] = string(e)
	}
	if !strings.Contains(joined[len(joined)-2], `"stop_reason":"end_turn"`) &&
		!strings.Contains(joined[len(joined)-2], `"stop_reason": "end_turn"`) {
		t.Errorf("message_delta = %s", joined[len(joined)-2])
	}
	if !strings.Contains(joined[len(joined)-2], `"output_tokens":5`) {
		t.Errorf("terminal usage missing output tokens: %s", joined[len(joined)-2])
	}
	if !strings.Contains(joined[len(joined)-1], `"type":"message_stop"`) &&
		!strings.Contains(joined[len(joined)-1], `"type": "message_stop"`) {
		t.Errorf("last event = %s", joined[len(joined)-1])
	}
}

func TestStreamClaudeToolCall(t *testing.T) {
	sc := NewStreamConverter("claude")
	// tool-first stream: no phantom text block
	events := feedAll(t, sc, []string{
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"f\",\"args\":{\"x\":1}}}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2},\"modelVersion\":\"gemini-test-1\"}\n\n",
	})
	joined := ""
	for _, e := range events {
		joined += string(e)
	}
	if strings.Contains(joined, `"type": "text"`) || strings.Contains(joined, `"type":"text"`) {
		t.Errorf("tool-only stream opened a phantom text block:\n%s", joined)
	}
	if !strings.Contains(joined, "input_json_delta") {
		t.Errorf("no input_json_delta:\n%s", joined)
	}
	if !strings.Contains(joined, `"stop_reason":"tool_use"`) && !strings.Contains(joined, `"stop_reason": "tool_use"`) {
		t.Errorf("stop_reason = not tool_use:\n%s", joined)
	}
}

func TestStreamResponses(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	events := feedAll(t, sc, []string{
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}],\"modelVersion\":\"gemini-test-1\"}\n\n",
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]}}],\"modelVersion\":\"gemini-test-1\"}\n\n",
		"data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":12,\"candidatesTokenCount\":5},\"modelVersion\":\"gemini-test-1\"}\n\n",
	})
	var signallers []string
	for _, e := range events {
		payload, _, _ := strings.Cut(string(e), "\ndata: ")
		_, data, _ := strings.Cut(string(e), "data: ")
		_ = payload
		var probe map[string]any
		if err := json.Unmarshal([]byte(data), &probe); err != nil {
			t.Fatalf("event not SSE JSON: %v\n%s", err, e)
		}
		signallers = append(signallers, probe["type"].(string))
	}
	want := []string{
		"response.created",
		"response.output_item.added",
		"response.output_text.delta", "response.output_text.delta",
		"response.output_text.done", "response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}
	if len(signallers) != len(want) {
		t.Fatalf("event types = %v, want %v", signallers, want)
	}
	for i, w := range want {
		if signallers[i] != w {
			t.Errorf("event[%d] = %q, want %q", i, signallers[i], w)
		}
	}
	// terminal carries usage + the message output item with full text
	var completed struct {
		Response struct {
			Status string `json:"status"`
			Usage  struct {
				InputTokens  int64 `json:"input_tokens"`
				OutputTokens int64 `json:"output_tokens"`
			} `json:"usage"`
			Output []struct {
				Type string `json:"type"`
			} `json:"output"`
		} `json:"response"`
	}
	_, termData, _ := strings.Cut(string(events[len(events)-1]), "data: ")
	if err := json.Unmarshal([]byte(termData), &completed); err != nil {
		t.Fatal(err)
	}
	if completed.Response.Status != "completed" {
		t.Errorf("status = %q", completed.Response.Status)
	}
	if completed.Response.Usage.InputTokens != 12 || completed.Response.Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v", completed.Response.Usage)
	}
	if len(completed.Response.Output) != 1 || completed.Response.Output[0].Type != "message" {
		t.Errorf("output = %+v", completed.Response.Output)
	}
}

func TestStreamPumpIntegrity(t *testing.T) {
	// Split the whole SSE stream at single-byte offsets in the first frame
	// to prove no frame is lost or duplicated across Feed boundaries.
	text := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"AB\"}]}}],\"modelVersion\":\"gemini-test-1\"}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":9,\"candidatesTokenCount\":2},\"modelVersion\":\"gemini-test-1\"}\n\n"
	for split := 1; split < 40; split++ {
		sc := NewStreamConverter("openai")
		var buf []byte
		ev1, _, eErr := sc.Feed([]byte(text[:split]))
		if eErr != nil {
			t.Fatalf("split %d: %v", split, eErr)
		}
		ev2, _, eErr := sc.Feed([]byte(text[split:]))
		if eErr != nil {
			t.Fatalf("split %d: %v", split, eErr)
		}
		ev1 = append(ev1, ev2...)
		ev1 = append(ev1, sc.Flush()...)
		for _, e := range ev1 {
			buf = append(buf, e...)
		}
		var chunks []map[string]any
		dec := json.NewDecoder(strings.NewReader(string(buf)))
		for {
			var m map[string]any
			if err := dec.Decode(&m); err != nil {
				break
			}
			chunks = append(chunks, m)
		}
		if len(chunks) != 3 {
			t.Errorf("split %d: got %d chunks, want 3\n%s", split, len(chunks), string(buf))
		}
	}
}

func TestStreamInFrameError(t *testing.T) {
	sc := NewStreamConverter("openai")
	_, _, eErr := sc.Feed([]byte("data: {\"error\":{\"code\":429,\"message\":\"quota exceeded\",\"status\":\"RESOURCE_EXHAUSTED\"}}\n\n"))
	if eErr == nil {
		t.Fatal("in-frame error not surfaced")
	}
}

func TestStreamMalformedFrame(t *testing.T) {
	sc := NewStreamConverter("openai")
	_, _, eErr := sc.Feed([]byte("data: {\"candidates\":\n\n"))
	if eErr == nil || !strings.Contains(eErr.Message, "malformed generateContent stream frame") {
		t.Fatalf("malformed frame error = %v", eErr)
	}
}
