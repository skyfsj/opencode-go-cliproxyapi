// Non-stream response conversion for the Gemini adapter (FR-006): an
// upstream generateContent HTTP result is translated back into the client
// protocol — OpenAI Chat Completions / Anthropic Messages / OpenAI
// Responses synthesis. Thinking ("thought") parts are explicitly omitted
// per the FR-005 omission policy.
package gemini

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"opencode-go-cliproxyapi/internal/adapter/shared"
	"opencode-go-cliproxyapi/internal/errclass"
)

// ---- upstream generateContent result shapes (decode-only) ----

type gemResponsePart struct {
	Text         string             `json:"text"`
	Thought      bool               `json:"thought"`
	FunctionCall *gemFunctionCallIn `json:"functionCall"`
}

type gemFunctionCallIn struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type gemContent struct {
	Parts []gemResponsePart `json:"parts"`
}

type gemCandidate struct {
	Content      gemContent `json:"content"`
	FinishReason string     `json:"finishReason"`
}

type gemUsage struct {
	PromptTokenCount        int64  `json:"promptTokenCount"`
	CandidatesTokenCount    int64  `json:"candidatesTokenCount"`
	CachedContentTokenCount *int64 `json:"cachedContentTokenCount"`
	ThoughtsTokenCount      *int64 `json:"thoughtsTokenCount"`
}

type gemResultIn struct {
	Candidates     []gemCandidate `json:"candidates"`
	UsageMetadata  *gemUsage      `json:"usageMetadata"`
	ModelVersion   string         `json:"modelVersion"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
}

// ConvertNonStreamResponse translates an upstream generateContent result
// back into the client protocol sourceFormat (FR-006). Statuses >= 400
// become classified redacted errors (§7); "openai", "claude", and
// "openai-response" are synthesized. Unknown formats are
// ClassUnsupported; malformed upstream bodies are ClassTranslation.
func ConvertNonStreamResponse(sourceFormat string, status int, upstreamBody []byte, tools ...*shared.ResponseTools) ([]byte, *errclass.Error) {
	if status >= 400 {
		return nil, shared.UpstreamStatusError(status, upstreamBody)
	}
	switch sourceFormat {
	case "openai":
		return toChat(upstreamBody)
	case "claude":
		return toClaude(upstreamBody)
	case "openai-response":
		return toResponses(upstreamBody, tools...)
	default:
		return nil, shared.UnsupportedFormat(sourceFormat, targetLabel)
	}
}

// decodeResult validates and decodes an upstream non-stream generateContent
// body. Candidates exhaustively EMPTY (no candidates at all) is a
// translation failure naming the prompt-block reason when one exists.
func decodeResult(body []byte) (*gemResultIn, *errclass.Error) {
	var resp gemResultIn
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, errclass.Translation("malformed generateContent response JSON: " + err.Error())
	}
	if len(resp.Candidates) == 0 {
		reason := ""
		if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
			reason = ` (blocked: ` + resp.PromptFeedback.BlockReason + `)`
		}
		return nil, errclass.Translation("generateContent response carries no candidates" + reason)
	}
	return &resp, nil
}

// gemCallIDs synthesizes client-facing tool-call ids for the response's
// functionCall parts. The Gemini wire carries no call ids, so ids encode
// the tool name and response position ("<name>#<n>"); request translation
// resolves tool names through the history's tool declarations, never by
// parsing ids.
func gemCallID(name string, ord int) string {
	return name + "#" + strconv.Itoa(ord)
}

// finishStatus classifies a Gemini finishReason into the §7 vocabulary:
// token truncation and disgorged-content refusals are distinguished, every
// other reason (including unknown values) completes normally.
func finishStatus(finish string) (cc string, claude string, resp string) {
	switch finish {
	case "MAX_TOKENS":
		return "length", "max_tokens", "incomplete"
	case "SAFETY", "RECITATION", "PROHIBITED_CONTENT", "SPII", "BLOCKLIST":
		return "content_filter", "refusal", "completed"
	default: // STOP, unspecified, unknown
		return "stop", "end_turn", "completed"
	}
}

// tokenDetails projects the usage counters per the FR-005/FR-006 policies:
// cached tokens ride as input detail, thinking tokens as output detail;
// total candidates/thoughts counts stay authoritative for sums (the
// majority-sum kernels are fed the raw counts directly).
func (u *gemUsage) details() shared.UsageDetails {
	var d shared.UsageDetails
	if u != nil {
		d.CachedTokens = u.CachedContentTokenCount
		d.ReasoningTokens = u.ThoughtsTokenCount
	}
	return d
}

func (u *gemUsage) counts() (output int64) {
	if u == nil {
		return 0
	}
	// Gemini reports thinking tokens separately from candidate tokens;
	// every client vocabulary counts reasoning inside the output span.
	return u.CandidatesTokenCount + valueOrZero(u.ThoughtsTokenCount)
}

func valueOrZero(v *int64) int64 {
	if v != nil {
		return *v
	}
	return 0
}

// observed collects the translatable content of one candidate: text plus
// tool calls, in part order, with thinking parts dropped (FR-005).
type observed struct {
	text     strings.Builder
	calls    []gemFunctionCallIn
	thoughts strings.Builder
}

func observe(c gemContent) observed {
	var o observed
	for _, p := range c.Parts {
		switch {
		case p.FunctionCall != nil:
			o.calls = append(o.calls, *p.FunctionCall)
		case p.Thought:
			o.thoughts.WriteString(p.Text)
		default:
			o.text.WriteString(p.Text)
		}
	}
	return o
}

// ---- Gemini -> openai (Chat Completions) (FR-006) ----

func toChat(body []byte) ([]byte, *errclass.Error) {
	resp, eErr := decodeResult(body)
	if eErr != nil {
		return nil, eErr
	}
	cand := resp.Candidates[0]
	o := observe(cand.Content)
	msg := shared.CCMessage{Role: "assistant"}
	if o.text.Len() > 0 {
		msg.Content, _ = json.Marshal(o.text.String())
	}
	for i, call := range o.calls {
		tc := shared.CCToolCall{ID: gemCallID(call.Name, i), Type: "function"}
		tc.Function.Name = call.Name
		tc.Function.Arguments = shared.DefaultArgs(string(call.Args))
		msg.ToolCalls = append(msg.ToolCalls, tc)
	}
	ccFinish, _, _ := finishStatus(cand.FinishReason)
	finish := shared.TerminalReason(len(o.calls) > 0, "tool_calls", ccFinish)
	var inputTokens int64
	if resp.UsageMetadata != nil {
		inputTokens = resp.UsageMetadata.PromptTokenCount
	}
	var completionTokens int64
	var details shared.UsageDetails
	if resp.UsageMetadata != nil {
		completionTokens = resp.UsageMetadata.counts()
		details = resp.UsageMetadata.details()
	}
	out := shared.CompletionEnvelope("", modelOrID(resp), time.Now().Unix(),
		[]map[string]any{{"index": 0, "message": msg, "finish_reason": finish}},
		shared.CCUsageFrom(inputTokens, completionTokens, details))
	b, _ := json.Marshal(out)
	return b, nil
}

func modelOrID(resp *gemResultIn) string {
	return resp.ModelVersion
}

// ---- Gemini -> claude (Anthropic Messages) (FR-006) ----

func toClaude(body []byte) ([]byte, *errclass.Error) {
	resp, eErr := decodeResult(body)
	if eErr != nil {
		return nil, eErr
	}
	cand := resp.Candidates[0]
	o := observe(cand.Content)
	var blocks []map[string]any
	if t := o.text.String(); t != "" {
		blocks = append(blocks, map[string]any{"type": "text", "text": t})
	}
	for i, call := range o.calls {
		input, eErr := shared.DecodeArgs(string(call.Args))
		if eErr != nil {
			return nil, eErr
		}
		blocks = append(blocks, map[string]any{
			"type": "tool_use", "id": gemCallID(call.Name, i), "name": call.Name, "input": input,
		})
	}
	_, claudeStop, _ := finishStatus(cand.FinishReason)
	stop := shared.TerminalReason(len(o.calls) > 0, "tool_use", claudeStop)
	var promptTokens int64
	var cachedRead *int64
	if resp.UsageMetadata != nil {
		promptTokens = resp.UsageMetadata.PromptTokenCount
		cachedRead = resp.UsageMetadata.CachedContentTokenCount
	}
	input := shared.ClampSubtract(promptTokens, cachedRead)
	var outputTokens int64
	if resp.UsageMetadata != nil {
		outputTokens = resp.UsageMetadata.counts()
	}
	b, _ := json.Marshal(shared.NewClaudeResult(resp.ModelVersion, resp.ModelVersion, stop,
		input, outputTokens, cachedRead, nil, blocks))
	return b, nil
}

// ---- Gemini -> openai-response (Responses) (FR-006) ----

func toResponses(body []byte, tools ...*shared.ResponseTools) ([]byte, *errclass.Error) {
	resp, eErr := decodeResult(body)
	if eErr != nil {
		return nil, eErr
	}
	cand := resp.Candidates[0]
	o := observe(cand.Content)
	_, _, respStatus := finishStatus(cand.FinishReason)
	out := shared.ResponsesResult{
		ID:     resp.ModelVersion,
		Object: "response",
		Model:  resp.ModelVersion,
		Status: respStatus,
		Output: []any{},
	}
	// CC-route placement parity: the message leads and appears only when
	// text is non-empty, matching chatToResponses (FR-006).
	oa := shared.NewOutputAssembler(resp.ModelVersion, tools...)
	oa.AddText(o.text.String())
	for i, call := range o.calls {
		oa.AppendFunctionCall(gemCallID(call.Name, i), call.Name, shared.DefaultArgs(string(call.Args)))
	}
	out.Output = oa.Render()
	if resp.UsageMetadata != nil {
		out.Usage = shared.NewResponsesUsageFrom(resp.UsageMetadata.PromptTokenCount, resp.UsageMetadata.counts(), resp.UsageMetadata.details())
	}
	b, _ := json.Marshal(out)
	return b, nil
}
