// Stream conversion for the Gemini adapter (FR-006): an upstream
// streamGenerateContent SSE stream (data:-only frames, alt=sse) is
// translated incrementally into the client protocol — openai Chat
// Completions chunks, claude Messages events, openai-response Responses
// events. The converter is sequential-use only: one instance per upstream
// stream, Feed called with consecutive network chunks in order.
//
// Upstream frame shape (alt=sse): each data: line carries one
// GenerateContentResponse with candidates[0].content.parts deltas — text
// parts stream token-by-token, thought parts stream with thought:true,
// and functionCall parts arrive whole. usageMetadata rides on any chunk
// (usually the last) and the stream closes without a [DONE] sentinel.
package gemini

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"opencode-go-cliproxyapi/internal/adapter/shared"
	"opencode-go-cliproxyapi/internal/errclass"
)

// StreamConverter translates a Gemini SSE stream into sourceFormat's
// stream shape.
type StreamConverter struct {
	sourceFormat string
	framer       *shared.SSEFramer
	created      int64
	done         bool

	started   bool // first chunk seen (identity + role/message announced)
	id        string
	model     string
	toolOrd   int // next synthesized tool-call ordinal
	toolsSeen bool

	// openai target state
	roleSent   bool
	heldFinish string // deferred terminal finish; flushed on a later frame or at close

	// claude target state
	claudeEm   *shared.ClaudeEventEmitter
	textOpen   bool
	textIndex  int
	nextIndex  int
	tools      map[int]*claudeToolState // by content-block index
	toolOrder  []int
	finishSeen bool // any finishReason observed (terminal bookkeeping)

	// responses target state
	msgIndex  int
	outCount  int
	respText  strings.Builder
	respTools *shared.ResponseTools
	respCalls []respCallItem

	// shared observation
	promptTokens int64
	usage        *gemUsage
}

// claudeToolState tracks one open tool_use block on the claude target.
type claudeToolState struct {
	blockIndex int
	id         string
	name       string
	args       strings.Builder
	stopped    bool
}

// respCallItem accumulates one function_call item on the responses target.
type respCallItem struct {
	outIdx int
	id     string
	name   string
	args   strings.Builder
}

// NewStreamConverter prepares stream conversion for sourceFormat ("openai"
// Chat Completions chunks, "claude" Messages events, "openai-response"
// Responses events).
func NewStreamConverter(sourceFormat string, tools ...*shared.ResponseTools) *StreamConverter {
	sc := &StreamConverter{
		sourceFormat: sourceFormat,
		framer:       shared.NewSSEFramer(false),
		created:      time.Now().Unix(),
		msgIndex:     -1,
		tools:        map[int]*claudeToolState{},
	}
	if len(tools) > 0 && tools[0] != nil {
		sc.respTools = tools[0]
	}
	return sc
}

// Feed consumes one network chunk and returns the client events completed
// by this chunk, whether the stream is finished, and a classified error
// for malformed chunks (FR-006). Errors carry at most a redacted snippet.
//
// Gemini streams carry no [DONE] sentinel — the upstream simply closes —
// so done only turns true when a FINISH frame already rendered a complete
// client terminal inline (claude target) or when the terminal was deferred
// and a subsequent frame confirmed nothing more is coming. In the common
// close-without-anything case the executor's close-without-terminal Flush
// seam renders the residual (F5).
func (sc *StreamConverter) Feed(chunk []byte) (events [][]byte, done bool, eErr *errclass.Error) {
	sc.framer.Push(chunk)
	for {
		_, data, _, ok := sc.framer.Next()
		if !ok {
			return events, false, nil
		}
		if strings.TrimSpace(data) == "" {
			continue
		}
		ev, eErr := sc.decodeFrame(data)
		if eErr != nil {
			return events, false, eErr
		}
		var frameEvents [][]byte
		frameEvents, eErr = sc.dispatch(ev)
		events = append(events, frameEvents...)
		if eErr != nil {
			return events, false, eErr
		}
		if sc.done {
			return events, true, nil
		}
	}
}

// gemStreamFrame is the typed decode of one streamGenerateContent SSE
// payload. Error stays raw: the classifier reads an open-ended object.
type gemStreamFrame struct {
	Candidates     []gemStreamCandidate `json:"candidates"`
	UsageMetadata  *gemUsage            `json:"usageMetadata"`
	ModelVersion   string               `json:"modelVersion"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	Error map[string]any `json:"error"`
}

type gemStreamCandidate struct {
	Content      gemContent `json:"content"`
	FinishReason string     `json:"finishReason"`
}

// decodeFrame parses one upstream SSE payload; malformed JSON is a
// translation failure with a redacted snippet, an in-frame error object a
// classified failure (FR-009).
func (sc *StreamConverter) decodeFrame(data string) (*gemStreamFrame, *errclass.Error) {
	var frame gemStreamFrame
	if err := json.Unmarshal([]byte(data), &frame); err != nil {
		return nil, errclass.Translation("malformed generateContent stream frame: " + shared.RedactedSnippet(data))
	}
	if frame.Error != nil {
		return nil, frameError(frame.Error)
	}
	if frame.UsageMetadata != nil {
		sc.usage = frame.UsageMetadata
	}
	if sc.model == "" && frame.ModelVersion != "" {
		sc.model = frame.ModelVersion
	}
	return &frame, nil
}

// frameError classifies an in-stream Gemini error object. Gemini error
// bodies carry {code, message, status}; a numeric code maps through
// FromStatus so §7 retryability semantics apply, otherwise the failure is
// a retryable upstream server failure.
func frameError(e map[string]any) *errclass.Error {
	message := sget(e, "message")
	status := 0
	switch c := e["code"].(type) {
	case float64:
		status = int(c)
	case string:
		// non-numeric code strings carry no status semantics
	}
	if status == 0 {
		switch strings.ToUpper(sget(e, "status")) {
		case "INVALID_ARGUMENT":
			status = 400
		case "UNAUTHENTICATED":
			status = 401
		case "PERMISSION_DENIED":
			status = 403
		case "NOT_FOUND":
			status = 404
		case "RESOURCE_EXHAUSTED":
			status = 429
		case "UNAVAILABLE":
			status = 503
		}
	}
	if status > 0 {
		return errclass.FromStatus(status, message)
	}
	return errclass.UpstreamFallback(message)
}

// dispatch routes one decoded frame into the target protocol's synthesis.
func (sc *StreamConverter) dispatch(frame *gemStreamFrame) ([][]byte, *errclass.Error) {
	switch sc.sourceFormat {
	case "openai":
		return sc.dispatchOpenAI(frame)
	case "claude":
		return sc.dispatchClaude(frame)
	case "openai-response":
		return sc.dispatchResponses(frame)
	}
	return nil, shared.UnsupportedFormat(sc.sourceFormat, targetLabel)
}

// observeCandidate extracts the translatable deltas of one candidate's
// content: text parts (non-thought) and whole functionCall parts.
func observeCandidate(c gemContent) (textParts []string, calls []gemFunctionCallIn) {
	for _, p := range c.Parts {
		switch {
		case p.FunctionCall != nil:
			calls = append(calls, *p.FunctionCall)
		case p.Thought:
			// FR-005 explicit omission policy: thought summaries have no
			// standard Chat Completions / Responses field and no
			// fabricable Messages block, so they are dropped in-stream.
		default:
			if p.Text != "" {
				textParts = append(textParts, p.Text)
			}
		}
	}
	return textParts, calls
}

// ---- Gemini -> openai (Chat Completions) stream (FR-006) ----

func (sc *StreamConverter) dispatchOpenAI(frame *gemStreamFrame) ([][]byte, *errclass.Error) {
	var events [][]byte
	if len(frame.Candidates) == 0 {
		return nil, nil // usage-only or empty trailer
	}
	cand := frame.Candidates[0]
	if !sc.started {
		sc.started = true
		sc.id = newStreamID()
		events = append(events, sc.chatChunks().RoleChunk())
		sc.roleSent = true
	}
	textParts, calls := observeCandidate(cand.Content)
	for _, text := range textParts {
		events = append(events, sc.chatChunks().Delta(map[string]any{"content": text}))
	}
	for _, call := range calls {
		sc.toolsSeen = true
		entry := shared.CCToolCallOpeningEntry(sc.toolOrd, gemCallID(call.Name, sc.toolOrd), call.Name, shared.DefaultArgs(string(call.Args)))
		sc.toolOrd++
		events = append(events, sc.chatChunks().Delta(map[string]any{"tool_calls": []any{entry}}))
	}
	if cand.FinishReason != "" && sc.heldFinish == "" {
		// Deferred like the sibling CC-route converter: the usage trailer
		// usually follows the finish frame and must land in the terminal
		// chunk; [DONE]-less streams flush it at close via Flush.
		sc.heldFinish = cand.FinishReason
	}
	return events, nil
}

// chatChunks binds the shared chunk kernel to the captured stream identity.
func (sc *StreamConverter) chatChunks() shared.ChatChunkBuilder {
	return shared.ChatChunkBuilder{ID: sc.id, Model: sc.model, Created: sc.created}
}

// openAITerminal renders the deferred finish chunk exactly once.
func (sc *StreamConverter) openAITerminal() [][]byte {
	finish := sc.heldFinish
	sc.heldFinish = ""
	if finish == "" {
		return nil
	}
	ccFinish, _, _ := finishStatus(finish)
	ccFinish = shared.TerminalReason(sc.toolsSeen, "tool_calls", ccFinish)
	var usage map[string]any
	input, completion := int64(0), int64(0)
	var details shared.UsageDetails
	if sc.usage != nil {
		input = sc.usage.PromptTokenCount
		completion = sc.usage.counts()
		details = sc.usage.details()
		usage = shared.CCUsageFrom(input, completion, details)
	}
	return [][]byte{sc.chatChunks().Finish(ccFinish, usage)}
}

// ---- Gemini -> claude (Messages) stream (FR-006) ----

func (sc *StreamConverter) dispatchClaude(frame *gemStreamFrame) ([][]byte, *errclass.Error) {
	var events [][]byte
	if len(frame.Candidates) == 0 {
		return nil, nil
	}
	cand := frame.Candidates[0]
	if !sc.started {
		sc.started = true
		sc.id = newStreamID()
		input := int64(0)
		if sc.usage != nil {
			input = sc.usage.PromptTokenCount
		}
		em := shared.NewClaudeEventEmitter(sc.id, sc.model)
		sc.claudeEm = &em
		events = append(events, sc.claudeEm.MessageStart(input))
	}
	textParts, calls := observeCandidate(cand.Content)
	for _, text := range textParts {
		if !sc.textOpen {
			events = append(events, sc.stopBlocks()...)
			sc.textIndex = sc.nextIndex
			sc.nextIndex++
			sc.textOpen = true
			events = append(events, sc.claudeEm.ContentBlockStart(sc.textIndex, "text", map[string]any{"text": ""}))
		}
		events = append(events, sc.claudeEm.ContentBlockDelta(sc.textIndex,
			map[string]any{"type": "text_delta", "text": text}))
	}
	for _, call := range calls {
		sc.toolsSeen = true
		events = append(events, sc.stopBlocks()...)
		t := &claudeToolState{blockIndex: sc.nextIndex, id: gemCallID(call.Name, sc.toolOrd), name: call.Name}
		sc.nextIndex++
		sc.tools[t.blockIndex] = t
		sc.toolOrder = append(sc.toolOrder, t.blockIndex)
		sc.toolOrd++
		events = append(events, sc.claudeEm.ContentBlockStart(t.blockIndex, "tool_use",
			map[string]any{"id": t.id, "name": t.name, "input": map[string]any{}}))
		args := shared.DefaultArgs(string(call.Args))
		t.args.WriteString(args)
		events = append(events, sc.claudeEm.ContentBlockDelta(t.blockIndex,
			map[string]any{"type": "input_json_delta", "partial_json": args}))
		events = append(events, sc.claudeEm.ContentBlockStop(t.blockIndex))
		t.stopped = true
	}
	if cand.FinishReason != "" && !sc.finishSeen {
		// Claude synthesis renders its terminal at close (Gemini streams
		// carry no explicit end frame); remember the reason, close open
		// blocks now.
		sc.finishSeen = true
		sc.heldFinish = cand.FinishReason
		events = append(events, sc.stopBlocks()...)
	}
	return events, nil
}

// stopBlocks closes the open text block and any open tool_use blocks
// exactly once (Anthropic forbids two concurrently-open blocks).
func (sc *StreamConverter) stopBlocks() [][]byte {
	var events [][]byte
	if sc.textOpen {
		sc.textOpen = false
		events = append(events, sc.claudeEm.ContentBlockStop(sc.textIndex))
	}
	for _, idx := range sc.toolOrder {
		t := sc.tools[idx]
		if t != nil && !t.stopped {
			t.stopped = true
			events = append(events, sc.claudeEm.ContentBlockStop(t.blockIndex))
		}
	}
	return events
}

// claudeTerminal renders message_delta with the mapped stop reason and
// captured usage, exactly once.
func (sc *StreamConverter) claudeTerminal() [][]byte {
	finish := sc.heldFinish
	sc.heldFinish = ""
	if finish == "" {
		return nil
	}
	_, claudeStop, _ := finishStatus(finish)
	reason := shared.TerminalReason(sc.toolsSeen, "tool_use", claudeStop)
	input, output := int64(0), int64(0)
	var cacheRead *int64
	if sc.usage != nil {
		input = sc.usage.PromptTokenCount
		output = sc.usage.counts()
		cacheRead = sc.usage.CachedContentTokenCount
	}
	return [][]byte{sc.claudeEm.MessageDelta(&reason,
		shared.ClaudeUsage(shared.ClampSubtract(input, cacheRead), output, cacheRead, nil))}
}

// ---- Gemini -> openai-response (Responses) stream (FR-006) ----

func (sc *StreamConverter) dispatchResponses(frame *gemStreamFrame) ([][]byte, *errclass.Error) {
	var events [][]byte
	if len(frame.Candidates) == 0 {
		return nil, nil
	}
	cand := frame.Candidates[0]
	if !sc.started {
		sc.started = true
		sc.id = newStreamID()
		events = append(events, sc.responsesEm().Created())
	}
	textParts, calls := observeCandidate(cand.Content)
	for _, text := range textParts {
		if sc.msgIndex < 0 {
			sc.msgIndex = sc.outCount
			sc.outCount++
			events = append(events, sc.responsesEm().ItemAdded(sc.msgIndex, map[string]any{
				"type": "message", "id": sc.id, "role": "assistant", "content": []any{},
			}))
		}
		sc.respText.WriteString(text)
		events = append(events, sc.responsesEm().TextDelta(sc.id, sc.msgIndex, text))
	}
	for _, call := range calls {
		sc.toolsSeen = true
		item := respCallItem{outIdx: sc.outCount, id: gemCallID(call.Name, sc.toolOrd), name: call.Name}
		sc.outCount++
		sc.toolOrd++
		args := shared.DefaultArgs(string(call.Args))
		item.args.WriteString(args)
		sc.respCalls = append(sc.respCalls, item)
		events = append(events, sc.responsesEm().ItemAdded(item.outIdx, map[string]any{
			"type": "function_call", "call_id": item.id, "name": item.name, "arguments": "",
		}))
		if sc.respTools.IsCustom(item.name) {
			// Custom tools stream as wrapped JSON ({"input":"..."}) that
			// only unwraps once complete; Gemini functionCall parts arrive
			// whole, so the entire unwrapped input emits as one delta.
			emitted := 0
			if tail := shared.CustomInputTail(args, &emitted); tail != "" {
				events = append(events, sc.responsesEm().InputDelta(item.id, item.outIdx, tail))
			}
		} else {
			events = append(events, sc.responsesEm().ArgsDelta(item.id, item.outIdx, args))
		}
	}
	if cand.FinishReason != "" && sc.heldFinish == "" {
		sc.heldFinish = cand.FinishReason
	}
	return events, nil
}

// responsesEm binds the shared Responses emitter kernel to the captured
// stream identity (FR-006 sibling parity).
func (sc *StreamConverter) responsesEm() shared.ResponsesEventEmitter {
	return shared.ResponsesEventEmitter{ID: sc.id, Model: sc.model, Tools: sc.respTools}
}

// responsesTerminal renders the terminal Responses event sequence:
// per-item done events followed by response.completed, mirroring the
// non-stream toResponses converter's output shape (FR-006 mode parity).
func (sc *StreamConverter) responsesTerminal() [][]byte {
	finish := sc.heldFinish
	sc.heldFinish = ""
	if finish == "" {
		return nil
	}
	_, _, respStatus := finishStatus(finish)
	if sc.toolsSeen {
		respStatus = "completed" // tool calls are represented by output items, not status
	}
	// The shared assembler reproduces the streamed announcement order: the
	// message slot is pinned at sc.msgIndex (assigned at the first text
	// delta, after any tools-first announcements).
	oa := shared.NewOutputAssembler(sc.id, sc.respTools)
	reserved := false
	for i := range sc.respCalls {
		item := &sc.respCalls[i]
		if sc.msgIndex >= 0 && !reserved && item.outIdx > sc.msgIndex {
			oa.ReserveTextSlot()
			reserved = true
		}
		oa.AppendFunctionCall(item.id, item.name, shared.DefaultArgs(item.args.String()))
	}
	if sc.msgIndex >= 0 && !reserved {
		oa.ReserveTextSlot()
	}
	oa.AddText(sc.respText.String())
	input, completion := int64(0), int64(0)
	var details shared.UsageDetails
	if sc.usage != nil {
		input = sc.usage.PromptTokenCount
		completion = sc.usage.counts()
		details = sc.usage.details()
	}
	usage := shared.NewResponsesUsageFrom(input, completion, details)
	items := oa.Render()
	em := sc.responsesEm()
	var events [][]byte
	for idx, item := range items {
		v, ok := item.(shared.RespItem)
		if !ok {
			continue
		}
		switch v.Type {
		case "message":
			text := sc.respText.String()
			events = append(events, em.TextDone(v.ID, idx, text), em.ContentPartDone(v.ID, idx, text), em.ItemDone(idx, v))
		case "function_call":
			events = append(events, em.ArgsDone(v.CallID, idx, v.Name, v.Arguments), em.ItemDone(idx, v))
		case "custom_tool_call":
			events = append(events, em.InputDone(v.CallID, idx, v.Input), em.ItemDone(idx, v))
		}
	}
	return append(events, em.Completed(respStatus, usage, items))
}

// Flush terminates a stream whose upstream closed without a client-protocol
// terminal: Gemini streams carry no [DONE] sentinel, so the pump usually
// breaks on upstream close with heldFinish set. Claude synthesis emits the
// held message_delta plus message_stop; openai the deferred finish chunk;
// responses the completed sequence. One-shot.
func (sc *StreamConverter) Flush() [][]byte {
	if sc.done {
		return nil
	}
	sc.done = true
	switch sc.sourceFormat {
	case "openai":
		var events [][]byte
		events = append(events, sc.openAITerminal()...)
		return events
	case "claude":
		events := sc.claudeTerminal()
		if sc.claudeEm == nil {
			return nil
		}
		return append(events, sc.claudeEm.MessageStop())
	case "openai-response":
		return sc.responsesTerminal()
	}
	return nil
}

// newStreamID renders one downstream-visible response identity.
func newStreamID() string {
	return "gemini-" + strconv.FormatInt(time.Now().UnixNano(), 36)
}

func sget(m map[string]any, k string) string {
	v, _ := m[k].(string)
	return v
}
