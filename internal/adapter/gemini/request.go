// Request translation for the Gemini adapter (FR-005): an inbound request
// body from any client protocol ("openai" Chat Completions, "claude"
// Anthropic Messages, "openai-response" Responses) is translated into a
// Gemini generateContent body for upstreamModel. Errors are descriptive
// and redacted — no silent loss of tools or reasoning controls.
package gemini

import (
	"encoding/json"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"

	"opencode-go-cliproxyapi/internal/adapter/shared"
	"opencode-go-cliproxyapi/internal/errclass"
	"opencode-go-cliproxyapi/internal/thinking"
)

// BuildRequest translates an inbound request body from sourceFormat into
// a Gemini generateContent body for upstreamModel (FR-005). ts carries
// the target model's thinking capability so reasoning controls map to a
// supported thinkingBudget instead of being silently degraded (FR-005).
// Unknown formats are ClassUnsupported; malformed input is
// ClassTranslation.
func BuildRequest(upstreamModel, sourceFormat string, sourceBody []byte, ts *pluginapi.ThinkingSupport, tools ...*shared.ResponseTools) ([]byte, *errclass.Error) {
	switch sourceFormat {
	case "openai":
		return fromChatCompletions(sourceBody, ts)
	case "claude":
		return fromClaude(sourceBody, ts)
	case "openai-response":
		return fromResponses(sourceBody, ts, shared.ResponseToolContext(tools))
	default:
		return nil, shared.UnsupportedFormat(sourceFormat, targetLabel)
	}
}

// ---- outbound Gemini shapes ----

// part is one Gemini content part. Text and InlineData are client content;
// FunctionCall/FunctionResponse only appear in history translation.
type part struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *inlineData       `json:"inlineData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
}

// inlineData carries one base64-encoded media part.
type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// functionCall is one Gemini tool invocation (assistant side).
type functionCall struct {
	Name string `json:"name"`
	Args any    `json:"args"`
}

// functionResponse is one Gemini tool result (user side); Response must be
// a JSON object, so plain-text tool results wrap under "result".
type functionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

// content is one Gemini turn; Role is "user" or "model".
type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type functionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type tool struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations,omitempty"`
}

type thinkingConfig struct {
	ThinkingBudget  *int64 `json:"thinkingBudget,omitempty"`
	IncludeThoughts bool   `json:"includeThoughts,omitempty"`
}

type generationConfig struct {
	MaxOutputTokens *int64          `json:"maxOutputTokens,omitempty"`
	StopSequences   []string        `json:"stopSequences,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"topP,omitempty"`
	ThinkingConfig  *thinkingConfig `json:"thinkingConfig,omitempty"`
}

type geminiRequest struct {
	Contents          []content         `json:"contents"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	Tools             []tool            `json:"tools,omitempty"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig *functionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type functionCallingConfig struct {
	Mode                 string   `json:"mode"` // AUTO | ANY | NONE
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// encode finalizes and encodes a translated request; contents always ship
// as an array. Composed types are always marshallable.
func encode(req *geminiRequest) []byte {
	if req.Contents == nil {
		req.Contents = []content{}
	}
	b, _ := json.Marshal(req)
	return b
}

// applyToolChoice maps a normalized tool choice onto the Gemini
// functionCallingConfig (FR-005): absent omits the config entirely (the
// protocol default is AUTO), auto→AUTO, any/required→ANY, named→ANY
// constrained to the one allowed name, none→NONE.
func applyToolChoice(req *geminiRequest, kind, name string) {
	if kind == shared.ToolChoiceAbsent {
		return
	}
	cfg := &functionCallingConfig{Mode: "AUTO"}
	switch kind {
	case shared.ToolChoiceAuto:
	case shared.ToolChoiceAny:
		cfg.Mode = "ANY"
	case shared.ToolChoiceNone:
		cfg.Mode = "NONE"
	case shared.ToolChoiceNamed:
		cfg.Mode = "ANY"
		cfg.AllowedFunctionNames = []string{name}
	}
	req.ToolConfig = &toolConfig{FunctionCallingConfig: cfg}
}

// applyThinking maps a resolved thinking budget onto
// generationConfig.thinkingConfig: budget > 0 enables thinking at the
// capability-clamped budget; an explicit off-state requests 0 when the
// model advertises ZeroAllowed and otherwise degrades to the weakest
// supported budget rather than silently re-enabling an unclamped default.
func applyThinking(req *geminiRequest, budget int64, ts *pluginapi.ThinkingSupport) {
	cfg := &thinkingConfig{}
	switch {
	case budget > 0:
		b := budget
		cfg.ThinkingBudget = &b
		cfg.IncludeThoughts = true
	case ts != nil && ts.ZeroAllowed:
		zero := int64(0)
		cfg.ThinkingBudget = &zero
	default:
		levels := thinking.SupportedLevels(ts)
		weakest, ok := thinking.BudgetFromEffort(levels[0], ts)
		if !ok || weakest <= 0 {
			return // nothing representable: omit the config entirely
		}
		cfg.ThinkingBudget = &weakest
	}
	if req.GenerationConfig == nil {
		req.GenerationConfig = &generationConfig{}
	}
	req.GenerationConfig.ThinkingConfig = cfg
}

// resolveClaudeBudget renders the Anthropic thinking control into a
// Gemini thinking budget through the shared context kernels.
func resolveClaudeBudget(t *shared.ClaudeThinking, ts *pluginapi.ThinkingSupport) int64 {
	if !shared.ThinkingEnabled(t) {
		return 0
	}
	b, _ := thinking.BudgetFromEffort(thinking.EffortFromBudget(t.BudgetTokens, ts), ts)
	return b
}

// imagePart converts an image URL into a Gemini part: base64 data URLs
// become inlineData; other URLs have no generateContent representation
// and are rejected descriptively (FR-005 explicit policy).
func imagePart(url string) (part, *errclass.Error) {
	if rest, ok := strings.CutPrefix(url, "data:"); ok {
		comma := strings.Index(rest, ",")
		if comma < 0 {
			return part{}, errclass.Translation("malformed data URL: missing ',' separator")
		}
		if strings.HasSuffix(rest[:comma], ";base64") {
			media := strings.TrimSuffix(rest[:comma], ";base64")
			if media == "" {
				media = "application/octet-stream"
			}
			return part{InlineData: &inlineData{MimeType: media, Data: rest[comma+1:]}}, nil
		}
	}
	return part{}, errclass.Translation("image url is not a base64 data URL; gemini generateContent carries inline data only")
}

// addFunctionDeclaration appends one function tool to the request's single
// functionDeclarations group, creating the group on first use.
func addFunctionDeclaration(req *geminiRequest, name, description string, schema json.RawMessage) {
	if len(req.Tools) == 0 {
		req.Tools = []tool{{}}
	}
	req.Tools[0].FunctionDeclarations = append(req.Tools[0].FunctionDeclarations, functionDeclaration{
		Name: name, Description: description, Parameters: SanitizeJSONSchema(schema),
	})
}

// setSystemInstruction collapses the collected system text into one
// systemInstruction content (blank-line joined, matching the sibling
// adapters' system joins).
func setSystemInstruction(req *geminiRequest, system []string) {
	if len(system) == 0 {
		return
	}
	req.SystemInstruction = &content{Parts: []part{{Text: strings.Join(system, "\n\n")}}}
}

// functionResponsePart renders one named Gemini tool-result part.
func functionResponsePart(name, result string) part {
	return part{FunctionResponse: &functionResponse{
		Name: name, Response: map[string]any{"result": result},
	}}
}

// ---- openai (Chat Completions) -> Gemini (FR-005) ----

func fromChatCompletions(body []byte, ts *pluginapi.ThinkingSupport) ([]byte, *errclass.Error) {
	var src shared.ChatCompletionsRequest
	if err := json.Unmarshal(body, &src); err != nil {
		return nil, errclass.Translation("malformed openai request JSON: " + err.Error())
	}
	req := &geminiRequest{}
	gen := &generationConfig{}
	maxTokens := int64(0)
	if src.MaxTokens != nil {
		maxTokens = *src.MaxTokens
	} else if src.MaxCompletionTokens != nil {
		maxTokens = *src.MaxCompletionTokens
	}
	mt := shared.ClaudeMaxTokens(maxTokens)
	gen.MaxOutputTokens = &mt
	if len(src.Stop) > 0 && string(src.Stop) != "null" {
		var one string
		if json.Unmarshal(src.Stop, &one) == nil {
			gen.StopSequences = []string{one}
		} else {
			var many []string
			if err := json.Unmarshal(src.Stop, &many); err != nil {
				return nil, errclass.Translation("stop must be a string or an array of strings")
			}
			gen.StopSequences = many
		}
	}
	gen.Temperature = src.Temperature
	gen.TopP = src.TopP
	req.GenerationConfig = gen

	var system []string
	// callNames tracks Chat Completions tool_call ids to their tool names:
	// Gemini functionResponse parts are addressed by NAME, and a tool
	// message carries only the id.
	callNames := map[string]string{}
	for i := range src.Messages {
		m := &src.Messages[i]
		switch m.Role {
		case "system", "developer":
			parts, eErr := ccParts(m.Content)
			if eErr != nil {
				return nil, eErr
			}
			for _, p := range parts {
				if p.InlineData != nil {
					return nil, shared.SystemImageRejected()
				}
				system = append(system, p.Text)
			}
		case "assistant":
			parts, eErr := ccParts(m.Content)
			if eErr != nil {
				return nil, eErr
			}
			if hasImage(parts) {
				return nil, errclass.Translation("assistant messages cannot carry image content")
			}
			c := content{Role: "model"}
			c.Parts = append(c.Parts, parts...)
			for _, tc := range m.ToolCalls {
				args, eErr := shared.DecodeArgs(tc.Function.Arguments)
				if eErr != nil {
					return nil, eErr
				}
				callNames[tc.ID] = tc.Function.Name
				c.Parts = append(c.Parts, part{FunctionCall: &functionCall{Name: tc.Function.Name, Args: args}})
			}
			if len(c.Parts) > 0 {
				req.Contents = append(req.Contents, c)
			}
		case "tool":
			text, eErr := shared.ToolResultText(m.Content, "tool messages carry text only")
			if eErr != nil {
				return nil, eErr
			}
			req.Contents = append(req.Contents, content{Role: "user", Parts: []part{
				functionResponsePart(callNames[m.ToolCallID], text),
			}})
		case "user":
			parts, eErr := ccParts(m.Content)
			if eErr != nil {
				return nil, eErr
			}
			c := content{Role: "user", Parts: parts}
			if len(c.Parts) > 0 {
				req.Contents = append(req.Contents, c)
			}
		default:
			return nil, shared.ValidateRole(m.Role, targetLabel)
		}
	}
	setSystemInstruction(req, system)

	for _, t := range src.Tools {
		if t.Type == "namespace" {
			continue
		}
		if eErr := shared.FunctionTool(t.Type, targetLabel); eErr != nil {
			return nil, eErr
		}
		addFunctionDeclaration(req, t.Function.Name, t.Function.Description, t.Function.Parameters)
	}
	kind, tcName, eErr := shared.DecodeToolChoice(src.ToolChoice)
	if eErr != nil {
		return nil, eErr
	}
	applyToolChoice(req, kind, tcName)

	if src.ReasoningEffort != "" {
		if eErr := thinking.ValidateEffort(src.ReasoningEffort, ts); eErr != nil {
			return nil, eErr
		}
		budget, _ := thinking.BudgetFromEffort(src.ReasoningEffort, ts)
		applyThinking(req, budget, ts)
	}
	return encode(req), nil
}

// ccParts normalizes OpenAI-style content into Gemini parts (FR-005
// multimodal preservation): text parts stay text, image parts become
// inlineData. The shared kernel owns validation and the empty-input
// policy.
func ccParts(raw json.RawMessage) ([]part, *errclass.Error) {
	decoded, eErr := shared.DecodeStringOrParts(raw, targetLabel)
	if eErr != nil {
		return nil, eErr
	}
	var out []part
	for _, p := range decoded {
		if p.ImageURL != "" {
			ip, eErr := imagePart(p.ImageURL)
			if eErr != nil {
				return nil, eErr
			}
			out = append(out, ip)
			continue
		}
		out = append(out, part{Text: p.Text})
	}
	return out, nil
}

func hasImage(parts []part) bool {
	for _, p := range parts {
		if p.InlineData != nil {
			return true
		}
	}
	return false
}

// ---- claude (Anthropic Messages) -> Gemini (FR-005) ----

func fromClaude(body []byte, ts *pluginapi.ThinkingSupport) ([]byte, *errclass.Error) {
	src, eErr := shared.DecodeClaudeMessages(body)
	if eErr != nil {
		return nil, eErr
	}
	req := &geminiRequest{}
	gen := &generationConfig{}
	mt := src.MaxTokens // already defaulted by ClaudeMaxTokens in the decode kernel
	gen.MaxOutputTokens = &mt
	if len(src.StopSequences) > 0 {
		gen.StopSequences = src.StopSequences
	}
	gen.Temperature = src.Temperature
	gen.TopP = src.TopP
	req.GenerationConfig = gen

	var system []string
	if src.System != "" {
		system = append(system, src.System)
	}
	// callNames tracks Claude tool_use ids so tool_result blocks can render
	// named functionResponse parts.
	callNames := map[string]string{}
	for i := range src.Messages {
		m := &src.Messages[i]
		switch m.Role {
		case "user":
			c := content{Role: "user"}
			toolOuts := []part{}
			if m.Content != "" {
				c.Parts = append(c.Parts, part{Text: m.Content})
			}
			for j := range m.Blocks {
				blk := &m.Blocks[j]
				switch blk.Kind {
				case "text":
					c.Parts = append(c.Parts, part{Text: blk.Text})
				case "image":
					ip, eErr := imagePart(blk.URL)
					if eErr != nil {
						return nil, eErr
					}
					c.Parts = append(c.Parts, ip)
				case "tool_result":
					text, eErr := shared.ToolResultText(blk.Result, "tool messages carry text only")
					if eErr != nil {
						return nil, eErr
					}
					if blk.IsError {
						text = shared.ToolResultErrorPrefix + text
					}
					toolOuts = append(toolOuts, functionResponsePart(callNames[blk.CallID], text))
				default:
					return nil, shared.UnsupportedPartType(blk.Kind, targetLabel)
				}
			}
			if len(c.Parts) > 0 {
				req.Contents = append(req.Contents, c)
			}
			if len(toolOuts) > 0 {
				req.Contents = append(req.Contents, content{Role: "user", Parts: toolOuts})
			}
		case "assistant":
			c := content{Role: "model"}
			if m.Content != "" {
				c.Parts = append(c.Parts, part{Text: m.Content})
			}
			for j := range m.Blocks {
				blk := &m.Blocks[j]
				switch blk.Kind {
				case "text":
					c.Parts = append(c.Parts, part{Text: blk.Text})
				case "thinking", "redacted_thinking":
					// FR-005 explicit omission policy: historical thinking
					// cannot be fabricated as a thought part without the
					// upstream's own signatures, so it is omitted.
				case "tool_use":
					args, eErr := shared.DecodeArgs(string(blk.Input))
					if eErr != nil {
						return nil, eErr
					}
					callNames[blk.CallID] = blk.Name
					c.Parts = append(c.Parts, part{FunctionCall: &functionCall{Name: blk.Name, Args: args}})
				default:
					return nil, shared.UnsupportedPartType(blk.Kind, targetLabel)
				}
			}
			if len(c.Parts) > 0 {
				req.Contents = append(req.Contents, c)
			}
		case "system":
			if m.Content != "" {
				system = append(system, m.Content)
			}
			for j := range m.Blocks {
				blk := &m.Blocks[j]
				switch blk.Kind {
				case "text":
					system = append(system, blk.Text)
				case "image":
					return nil, shared.SystemImageRejected()
				default:
					return nil, shared.UnsupportedPartType(blk.Kind, targetLabel)
				}
			}
		default:
			return nil, shared.ValidateRole(m.Role, targetLabel)
		}
	}
	setSystemInstruction(req, system)

	for _, t := range src.Tools {
		addFunctionDeclaration(req, t.Name, t.Description, t.InputSchema)
	}
	applyToolChoice(req, src.ToolChoiceKind, src.ToolChoiceName)
	if shared.ThinkingEnabled(src.Thinking) {
		applyThinking(req, resolveClaudeBudget(src.Thinking, ts), ts)
	}
	return encode(req), nil
}

// ---- openai-response (Responses) -> Gemini (FR-005) ----

func fromResponses(body []byte, ts *pluginapi.ThinkingSupport, tools *shared.ResponseTools) ([]byte, *errclass.Error) {
	var src shared.ResponsesRequest
	if err := json.Unmarshal(body, &src); err != nil {
		return nil, errclass.Translation("malformed openai-response request JSON: " + err.Error())
	}
	req := &geminiRequest{}
	gen := &generationConfig{}
	if src.MaxOutputTokens != nil && *src.MaxOutputTokens > 0 {
		mt := *src.MaxOutputTokens
		gen.MaxOutputTokens = &mt
	}
	gen.Temperature = src.Temperature
	gen.TopP = src.TopP
	req.GenerationConfig = gen

	var system []string
	instr, eErr := src.DecodeInstructions()
	if eErr != nil {
		return nil, eErr
	}
	if instr != "" {
		system = append(system, instr)
	}

	items, eErr := tools.Normalize(&src, targetLabel)
	if eErr != nil {
		return nil, eErr
	}
	// callNames tracks Responses call ids so function_call_output items
	// render named functionResponse parts.
	callNames := map[string]string{}
	appendSystem := func(text string) {
		if text != "" {
			system = append(system, text)
		}
	}
	for _, item := range items {
		switch item.Type {
		case "message":
			decoded, eErr := shared.DecodeStringOrParts(item.Content, targetLabel)
			if eErr != nil {
				return nil, eErr
			}
			switch item.Role {
			case "user":
				c := content{Role: "user"}
				for _, p := range decoded {
					if p.ImageURL != "" {
						ip, eErr := imagePart(p.ImageURL)
						if eErr != nil {
							return nil, eErr
						}
						c.Parts = append(c.Parts, ip)
						continue
					}
					c.Parts = append(c.Parts, part{Text: p.Text})
				}
				if len(c.Parts) > 0 {
					req.Contents = append(req.Contents, c)
				}
			case "assistant":
				c := content{Role: "model"}
				for _, p := range decoded {
					if p.ImageURL != "" {
						return nil, errclass.Translation("assistant messages cannot carry image content")
					}
					c.Parts = append(c.Parts, part{Text: p.Text})
				}
				if len(c.Parts) > 0 {
					req.Contents = append(req.Contents, c)
				}
			case "system", "developer":
				for _, p := range decoded {
					if p.ImageURL != "" {
						return nil, shared.SystemImageRejected()
					}
					appendSystem(p.Text)
				}
			default:
				return nil, shared.ValidateRole(item.Role, targetLabel)
			}
		case "function_call", "custom_tool_call":
			args := item.Arguments
			if args == "" && item.Input != "" {
				args = item.Input
			}
			decoded, eErr := shared.DecodeArgs(args)
			if eErr != nil {
				// Freeform custom tool input is not JSON: wrap verbatim.
				decoded = map[string]any{"input": args}
			}
			callNames[item.CallID] = item.Name
			req.Contents = append(req.Contents, content{Role: "model", Parts: []part{
				{FunctionCall: &functionCall{Name: item.Name, Args: decoded}},
			}})
		case "function_call_output", "custom_tool_call_output":
			text, eErr := shared.RespOutputText(item.Output, "tool outputs carry text only")
			if eErr != nil {
				return nil, eErr
			}
			req.Contents = append(req.Contents, content{Role: "user", Parts: []part{
				functionResponsePart(callNames[item.CallID], text),
			}})
		case "reasoning":
			// FR-005 explicit omission policy: reasoning summaries have no
			// representable Gemini history part.
		default:
			return nil, shared.UnsupportedInputItemType(item.Type)
		}
	}
	setSystemInstruction(req, system)

	for _, t := range src.Tools {
		if eErr := shared.FunctionTool(t.Type, targetLabel); eErr != nil {
			return nil, eErr
		}
		addFunctionDeclaration(req, t.Name, t.Description, t.Parameters)
	}
	kind, tcName, eErr := shared.DecodeToolChoice(src.ToolChoice)
	if eErr != nil {
		return nil, eErr
	}
	applyToolChoice(req, kind, tcName)

	effort := ""
	if src.Reasoning != nil {
		effort = src.Reasoning.Effort
	}
	if effort != "" {
		if eErr := thinking.ValidateEffort(effort, ts); eErr != nil {
			return nil, eErr
		}
		budget, _ := thinking.BudgetFromEffort(effort, ts)
		applyThinking(req, budget, ts)
	}
	return encode(req), nil
}
