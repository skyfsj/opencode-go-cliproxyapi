package gemini

import (
	"encoding/json"
	"strings"
	"testing"
)

// zcodeLikeSchema mirrors a real coding-agent tool schema: $defs/$ref,
// anyOf with null branches, arrays, additionalProperties:false — everything
// Gemini's function declaration proto rejects with a 400.
func zcodeLikeSchema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"command": {"anyOf": [
				{"type": "string"},
				{"type": "array", "items": {"type": "string"}},
				{"type": "object", "properties": {"cmd": {"type": "string"}}}
			]},
			"mode": {"$defs": {"m": {"type": "string", "enum": ["fast","slow"]}}, "oneOf": [{"$ref": "#/$defs/m"}, {"type": "null"}]},
			"timeout": {"type": "integer", "default": 30, "exclusiveMinimum": 0},
			"tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
			"tree": {"$ref": "#/$defs/node"}
		},
		"$defs": {"node": {"type": "object", "properties": {"children": {"$ref": "#/$defs/node"}, "name": {"type": "string"}}}},
		"required": ["command"],
		"additionalProperties": false
	}`)
}

func decodeSan(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("sanitized schema not JSON: %v\n%s", err, raw)
	}
	return m
}

func keysForbidden(t *testing.T, m map[string]any) []string {
	t.Helper()
	bad := []string{}
	var walk func(node any, path string)
	walk = func(node any, path string) {
		mm, ok := node.(map[string]any)
		if !ok {
			return
		}
		for k, v := range mm {
			switch k {
			case "$defs", "definitions", "$ref", "anyOf", "oneOf", "allOf", "not", "if", "then", "else":
				bad = append(bad, path+"."+k)
			case "format":
				if f, _ := v.(string); !schemaFormatAllowlist[f] {
					bad = append(bad, path+".format="+f)
				}
			}
			walk(v, path+"."+k)
		}
	}
	walk(m, "")
	return bad
}

func TestSanitizeZCodeLikeSchema(t *testing.T) {
	san := decodeSan(t, SanitizeJSONSchema(zcodeLikeSchema()))
	if bad := keysForbidden(t, san); len(bad) > 0 {
		t.Fatalf("forbidden keys survived: %v\n%s", bad, san)
	}
	if san["type"] != "object" {
		t.Errorf("root type = %v", san["type"])
	}
	props := san["properties"].(map[string]any)
	// anyOf multi-type union keeps the first representable branch on the
	// wire and names the dropped ones in the description (no null here, so
	// no nullable flag).
	cmd, _ := props["command"].(map[string]any)
	if cmd == nil || cmd["type"] != "string" {
		t.Errorf("command = %v, want the string branch kept on the wire", cmd)
	}
	if desc, _ := cmd["description"].(string); desc == "" || !strings.Contains(desc, "anyOf also allows") {
		t.Errorf("command lost the dropped-branch description: %v", cmd["description"])
	}
	// $ref to a local enum table inlines to the enum + nullable.
	mode, _ := props["mode"].(map[string]any)
	if mode == nil {
		t.Fatalf("mode missing: %s", san)
	}
	if en, ok := mode["enum"].([]any); !ok || len(en) != 2 || mode["nullable"] != true {
		t.Errorf("mode = %v, want inlined enum + nullable", mode)
	}
	// default folded into the description, exclusiveMinimum dropped.
	timeout, _ := props["timeout"].(map[string]any)
	if timeout["default"] != nil || timeout["exclusiveMinimum"] != nil {
		t.Errorf("timeout kept dead keys: %v", timeout)
	}
	if desc, _ := timeout["description"].(string); desc == "" {
		t.Errorf("timeout lost its normalization note: %v", timeout)
	}
	// Recursive $ref degrades to a permissive object, no infinite loop.
	tree, _ := props["tree"].(map[string]any)
	if tree == nil || tree["type"] != "object" {
		t.Errorf("tree = %v, want degraded object", tree)
	}
	// additionalProperties:false has no note-worthy content (dropped
	// silently — false adds nothing the model needs).
	if _, present := san["additionalProperties"]; present {
		t.Errorf("additionalProperties must not survive: %v", san["additionalProperties"])
	}
}

func TestSanitizeAlwaysValid(t *testing.T) {
	cases := []json.RawMessage{
		nil,
		json.RawMessage("null"),
		json.RawMessage(`"not an object"`),
		json.RawMessage(`{}`),
		json.RawMessage(`{"type":["string","null"]}`),
		json.RawMessage(`{"type":"string","format":"uuid"}`),
		json.RawMessage(`{"const":"fixed"}`),
		json.RawMessage(`{"enum":[1,2,3]}`),
		json.RawMessage(`{"items":[{"type":"string"},{"type":"number"}]}`),
		json.RawMessage(`{"allOf":[{"type":"object"},{"properties":{"a":{"type":"string"}}}]}`),
		json.RawMessage(`{"$ref":"#/definitions/missing"}`),
	}
	for i, c := range cases {
		out := SanitizeJSONSchema(c)
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatalf("case %d: invalid output %s", i, out)
		}
		if bad := keysForbidden(t, m); len(bad) > 0 {
			t.Errorf("case %d: forbidden keys %v in %s", i, bad, out)
		}
	}
}

func TestBuildRequestGeminiDeclarationsSanitized(t *testing.T) {
	src := json.RawMessage(`{
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {
			"type": "object",
			"properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
			"$defs": {"a": {"type": "string"}}
		}}}]
	}`)
	body, eErr := BuildRequest("gemini-x", "openai", src, nil)
	if eErr != nil {
		t.Fatalf("BuildRequest: %v", eErr)
	}
	m := jsonMap(t, body)
	decl := m["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	params := decl["parameters"].(map[string]any)
	if _, has := params["$defs"]; has {
		t.Errorf("$defs leaked: %s", body)
	}
	x := params["properties"].(map[string]any)["x"].(map[string]any)
	if x["type"] != "string" || x["nullable"] != true {
		t.Errorf("x = %v, want string+nullable", x)
	}
}
