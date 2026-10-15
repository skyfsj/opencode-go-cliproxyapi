// Tool-schema sanitization for Gemini function declarations: the upstream
// accepts only an OpenAPI 3.0 subset of JSON Schema, while coding agents
// (coding CLI harnesses) declare tools with full-JSON-Schema keywords
// ($defs/$ref, anyOf/oneOf/allOf, not, if/then/else, ...). The upstream
// rejects such declarations with a 400 that breaks every subsequent agent
// turn, so unsupported constructs are rewritten gracefully here instead:
// refs inline, composites collapse, and anything the target cannot express
// is folded into description text so model intent survives normalization.
package gemini

import (
	"encoding/json"
	"fmt"
	"strings"
)

// schemaMaxRefDepth bounds $ref inlining; deeper cycles degrade to a
// permissive placeholder so recursive schemas (trees, linked lists) still
// produce a valid declaration.
const schemaMaxRefDepth = 4

// schemaKeepableKeys are the JSON field names Gemini's Schema proto accepts
// in functionDeclarations.parameters. Everything else either gets a local
// rewrite (refs, composites) or is folded into the description note.
var schemaKeepableKeys = map[string]bool{
	"type": true, "nullable": true, "enum": true, "format": true,
	"description": true, "properties": true, "required": true,
	"items": true, "additionalProperties": true, "pattern": true,
	"minItems": true, "maxItems": true, "minLength": true,
	"maxLength": true, "example": true, "propertyOrdering": true,
	"title": true,
}

// schemaFormatAllowlist mirrors the formats the Gemini string/number
// schema fields accept; anything else is dropped (the proto rejects
// unknown format values).
var schemaFormatAllowlist = map[string]bool{
	"date-time": true, "date": true, "time": true, "duration": true,
	"float": true, "double": true, "int32": true, "int64": true,
}

// SanitizeJSONSchema renders a client tool-parameter schema into the
// Gemini-acceptable subset. It is best-effort, never an error: malformed
// or exotic schemas degrade to describable remainders, because a slightly
// less precise declaration lets the agent loop run while a 400 kills it.
func SanitizeJSONSchema(raw json.RawMessage) json.RawMessage {
	if !sharedHasContent(raw) {
		return emptyObjectSchemaJSON()
	}
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil || root == nil {
		return emptyObjectSchemaJSON()
	}
	refs := collectDefs(root)
	san := sanitizeSchemaNode(root, refs, 0, map[string]bool{})
	out, err := json.Marshal(san)
	if err != nil {
		return emptyObjectSchemaJSON()
	}
	return out
}

func emptyObjectSchemaJSON() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{}}`)
}

func sharedHasContent(raw json.RawMessage) bool {
	return len(raw) > 0 && string(raw) != "null"
}

// collectDefs hoists local $defs/definitions tables so $ref pointers
// (#/defs/NAME, #/definitions/NAME) resolve during inlining.
func collectDefs(root map[string]any) map[string]map[string]any {
	refs := map[string]map[string]any{}
	for _, tableKey := range []string{"$defs", "definitions"} {
		if table, ok := root[tableKey].(map[string]any); ok {
			for name, def := range table {
				if m, ok := def.(map[string]any); ok {
					refs[name] = m
				}
			}
		}
	}
	return refs
}

// sanitizeSchemaNode rewrites one schema node in place. notes accumulates
// normalization hints appended to the node description.
func sanitizeSchemaNode(node map[string]any, refs map[string]map[string]any, depth int, seen map[string]bool) map[string]any {
	notes := []string{}

	// JSON Schema allows $defs at ANY node, not just the root: a nested
	// table also feeds ref resolution for later walk steps.
	for _, tableKey := range []string{"$defs", "definitions"} {
		if table, ok := node[tableKey].(map[string]any); ok {
			for name, def := range table {
				if m, ok := def.(map[string]any); ok {
					if _, exists := refs[name]; !exists {
						refs[name] = m
					}
				}
			}
		}
	}

	// Inline $ref before anything else so composite handling sees the
	// resolved shape.
	if ref, ok := node["$ref"].(string); ok {
		delete(node, "$ref")
		target, ok := resolveRef(ref, refs)
		if !ok || depth >= schemaMaxRefDepth || seen[cycleKey(ref, seen)] {
			node["type"] = "object"
			node["properties"] = map[string]any{}
			node["description"] = fmt.Sprintf("schema reference %s inlined as generic object", ref)
			delete(node, "title")
			return node
		}
		resolved := map[string]any{}
		for k, v := range target {
			resolved[k] = v
		}
		seen[cycleKey(ref, seen)] = true
		for k, v := range node {
			if _, exists := resolved[k]; !exists || k == "description" || k == "required" {
				resolved[k] = v
			}
		}
		return sanitizeSchemaNode(resolved, refs, depth+1, seen)
	}

	// allOf merges shallowly: later members fill keys the earlier ones
	// did not set (property conflicts keep first-wins to stay stable).
	if branches, ok := node["allOf"].([]any); ok {
		delete(node, "allOf")
		for _, branch := range branches {
			bm, ok := branch.(map[string]any)
			if !ok {
				notes = append(notes, "allOf branch not merged")
				continue
			}
			bm = sanitizeSchemaNode(cloneMap(bm), refs, depth+1, seen)
			for k, v := range bm {
				if _, exists := node[k]; !exists {
					node[k] = v
				}
			}
		}
	}

	// anyOf/oneOf collapse. Common cases survive structurally:
	//   - {"anyOf":[X, {"type":"null"}]}  → X with nullable:true
	//   - single representable branch      → inlined verbatim
	//   - multi-type unions                → first type kept, remainder
	//                                        described ("anyOf: ...")
	for _, composite := range []string{"anyOf", "oneOf"} {
		branches, ok := node[composite].([]any)
		if !ok {
			continue
		}
		delete(node, composite)
		var representable []map[string]any
		nullable := false
		notes2 := []string{}
		for _, branch := range branches {
			bm, isMap := branch.(map[string]any)
			if !isMap {
				notes2 = append(notes2, describeJSON(branch))
				continue
			}
			san := sanitizeSchemaNode(cloneMap(bm), refs, depth+1, seen)
			// Null detection must use the PRE-sanitization type: the
			// empty-node fallback below rewrites typeless branches to
			// "string", which would swallow {"type":"null"} otherwise.
			rawType, _ := bm["type"].(string)
			if rawType == "null" {
				nullable = true
				continue
			}
			if t, _ := san["type"].(string); t == "" {
				// Branch without type (only description/enum): still
				// describable; represent it as a string enum-or-text.
				san["type"] = "string"
			}
			representable = append(representable, san)
		}
		if len(representable) == 1 {
			for k, v := range representable[0] {
				if _, exists := node[k]; !exists {
					node[k] = v
				}
			}
		} else if len(representable) > 1 {
			// Keep the simplest branch for the wire; name every dropped
			// one so the model can still see the union's intent.
			first := representable[0]
			for k, v := range first {
				if _, exists := node[k]; !exists {
					node[k] = v
				}
			}
			dropped := make([]string, 0, len(representable)-1)
			for _, other := range representable[1:] {
				dropped = append(dropped, describeSchema(other))
			}
			if len(dropped) > 0 {
				notes = append(notes, composite+" also allows: "+strings.Join(dropped, " | "))
			}
		} else if len(notes2) > 0 {
			notes = append(notes, composite+": "+strings.Join(notes2, " | "))
		}
		if nullable {
			node["nullable"] = true
		}
	}

	// Keywords with no Gemini representation: fold intent into the note.
	for _, dead := range []string{"$schema", "$id", "$comment", "$anchor", "not", "if", "then", "else", "unevaluatedProperties", "unevaluatedItems", "patternProperties", "propertyNames", "contains", "dependencies", "dependentRequired", "dependentSchemas", "minProperties", "maxProperties", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "uniqueItems", "contentEncoding", "contentMediaType", "readOnly", "writeOnly", "deprecated", "$deprecated", "default", "examples", "const"} {
		if v, present := node[dead]; present {
			delete(node, dead)
			if dead == "const" {
				node["enum"] = []any{v}
				notes = append(notes, "const required")
				continue
			}
			if dead == "default" || dead == "examples" {
				notes = append(notes, fmt.Sprintf("%s=%s", dead, describeJSON(v)))
			}
		}
	}

	// Type normalization: array-form ["T","null"] → type T + nullable.
	if types, ok := node["type"].([]any); ok {
		picked := ""
		for _, t := range types {
			ts, _ := t.(string)
			if ts == "null" {
				node["nullable"] = true
				continue
			}
			if picked == "" {
				picked = normalizeTypeName(ts)
			} else if normalizeTypeName(ts) != picked {
				notes = append(notes, "type also allows "+ts)
			}
		}
		if picked != "" {
			node["type"] = picked
		} else {
			delete(node, "type")
		}
	} else if ts, ok := node["type"].(string); ok {
		if parts := strings.Split(ts, "|"); len(parts) > 1 {
			// "string|null" textual unions common in hand-written schemas.
			picked := ""
			for _, p := range parts {
				p = strings.TrimSpace(p)
				if p == "null" {
					node["nullable"] = true
					continue
				}
				if picked == "" {
					picked = normalizeTypeName(p)
				} else if normalizeTypeName(p) != picked {
					notes = append(notes, "type also allows "+p)
				}
			}
			if picked != "" {
				node["type"] = picked
			} else {
				delete(node, "type")
			}
		} else {
			node["type"] = normalizeTypeName(ts)
		}
	}

	// Properties imply object; Gemini rejects typeless declarations.
	if _, hasProps := node["properties"]; hasProps {
		if t, _ := node["type"].(string); t == "" {
			node["type"] = "object"
		}
		if props, ok := node["properties"].(map[string]any); ok {
			for name, prop := range props {
				if pm, ok := prop.(map[string]any); ok {
					props[name] = sanitizeSchemaNode(pm, refs, depth+1, seen)
				} else if sharedHasContent(jsonMarshal(prop)) {
					props[name] = sanitizeSchemaNode(unmarshalMap(prop), refs, depth+1, seen)
				}
			}
		}
	}
	if items, ok := node["items"].(map[string]any); ok {
		node["items"] = sanitizeSchemaNode(items, refs, depth+1, seen)
	} else if itemsList, ok := node["items"].([]any); ok {
		// Tuple schemas: Gemini items is a single schema — keep the
		// broadest (first)branch and describe the rest.
		var first map[string]any
		var rest []string
		for i, it := range itemsList {
			im, ok := it.(map[string]any)
			if !ok {
				continue
			}
			san := sanitizeSchemaNode(cloneMap(im), refs, depth+1, seen)
			if i == 0 {
				first = san
			} else {
				rest = append(rest, describeSchema(san))
			}
		}
		if first != nil {
			node["items"] = first
			if len(rest) > 0 {
				notes = append(notes, "tuple slots also allow: "+strings.Join(rest, " | "))
			}
		} else {
			delete(node, "items")
		}
	}
	if ap, ok := node["additionalProperties"]; ok {
		// This gateway's Schema proto does not accept additionalProperties
		// in any form (bool or nested schema) — drop with a note.
		delete(node, "additionalProperties")
		notes2 := ""
		if apb, isBool := ap.(bool); isBool && apb {
			notes2 = "unconstrained"
		} else if apm, isMap := ap.(map[string]any); isMap {
			notes2 = describeSchema(apm)
		}
		if notes2 != "" {
			notes = append(notes, "extra properties: "+notes2)
		}
	}

	// enum: proto type is array-of-string; coerce non-strings verbatim.
	if en, ok := node["enum"].([]any); ok && len(en) > 0 {
		count := len(en)
		allString := true
		for _, v := range en {
			if _, is := v.(string); !is {
				allString = false
				break
			}
		}
		if !allString {
			str := make([]string, 0, count)
			for _, v := range en {
				str = append(str, describeJSON(v))
			}
			node["enum"] = str
			if t, _ := node["type"].(string); t == "" {
				node["type"] = "string"
			}
			notes = append(notes, "enum normalized to text")
		}
	}

	// format: only the whitelisted values survive the proto validation.
	if f, ok := node["format"].(string); ok && !schemaFormatAllowlist[f] {
		delete(node, "format")
		notes = append(notes, "format "+f)
	}

	// Unused defs tables are unwalked (collectDefs copies terget shapes);
	// nothing to emit — $defs itself never reaches the wire because it is
	// not keepable.
	delete(node, "$defs")
	delete(node, "definitions")

	// Drop unknown keys silently; known-but-unrepresentable ones were
	// handled above.
	for k := range node {
		if !schemaKeepableKeys[k] {
			delete(node, k)
		}
	}

	if len(notes) > 0 {
		note := "(schema normalized: " + strings.Join(notes, "; ") + ")"
		if desc, ok := node["description"].(string); ok && desc != "" {
			node["description"] = desc + "\n" + note
		} else {
			node["description"] = note
		}
	}
	if node["type"] == nil && node["description"] == nil && node["enum"] == nil && node["properties"] == nil {
		// Fully empty sub-node: give it a harmless missing-membership.
		node["type"] = "string"
	}
	return node
}

// resolveRef walks one $ref pointer path against the hoisted def tables.
func resolveRef(ref string, refs map[string]map[string]any) (map[string]any, bool) {
	name, ok := strings.CutPrefix(ref, "#/$defs/")
	if !ok {
		name, ok = strings.CutPrefix(ref, "#/definitions/")
	}
	if !ok {
		name, ok = strings.CutPrefix(ref, "#/defs/")
	}
	if !ok {
		name = strings.TrimPrefix(ref, "#")
		name = strings.TrimPrefix(name, "/")
		name = strings.ReplaceAll(name, "~1", "/")
	}
	target, found := refs[name]
	return target, found
}

func cycleKey(ref string, seen map[string]bool) string {
	return ref
}

func normalizeTypeName(t string) string {
	switch t {
	case "number":
		return "number"
	case "integer":
		return "integer"
	case "string", "boolean", "array", "object":
		return t
	case "":
		return ""
	default:
		return "string"
	}
}

// describeSchema flattens a sanitized branch into description text.
func describeSchema(m map[string]any) string {
	return describeJSON(m)
}

func describeJSON(v any) string {
	b, err := json.Marshal(clipForNotes(v))
	if err != nil {
		return "?"
	}
	s := string(b)
	if len(s) > 120 {
		return s[:120] + "…"
	}
	return s
}

func clipForNotes(v any) any {
	m, ok := v.(map[string]any)
	if !ok {
		return v
	}
	out := map[string]any{}
	for k, val := range m {
		switch k {
		case "description", "title":
			continue
		case "properties":
			if props, ok := val.(map[string]any); ok {
				out["properties"] = props
			}
		default:
			out[k] = val
		}
	}
	return out
}

func cloneMap(src map[string]any) map[string]any {
	out := make(map[string]any, len(src))
	for k, v := range src {
		out[k] = cloneAny(v)
	}
	return out
}

func cloneAny(v any) any {
	switch t := v.(type) {
	case map[string]any:
		return cloneMap(t)
	case []any:
		out := make([]any, len(t))
		for i, e := range t {
			out[i] = cloneAny(e)
		}
		return out
	default:
		return v
	}
}

func jsonMarshal(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return b
}

func unmarshalMap(raw any) map[string]any {
	b := jsonMarshal(raw)
	var m map[string]any
	if json.Unmarshal(b, &m) != nil {
		return map[string]any{}
	}
	return m
}
