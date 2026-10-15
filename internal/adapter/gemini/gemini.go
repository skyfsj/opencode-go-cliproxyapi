// Package gemini implements the Gemini generateContent protocol adapter
// (arch §8 pattern, route gemini): request translation (FR-005), non-stream
// response conversion (FR-006), and SSE stream conversion (AC §B) for
// models routed to the Gemini wire format.
//
// Upstream endpoints are MODEL-DEPENDENT (…/models/<model>:generateContent
// and …/models/<model>:streamGenerateContent?alt=sse), unlike the sibling
// OpenAI/Anthropic routes: catalog stamps the non-stream path into each
// record and GeminiStreamEndpointPath derives the streaming path from the
// same upstream id.
package gemini

import (
	"net/http"
	"strings"

	"opencode-go-cliproxyapi/internal/catalog"
)

// StreamQuerySuffix is the alt=sse query that switches streamGenerateContent
// to SSE framing; without it the endpoint answers with JSON array chunks.
const streamQuerySuffix = "?alt=sse"

// GeminiStreamEndpointPath returns the streaming endpoint path for one
// upstream model id: /v1/models/<id>:streamGenerateContent?alt=sse.
func GeminiStreamEndpointPath(modelID string) string {
	return "/v1/models/" + modelID + ":streamGenerateContent" + streamQuerySuffix
}

// UpstreamPath returns the upstream path holding the model id from either
// Gemini endpoint variant; "" when path is not a Gemini endpoint.
func UpstreamPath(path string) (string, bool) {
	rest, ok := strings.CutPrefix(path, "/v1/models/")
	if !ok {
		return "", false
	}
	if model, ok := strings.CutSuffix(rest, ":streamGenerateContent"+streamQuerySuffix); ok {
		return model, true
	}
	model, ok := strings.CutSuffix(rest, ":generateContent")
	return model, ok
}

// AuthHeaders returns the FR-007 authorization headers for an OpenCode Go
// key on the Gemini route (same bearer auth as the sibling OpenAI-style
// routes; the executor wires x-opencode-session on top).
func AuthHeaders(key string) http.Header {
	h := http.Header{}
	h.Set("Authorization", "Bearer "+key)
	return h
}

// Route is the catalog route served by this adapter (kept local so the
// adapter does not depend on anything beyond the route constant).
var Route = catalog.RouteGemini

// TargetLabels names this adapter in client-facing diagnostics.
const targetLabel = "gemini generateContent"
