// Package catalog fetches the live OpenCode Go /v1/models catalog,
// normalizes metadata, assigns routes, and serves the last good snapshot
// (FR-002, FR-003, FR-004, FR-010).
//
// Route resolution priority follows arch spec §5: user route override >
// built-in compatibility table > unsupported (excluded from the routable set,
// kept in diagnostics).
//
// No scheduling lives here; M3 wires the refresh cadence.
package catalog

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"

	"opencode-go-cliproxyapi/internal/config"
)

// catalogBudgetFloor keeps an undersized max-response-bytes knob from
// freezing stale serving forever: catalogs grow larger than typical
// completion bodies, so the fetch budget never drops below 16 MiB while
// max-response-bytes stays meaningful for completions elsewhere.
const catalogBudgetFloor = 16 << 20

// Route is the upstream protocol a model is served through (FR-004).
type Route string

const (
	RouteChatCompletions Route = "chat-completions"
	RouteMessages        Route = "messages"
	RouteResponses       Route = "responses"
	RouteGemini          Route = "gemini"
)

// EndpointPath returns the upstream endpoint path for the route (FR-004).
// routeEndpoints maps each route to its upstream endpoint path; RouteGemini
// has NO static endpoint — its paths are model-specific
// (…/models/<model>:generateContent) and are stamped per record in swap.
var routeEndpoints = map[Route]string{
	RouteChatCompletions: "/v1/chat/completions",
	RouteMessages:        "/v1/messages",
	RouteResponses:       "/v1/responses",
}

// geminiActionSuffixes are the model-dependent Gemini endpoint actions.
const (
	geminiGenerateSuffix = ":generateContent"
	geminiStreamSuffix   = ":streamGenerateContent?alt=sse"
)

// GeminiEndpointPath returns the non-streaming Gemini upstream endpoint
// path for one model id: /v1/models/<id>:generateContent. The leading
// /v1 segment is dropped by JoinUpstreamURL exactly like the static
// routes, so the configured base-url's version segment is never doubled.
func GeminiEndpointPath(modelID string) string {
	return "/v1/models/" + modelID + geminiGenerateSuffix
}

// EndpointPath returns the route's upstream endpoint path, or "" for
// routes without one (RouteGemini).
func (r Route) EndpointPath() string {
	return routeEndpoints[r]
}

func routeFromString(s string) (Route, bool) {
	if _, ok := routeEndpoints[Route(s)]; ok {
		return Route(s), true
	}
	return "", false
}

// ModelRecord is one routable normalized catalog entry (arch §7, FR-003).
type ModelRecord struct {
	PublicID     string
	UpstreamID   string
	DisplayName  string
	Protocol     Route
	EndpointPath string
	ContextLimit int64
	OutputLimit  int64
	InputModes   []string
	OutputModes  []string
	Thinking     *pluginapi.ThinkingSupport
}

// UnsupportedModel is a discovered model excluded from the routable set,
// retained for diagnostics (FR-004, open questions §10).
type UnsupportedModel struct {
	UpstreamID string
	Reason     string
}

// HostClient issues HTTP requests through the host (pluginapi.HTTPDo
// semantics); M3's host callback satisfies it directly.
type HostClient interface {
	Do(ctx context.Context, req pluginapi.HTTPRequest) (pluginapi.HTTPResponse, error)
}

// rawModel tolerantly decodes one catalog entry; unknown fields —
// including protocol/endpoint fields the catalog does not provide and
// reasoning/tool_calling flags the plugin does not act on — are ignored via
// json tags. Modality metadata accepts both spellings.
type rawModel struct {
	ID               string       `json:"id"`
	DisplayName      string       `json:"display_name"`
	ContextLength    int64        `json:"context_length"`
	MaxOutputTokens  int64        `json:"max_output_tokens"`
	InputModes       []string     `json:"input_modes"`
	OutputModes      []string     `json:"output_modes"`
	InputModalities  []string     `json:"input_modalities"`
	OutputModalities []string     `json:"output_modalities"`
	Thinking         *rawThinking `json:"thinking"`
}

// rawThinking tolerantly decodes the optional per-model thinking object;
// pluginapi.ThinkingSupport itself is untagged, so snake_case upstream keys
// are decoded here and mapped field-by-field (partial objects decode to a
// partial struct).
type rawThinking struct {
	Min            *int     `json:"min"`
	Max            *int     `json:"max"`
	ZeroAllowed    *bool    `json:"zero_allowed"`
	DynamicAllowed *bool    `json:"dynamic_allowed"`
	Levels         []string `json:"levels"`
}

// prefixRoutes covers unknown variants of known families;
// longest prefix wins (list is ordered longest-first).
var prefixRoutes = []struct {
	prefix string
	route  Route
}{
	{"muse-spark", RouteResponses},
	{"deepseek", RouteChatCompletions},
	{"minimax", RouteMessages},
	{"longcat", RouteChatCompletions},
	{"grok", RouteResponses},
	{"gpt", RouteResponses},
	{"kimi", RouteChatCompletions},
	{"qwen", RouteMessages},
	{"glm", RouteChatCompletions},
	{"gemini", RouteGemini},
	{"mimo", RouteChatCompletions},
	{"hy", RouteChatCompletions},
}

// Manager owns the catalog snapshot. All accessors are safe for
// concurrent use alongside Refresh.
type Manager struct {
	cfg    config.Config
	client HostClient

	mu sync.Mutex
	// raw retains the decoded upstream entries behind the current snapshot
	// so SeedFrom can rebuild records against a NEW config (route overrides,
	// protocol flags, and prefix settings may all have changed since the
	// snapshot was built).
	raw    []rawModel
	models []ModelRecord
	// index maps both PublicID and UpstreamID to their record so ID
	// resolution is O(1); rebuilt atomically with models on every swap.
	index map[string]ModelRecord
	unsup []UnsupportedModel
	warns []string
}

// New returns a Manager serving cfg through the host client.
func New(cfg config.Config, client HostClient) *Manager {
	return &Manager{cfg: cfg, client: client}
}

// SeedFrom republishes prev's last-good snapshot into m by REBUILDING every
// record from prev's raw upstream entries through the same
// resolution/validation path a refresh applies, but against m's OWN cfg:
// routes re-resolve (a removed override drops its model), the protocolEnabled
// gate re-applies, PublicIDs recompute under m's prefix, endpoints re-check
// against m's base-url, and collisions/diagnostics regenerate (arch §5).
//
// m keeps its OWN cfg: subsequent Refresh calls use m's base-url/catalog-url/
// protocols/prefix (F5: a failed-refresh reconfigure must adopt the new
// manager rather than keep ticking the old URL). Because seeding rebuilds
// instead of copying, a config change between snapshots takes effect at seed
// time — e.g. a protocol kill-switch flipped during a catalog outage demotes
// its routes immediately instead of serving stale routable records until the
// outage ends.
func (m *Manager) SeedFrom(prev *Manager) {
	prev.mu.Lock()
	raw := prev.raw
	prev.mu.Unlock()
	m.swap(raw)
}

// Refresh fetches and swaps the catalog snapshot. On failure it returns a
// classified, key-free error; the previous snapshot keeps serving while
// catalog.stale-while-unavailable is enabled, otherwise the routable set
// is cleared until the next success (FR-002).
func (m *Manager) Refresh(ctx context.Context, apiKey string) error {
	if m.client == nil {
		// Only direct construction can produce a nil host client
		// (production always wires the bridge); fail loudly through the
		// classified path instead of panicking inside Do.
		return m.fail("host client unavailable")
	}
	req := pluginapi.HTTPRequest{
		Method: http.MethodGet,
		URL:    m.cfg.CatalogURL,
		Headers: http.Header{
			"Authorization": []string{"Bearer " + apiKey},
			"Accept":        []string{"application/json"},
		},
	}
	resp, err := m.client.Do(ctx, req)
	if err != nil {
		return m.fail("network error")
	}
	budget := max(m.cfg.MaxResponseBytes, catalogBudgetFloor)
	if int64(len(resp.Body)) > budget {
		return m.fail("response exceeds max-response-bytes")
	}
	if resp.StatusCode != http.StatusOK {
		return m.fail(fmt.Sprintf("http %d", resp.StatusCode))
	}
	var env struct {
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(resp.Body, &env); err != nil {
		return m.fail("invalid json")
	}
	var entries []rawModel
	var warns []string
	if env.Data == nil {
		// Key ABSENCE is upstream shape drift, never an intended clear:
		// clear like an empty plan (swap(nil)) but leave a diagnostic.
		// A present-but-empty array stays silent (tested-intended state).
		warns = append(warns, `upstream catalog response missing "data" field`)
	} else if err := json.Unmarshal(env.Data, &entries); err != nil {
		return m.fail("invalid json")
	}
	m.swap(entries, warns...)
	return nil
}

// fail applies the stale policy and returns the classified error.
func (m *Manager) fail(category string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.cfg.Catalog.StaleWhileUnavailable {
		// Clear the index too — Lookup must stop resolving IDs whose
		// records are gone (FR-002). Raw entries go with them: a cleared
		// snapshot has nothing to seed.
		m.raw, m.models, m.index, m.unsup, m.warns = nil, nil, nil, nil, nil
	}
	return fmt.Errorf("catalog refresh failed: %s", category)
}

// swap builds the new snapshot from decoded entries and installs it
// atomically under the mutex (FR-010 dedup, arch §5 route priority).
// extraWarns are caller-supplied snapshot diagnostics (e.g. decode-level
// shape-drift notices) recorded alongside the per-entry ones.
func (m *Manager) swap(entries []rawModel, extraWarns ...string) {
	models := make([]ModelRecord, 0, len(entries))
	index := make(map[string]ModelRecord, len(entries)*2)
	var unsup []UnsupportedModel
	var warns []string
	warns = append(warns, extraWarns...)
	seen := make(map[string]bool, len(entries))
	for _, e := range entries {
		if e.ID == "" {
			unsup = append(unsup, UnsupportedModel{Reason: "missing id"})
			continue
		}
		if seen[e.ID] {
			warns = append(warns, fmt.Sprintf("duplicate model %q ignored; keeping first occurrence", e.ID))
			continue
		}
		seen[e.ID] = true
		route, endpoint, ok, warn := m.resolve(e)
		if warn != "" {
			warns = append(warns, warn)
		}
		if !ok {
			unsup = append(unsup, UnsupportedModel{UpstreamID: e.ID, Reason: "no route determined"})
			continue
		}
		// Model-name mapping: an alias publishes under its own ID but
		// executes against the mapped upstream ID. Unless the alias has an
		// explicit route override (which always wins), the route re-resolves
		// from the mapped ID so "my-model" -> "glm-5.2" lands on the glm
		// family's protocol even though the alias matches no prefix.
		upstreamID := e.ID
		mapped := config.MapUpstreamID(m.cfg, e.ID)
		if mapped != e.ID {
			upstreamID = mapped
			if o, hasOverride := m.cfg.RouteOverrides[e.ID]; hasOverride {
				route, endpoint = Route(o.Protocol), o.Endpoint
				warn = fmt.Sprintf("model %q routed via user override (%s %s)", e.ID, o.Protocol, o.Endpoint)
			} else if tRoute, tEndpoint, tok, twarn := m.resolve(rawModel{ID: mapped}); tok {
				route, endpoint, warn = tRoute, tEndpoint, twarn
			}
		}
		if !m.protocolEnabled(route) {
			// Flags apply at snapshot build; a stale snapshot kept per
			// FR-002 reflects the flags it was built with until the next
			// successful refresh. Checked AFTER mapping so a redirected
			// alias demotes with its TARGET's protocol.
			unsup = append(unsup, UnsupportedModel{
				UpstreamID: e.ID,
				Reason:     fmt.Sprintf("protocol %s disabled by config", route),
			})
			continue
		}
		// The Gemini route has no static endpoint: its path embeds the
		// upstream model id (…/models/<id>:generateContent), stamped against
		// the final upstream ID so aliases execute on the target's endpoint.
		// User overrides keep their own endpoint verbatim.
		if endpoint == "" {
			if route == RouteGemini {
				endpoint = GeminiEndpointPath(upstreamID)
			} else {
				endpoint = route.EndpointPath()
			}
		}
		if endpointEscapesBase(m.cfg.BaseURL, endpoint) {
			// The joined URL must stay on the configured upstream
			// authority. Override endpoints and the generated Gemini paths
			// flow through here, so neither can rewrite the URL host —
			// e.g. "/v1@evil.com/x" against a bare-host base-url turns the
			// base into userinfo and would send credentials/prompts to an
			// attacker host (spec 05 §2 trust boundary).
			unsup = append(unsup, UnsupportedModel{
				UpstreamID: e.ID,
				Reason:     "resolved endpoint escapes the configured upstream host",
			})
			continue
		}
		inputModes := firstNonEmpty(e.InputModes, e.InputModalities)
		outputModes := firstNonEmpty(e.OutputModes, e.OutputModalities)
		display := e.DisplayName
		if display == "" {
			display = e.ID
		}
		rec := ModelRecord{
			PublicID:     config.PublicID(m.cfg, e.ID),
			UpstreamID:   upstreamID,
			DisplayName:  display,
			Protocol:     route,
			EndpointPath: endpoint,
			ContextLimit: e.ContextLength,
			OutputLimit:  e.MaxOutputTokens,
			InputModes:   inputModes,
			OutputModes:  outputModes,
			Thinking:     normalizeThinking(e.Thinking),
		}
		// With a prefix enabled, one record's PublicID can equal another
		// record's UpstreamID (upstream "foo" and "opencode-go/foo" both
		// claim index key "<prefix>/foo"); last-write-wins would silently
		// misroute. First in catalog order wins (same dedup rule as
		// duplicate IDs); the later record is excluded. Same-role clashes
		// are impossible — distinct upstream IDs and an injective public
		// mapping — so the slot owner always holds the key via its other
		// role, and a record's own public==upstream rewrite is not a clash.
		if other, ok := index[rec.PublicID]; ok {
			reason := fmt.Sprintf("public id %q collides with upstream id %q", rec.PublicID, other.UpstreamID)
			warns = append(warns, reason)
			unsup = append(unsup, UnsupportedModel{UpstreamID: e.ID, Reason: reason})
			continue
		}
		if other, ok := index[rec.UpstreamID]; ok {
			reason := fmt.Sprintf("public id %q collides with upstream id %q", other.PublicID, rec.UpstreamID)
			warns = append(warns, reason)
			unsup = append(unsup, UnsupportedModel{UpstreamID: e.ID, Reason: reason})
			continue
		}
		models = append(models, rec)
		index[rec.PublicID] = rec
		index[rec.UpstreamID] = rec
	}

	// Pass 2 — mapping-only aliases: a model-mappings entry whose key was
	// never seen in the catalog registers a SYNTHETIC model. The alias is
	// the only client-facing name (unless the target was independently
	// discovered), and its route resolves from the mapped target on the
	// same precedence ladder (override > family prefix > unroutable), so
	// off-catalog models - e.g. an upstream that stops listing a family -
	// stay configurable purely through model-mappings. Sorted keys keep
	// the snapshot deterministic (map iteration is not).
	aliases := make([]string, 0, len(m.cfg.ModelMappings))
	for a := range m.cfg.ModelMappings {
		if !seen[a] {
			aliases = append(aliases, a)
		}
	}
	sort.Strings(aliases)
	for _, alias := range aliases {
		target := m.cfg.ModelMappings[alias].To
		var route Route
		var endpoint string
		var ok bool
		var warn string
		if o, hasOverride := m.cfg.RouteOverrides[alias]; hasOverride {
			// The alias's own override outranks the target's family route,
			// matching the pass-1 precedence.
			route, endpoint, ok, warn = Route(o.Protocol), o.Endpoint, true,
				fmt.Sprintf("mapped model %q routed via user override (%s %s)", alias, o.Protocol, o.Endpoint)
		} else {
			route, endpoint, ok, warn = m.resolve(rawModel{ID: target})
		}
		if ok {
			if warn != "" {
				warns = append(warns, fmt.Sprintf("mapped model %q: %s", alias, warn))
			}
			if !m.protocolEnabled(route) {
				unsup = append(unsup, UnsupportedModel{
					UpstreamID: alias,
					Reason:     fmt.Sprintf("protocol %s disabled by config", route),
				})
				continue
			}
			if endpoint == "" {
				if route == RouteGemini {
					endpoint = GeminiEndpointPath(target)
				} else {
					endpoint = route.EndpointPath()
				}
			}
			if endpointEscapesBase(m.cfg.BaseURL, endpoint) {
				unsup = append(unsup, UnsupportedModel{
					UpstreamID: alias,
					Reason:     "resolved endpoint escapes the configured upstream host",
				})
				continue
			}
			rec := ModelRecord{
				PublicID:     config.PublicID(m.cfg, alias),
				UpstreamID:   target,
				DisplayName:  alias,
				Protocol:     route,
				EndpointPath: endpoint,
			}
			// The alias takes BOTH its client-facing keys (prefixed public
			// id and bare alias); the mapped target itself gains no new
			// index key — with no catalog entry behind it, the target name
			// only resolves through its alias.
			planted := false
			for _, key := range []string{rec.PublicID, alias} {
				if other, exists := index[key]; exists && other.UpstreamID != rec.UpstreamID {
					reason := fmt.Sprintf("mapped model %q: client id %q already resolves to %q", alias, key, other.PublicID)
					warns = append(warns, reason)
					unsup = append(unsup, UnsupportedModel{UpstreamID: alias, Reason: reason})
					planted = true
					break
				}
			}
			if planted {
				continue
			}
			models = append(models, rec)
			index[rec.PublicID] = rec
			if alias != rec.PublicID {
				index[alias] = rec
			}
			continue
		}
		reason := fmt.Sprintf("mapped model %q: target %q has no route determined", alias, target)
		warns = append(warns, reason)
		unsup = append(unsup, UnsupportedModel{UpstreamID: alias, Reason: reason})
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.raw, m.models, m.index, m.unsup, m.warns = entries, models, index, unsup, warns
}

// protocolEnabled reports whether the resolved route's protocol flag is on
// (spec 04 §4); flags default to true.
func (m *Manager) protocolEnabled(r Route) bool {
	switch r {
	case RouteChatCompletions:
		return m.cfg.Protocols.ChatCompletions
	case RouteMessages:
		return m.cfg.Protocols.Messages
	case RouteResponses:
		return m.cfg.Protocols.Responses
	case RouteGemini:
		return m.cfg.Protocols.Gemini
	}
	return true
}

// resolve applies route priority strictly: user override, compatibility
// table (family prefix), none (arch §5).
func (m *Manager) resolve(e rawModel) (Route, string, bool, string) {
	if o, ok := m.cfg.RouteOverrides[e.ID]; ok {
		// Spec 04 §5: an override MUST be visible in diagnostics. Load
		// validation (spec 04 §6) guarantees a known protocol, so the
		// map lookup above is the only decision point here.
		return Route(o.Protocol), o.Endpoint, true, fmt.Sprintf(
			"model %q routed via user override (%s %s)", e.ID, o.Protocol, o.Endpoint)
	}
	id := strings.ToLower(e.ID)
	for _, p := range prefixRoutes {
		if strings.HasPrefix(id, p.prefix) {
			return p.route, "", true, ""
		}
	}
	return "", "", false, ""
}

// JoinUpstreamURL joins the configured base-url onto an endpoint path and
// is the single source of truth for that join (spec 05 §2): base-url
// already carries the version segment (default .../zen/go/v1, catalog at
// {base}/models), so a leading "/v1" on the endpoint is dropped instead of
// doubled. The authority-escape check (endpointEscapesBase) must parse
// exactly the URL this produces — two independent join formulas would be
// free to drift apart and validate one URL while sending another.
func JoinUpstreamURL(baseURL, endpoint string) string {
	return strings.TrimSuffix(baseURL, "/") + strings.TrimPrefix(endpoint, "/v1")
}

// endpointEscapesBase reports whether joining baseURL with endpoint via
// JoinUpstreamURL would send the request to a different authority or scheme
// than the configured base-url. Any parse failure counts as escaping (fail
// closed).
func endpointEscapesBase(baseURL, endpoint string) bool {
	base, err := url.Parse(baseURL)
	if err != nil {
		return true
	}
	got, err := url.Parse(JoinUpstreamURL(baseURL, endpoint))
	if err != nil {
		return true
	}
	return !strings.EqualFold(got.Host, base.Host) || got.Scheme != base.Scheme
}

func firstNonEmpty(a, b []string) []string {
	if len(a) > 0 {
		return a
	}
	return b
}

// normalizeThinking converts the decoded raw thinking object into its
// pluginapi form; absent objects yield nil, partial ones a partial struct.
func normalizeThinking(rt *rawThinking) *pluginapi.ThinkingSupport {
	if rt == nil {
		return nil
	}
	t := &pluginapi.ThinkingSupport{}
	if rt.Min != nil {
		t.Min = *rt.Min
	}
	if rt.Max != nil {
		t.Max = *rt.Max
	}
	if rt.ZeroAllowed != nil {
		t.ZeroAllowed = *rt.ZeroAllowed
	}
	if rt.DynamicAllowed != nil {
		t.DynamicAllowed = *rt.DynamicAllowed
	}
	if len(rt.Levels) > 0 {
		t.Levels = append([]string(nil), rt.Levels...)
	}
	return t
}

// Models returns the routable snapshot (FR-003 identity fields). The slice
// is a shallow copy; records are rebuilt fresh on every swap and consumers
// treat them as read-only, so nested slices/thinking are shared safely.
func (m *Manager) Models() []ModelRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]ModelRecord, len(m.models))
	copy(out, m.models)
	return out
}

// Lookup resolves an ID to its routable record, accepting the public ID
// (with or without prefix) or the bare upstream ID (FR-003): the PublicID
// key wins over UpstreamID when both exist.
func (m *Manager) Lookup(id string) (ModelRecord, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	rec, ok := m.index[id]
	return rec, ok
}

// Unsupported returns models excluded from the routable set with reasons.
func (m *Manager) Unsupported() []UnsupportedModel {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]UnsupportedModel, len(m.unsup))
	copy(out, m.unsup)
	return out
}

// Warnings returns dedup and override diagnostics.
func (m *Manager) Warnings() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, len(m.warns))
	copy(out, m.warns)
	return out
}
