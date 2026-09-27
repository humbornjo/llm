package providers

import (
	"encoding/json"
	"testing"
)

func TestProviders_MessageContentJSONRoundTrip(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		in   Message
		want string
	}{
		{
			name: "text",
			in:   Message{Role: ROLE_USER, Content: ContentFromString("hello \"world\"")},
			want: `{"role":"user","content":"hello \"world\""}`,
		},
		{
			name: "parts",
			in: Message{Role: ROLE_USER, Content: ContentFromParts(
				&ContentPartText{Text: "hello"},
				&ContentPartImage{ImageURL: &ImageURL{URL: "https://example.com/a.png", Detail: "high"}},
				&ContentPartAudio{InputAudio: &InputAudio{Data: "AQI=", Format: "wav"}},
				&ContentPartFile{File: &File{FileId: "file-1"}},
			)},
			want: `{"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"https://example.com/a.png","detail":"high"}},{"type":"input_audio","input_audio":{"data":"AQI=","format":"wav"}},{"type":"file","file":{"file_id":"file-1"}}]}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded, err := json.Marshal(tc.in)
			if err != nil {
				t.Fatal(err)
			}
			if string(encoded) != tc.want {
				t.Fatalf("encoded content = %s, want %s", encoded, tc.want)
			}

			var got Message
			if err := json.Unmarshal(encoded, &got); err != nil {
				t.Fatal(err)
			}
			if got.Role != tc.in.Role || got.ContentString() != tc.in.ContentString() {
				t.Fatalf("round trip message = %#v, want %#v", got, tc.in)
			}
			if len(got.ContentParts()) != len(tc.in.ContentParts()) {
				t.Fatalf("round trip parts = %d, want %d", len(got.ContentParts()), len(tc.in.ContentParts()))
			}
		})
	}
}

func TestProviders_MessageContentJSONRejectsMalformedContent(t *testing.T) {
	t.Parallel()

	cases := []string{
		`{"role":"user","content":null}`,
		`{"role":"user","content":42}`,
		`{"role":"user","content":[]}`,
		`{"role":"user","content":[{"type":"unknown"}]}`,
		`{"role":"user","content":[{"text":"missing type"}]}`,
		`{"role":"user","content":[{"type":"text"}]}`,
		`{"role":"user","content":[{"type":"text","text":"ok","extra":true}]}`,
		`{"role":"user","content":"ok","extra":true}`,
		`{"role":"user","content":"ok"} trailing`,
	}
	for _, input := range cases {
		t.Run(input, func(t *testing.T) {
			var message Message
			if err := json.Unmarshal([]byte(input), &message); err == nil {
				t.Fatalf("expected malformed content to fail: %s", input)
			}
		})
	}
}

func TestProviders_ContentPartsJSONRejectsNilPart(t *testing.T) {
	t.Parallel()

	parts := ContentParts{nil}
	if _, err := json.Marshal(&parts); err == nil {
		t.Fatal("expected nil content part to fail")
	}

	var decoded ContentParts
	if err := json.Unmarshal([]byte(`[null]`), &decoded); err == nil {
		t.Fatal("expected null content part to fail")
	}
}
