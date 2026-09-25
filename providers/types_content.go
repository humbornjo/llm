package providers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// ContentFromString creates scalar text message content.
func ContentFromString(text string) Content {
	return Content{value: new(ContentString(text))}
}

// ContentFromParts creates multimodal message content from the given
// parts, which must not be empty.
func ContentFromParts(parts ...IsContentPart) Content {
	wrapped := make(ContentParts, 0, len(parts))
	for _, part := range parts {
		wrapped = append(wrapped, ContentPart{value: part})
	}
	return Content{value: &wrapped}
}

// Content is either scalar text or a non-empty list of typed content
// parts. The zero value carries no content.
type Content struct {
	value IsContent
}

// IsContent is the sealed union of message content variants: scalar text
// (ContentString) or a non-empty list of typed parts (ContentParts).
type IsContent interface {
	isContent()
	GetType() ContentType
}

var (
	_ IsContent = (*ContentString)(nil)
	_ IsContent = (*ContentParts)(nil)
)

// Unwrap returns the underlying content variant, or nil for the zero
// value.
func (c Content) Unwrap() IsContent {
	return c.value
}

// UnmarshalJSON decodes content from either a JSON string or an array of
// typed content parts.
func (c *Content) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return errors.New("content is required")
	}
	switch trimmed[0] {
	case '"':
		var text string
		if err := unmarshalStrict(data, &text); err != nil {
			return err
		}
		c.value = new(ContentString(text))
		return nil
	case '[':
		var parts ContentParts
		if err := json.Unmarshal(data, &parts); err != nil {
			return err
		}
		c.value = &parts
		return nil
	default:
		return errors.New("content must be a string or array")
	}
}

type ContentType string

const (
	CONTENT_TYPE_STRING ContentType = "string"
	CONTENT_TYPE_PARTS  ContentType = "content_parts"
)

// ContentString is scalar text message content.
type ContentString string

func (*ContentString) GetType() ContentType {
	return CONTENT_TYPE_STRING
}

func (*ContentString) isContent() {}

// ContentParts is multimodal message content.
type ContentParts []ContentPart

func (*ContentParts) GetType() ContentType {
	return CONTENT_TYPE_PARTS
}

func (*ContentParts) isContent() {}

// ContentPart is one typed part of multimodal message content. The zero
// value carries no part.
type ContentPart struct {
	value IsContentPart
}

// IsContentPart is the sealed union of content part variants: text
// (ContentPartText), image URL (ContentPartImage), input audio
// (ContentPartAudio), or file (ContentPartFile).
type IsContentPart interface {
	isContentPart()
	GetType() ContentPartType
}

var (
	_ IsContentPart = (*ContentPartText)(nil)
	_ IsContentPart = (*ContentPartImage)(nil)
	_ IsContentPart = (*ContentPartAudio)(nil)
	_ IsContentPart = (*ContentPartFile)(nil)
)

// Unwrap returns the underlying part variant, or nil for the zero value.
func (p ContentPart) Unwrap() IsContentPart {
	return p.value
}

// ContentPartType is the JSON discriminator for a content part.
type ContentPartType string

const (
	CONTENT_PART_TEXT        ContentPartType = "text"
	CONTENT_PART_FILE        ContentPartType = "file"
	CONTENT_PART_IMAGE_URL   ContentPartType = "image_url"
	CONTENT_PART_INPUT_AUDIO ContentPartType = "input_audio"
)

// ContentPartText is a text content part.
type ContentPartText struct {
	Text string
}

func (*ContentPartText) isContentPart() {}

func (*ContentPartText) GetType() ContentPartType {
	return CONTENT_PART_TEXT
}

// ContentPartImage is an image URL content part.
type ContentPartImage struct {
	ImageURL *ImageURL
}

func (*ContentPartImage) isContentPart() {}

func (*ContentPartImage) GetType() ContentPartType {
	return CONTENT_PART_IMAGE_URL
}

// ContentPartAudio is an input-audio content part.
type ContentPartAudio struct {
	InputAudio *InputAudio `json:"input_audio"`
}

func (*ContentPartAudio) isContentPart() {}

func (*ContentPartAudio) GetType() ContentPartType {
	return CONTENT_PART_INPUT_AUDIO
}

// ContentPartFile is a file content part.
type ContentPartFile struct {
	File *File
}

func (*ContentPartFile) isContentPart() {}

func (*ContentPartFile) GetType() ContentPartType {
	return CONTENT_PART_FILE
}

func (p *ContentPartText) MarshalJSON() ([]byte, error) {
	if p == nil {
		return nil, errors.New("nil text content part")
	}
	text, err := json.Marshal(p.Text)
	if err != nil {
		return nil, err
	}
	return []byte(`{"type":"text","text":` + string(text) + `}`), nil
}

func (p *ContentPartImage) MarshalJSON() ([]byte, error) {
	if p == nil || p.ImageURL == nil {
		return nil, errors.New("image content part requires image_url")
	}
	image, err := json.Marshal(p.ImageURL)
	if err != nil {
		return nil, err
	}
	return []byte(`{"type":"image_url","image_url":` + string(image) + `}`), nil
}

func (p *ContentPartAudio) MarshalJSON() ([]byte, error) {
	if p == nil || p.InputAudio == nil {
		return nil, errors.New("audio content part requires input_audio")
	}
	audio, err := json.Marshal(p.InputAudio)
	if err != nil {
		return nil, err
	}
	return []byte(`{"type":"input_audio","input_audio":` + string(audio) + `}`), nil
}

func (p *ContentPartFile) MarshalJSON() ([]byte, error) {
	if p == nil || p.File == nil {
		return nil, errors.New("file content part requires file")
	}
	file, err := json.Marshal(p.File)
	if err != nil {
		return nil, err
	}
	return []byte(`{"type":"file","file":` + string(file) + `}`), nil
}

// MarshalJSON encodes the part with its type discriminator.
func (p ContentPart) MarshalJSON() ([]byte, error) {
	if p.value == nil {
		return nil, errors.New("content part is required")
	}
	return json.Marshal(p.value)
}

// UnmarshalJSON decodes one content part, discriminated by its type
// field.
func (p *ContentPart) UnmarshalJSON(data []byte) error {
	var discriminator struct {
		Type *ContentPartType `json:"type"`
	}
	if err := json.Unmarshal(data, &discriminator); err != nil {
		return err
	}
	if discriminator.Type == nil || *discriminator.Type == "" {
		return errors.New("type is required")
	}

	switch *discriminator.Type {
	case CONTENT_PART_TEXT:
		var value struct {
			Type ContentPartType `json:"type"`
			Text *string         `json:"text"`
		}
		if err := unmarshalStrict(data, &value); err != nil {
			return err
		}
		if value.Type != CONTENT_PART_TEXT || value.Text == nil {
			return errors.New("text is required")
		}
		p.value = &ContentPartText{Text: *value.Text}
	case CONTENT_PART_IMAGE_URL:
		var value struct {
			Type     ContentPartType `json:"type"`
			ImageURL *ImageURL       `json:"image_url"`
		}
		if err := unmarshalStrict(data, &value); err != nil {
			return err
		}
		if value.Type != CONTENT_PART_IMAGE_URL || value.ImageURL == nil {
			return errors.New("image_url is required")
		}
		p.value = &ContentPartImage{ImageURL: value.ImageURL}
	case CONTENT_PART_INPUT_AUDIO:
		var value struct {
			Type       ContentPartType `json:"type"`
			InputAudio *InputAudio     `json:"input_audio"`
		}
		if err := unmarshalStrict(data, &value); err != nil {
			return err
		}
		if value.Type != CONTENT_PART_INPUT_AUDIO || value.InputAudio == nil {
			return errors.New("input_audio is required")
		}
		p.value = &ContentPartAudio{InputAudio: value.InputAudio}
	case CONTENT_PART_FILE:
		var value struct {
			Type ContentPartType `json:"type"`
			File *File           `json:"file"`
		}
		if err := unmarshalStrict(data, &value); err != nil {
			return err
		}
		if value.Type != CONTENT_PART_FILE || value.File == nil {
			return errors.New("file is required")
		}
		p.value = &ContentPartFile{File: value.File}
	default:
		return fmt.Errorf("unknown type %q", *discriminator.Type)
	}
	return nil
}

func (p *ContentParts) MarshalJSON() ([]byte, error) {
	if p == nil || len(*p) == 0 {
		return nil, errors.New("content parts must not be empty")
	}
	for _, part := range *p {
		if part.Unwrap() == nil {
			return nil, errors.New("content part must not be nil")
		}
	}
	return json.Marshal([]ContentPart(*p))
}

func (p *ContentParts) UnmarshalJSON(data []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("content parts must be an array: %w", err)
	}
	if len(raw) == 0 {
		return errors.New("content parts must not be empty")
	}

	parts := make(ContentParts, len(raw))
	for i, item := range raw {
		if err := json.Unmarshal(item, &parts[i]); err != nil {
			return fmt.Errorf("content part %d is invalid: %w", i, err)
		}
	}
	*p = parts
	return nil
}

func unmarshalStrict(data []byte, value any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("trailing JSON value")
		}
		return err
	}
	return nil
}

func (m Message) MarshalJSON() ([]byte, error) {
	// A message that only carries tool calls has no content; the
	// OpenAI wire form is an explicit null, and strict providers
	// reject an empty string or empty parts list. Any other
	// content-less message is a caller bug.
	if m.Content.Unwrap() == nil && len(m.ToolCalls) == 0 {
		return nil, errors.New("message content is required")
	}
	content := []byte("null")
	if m.Content.Unwrap() != nil {
		var err error
		content, err = json.Marshal(m.Content.Unwrap())
		if err != nil {
			return nil, fmt.Errorf("message content is invalid: %w", err)
		}
	}
	role, err := json.Marshal(m.Role)
	if err != nil {
		return nil, err
	}
	var out bytes.Buffer
	out.WriteByte('{')
	out.WriteString(`"role":`)
	out.Write(role)
	out.WriteString(`,"content":`)
	out.Write(content)
	if m.Name != "" {
		name, err := json.Marshal(m.Name)
		if err != nil {
			return nil, err
		}
		out.WriteString(`,"name":`)
		out.Write(name)
	}
	if len(m.ToolCalls) != 0 {
		calls, err := json.Marshal(m.ToolCalls)
		if err != nil {
			return nil, err
		}
		out.WriteString(`,"tool_calls":`)
		out.Write(calls)
	}
	if m.ToolCallID != "" {
		id, err := json.Marshal(m.ToolCallID)
		if err != nil {
			return nil, err
		}
		out.WriteString(`,"tool_call_id":`)
		out.Write(id)
	}
	if m.Reasoning != nil {
		reasoning, err := json.Marshal(m.Reasoning)
		if err != nil {
			return nil, err
		}
		out.WriteString(`,"reasoning":`)
		out.Write(reasoning)
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var fields map[string]json.RawMessage
	if err := unmarshalStrict(data, &fields); err != nil {
		return err
	}
	for field := range fields {
		switch field {
		case "role", "content", "name", "tool_calls", "tool_call_id", "reasoning":
		default:
			return fmt.Errorf("unknown message field %q", field)
		}
	}
	contentRaw, ok := fields["content"]
	if !ok || len(contentRaw) == 0 || bytes.Equal(bytes.TrimSpace(contentRaw), []byte("null")) {
		return errors.New("message content is required")
	}

	var content Content
	if err := json.Unmarshal(contentRaw, &content); err != nil {
		return fmt.Errorf("message content is invalid: %w", err)
	}

	if raw, ok := fields["role"]; ok {
		if err := json.Unmarshal(raw, &m.Role); err != nil {
			return fmt.Errorf("message role is invalid: %w", err)
		}
	} else {
		m.Role = ""
	}
	m.Content = content
	if raw, ok := fields["name"]; ok {
		if err := json.Unmarshal(raw, &m.Name); err != nil {
			return fmt.Errorf("message name is invalid: %w", err)
		}
	} else {
		m.Name = ""
	}
	if raw, ok := fields["tool_calls"]; ok {
		if err := json.Unmarshal(raw, &m.ToolCalls); err != nil {
			return fmt.Errorf("message tool_calls is invalid: %w", err)
		}
	} else {
		m.ToolCalls = nil
	}
	if raw, ok := fields["tool_call_id"]; ok {
		if err := json.Unmarshal(raw, &m.ToolCallID); err != nil {
			return fmt.Errorf("message tool_call_id is invalid: %w", err)
		}
	} else {
		m.ToolCallID = ""
	}
	if raw, ok := fields["reasoning"]; ok {
		if err := json.Unmarshal(raw, &m.Reasoning); err != nil {
			return fmt.Errorf("message reasoning is invalid: %w", err)
		}
	} else {
		m.Reasoning = nil
	}
	return nil
}
