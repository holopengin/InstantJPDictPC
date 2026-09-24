//! Ungated Yomitan/Jitendex definition formatting (package 04).
//!
//! This module owns the pure stages the desktop `overlay_state` used to carry
//! behind the `db` feature gate, none of which touch the database:
//!
//! * the structured-content walker ([`parse_definition`]) with its sense-group
//!   splitter ([`parse_glossary`]), redirect extractor
//!   ([`extract_redirect_targets`]) and helpers;
//! * the generic plain-text walker ([`plain_text`]/[`plain_all_text`]) behind
//!   bookmark snapshots and CSV export (mirrors Android `Definitions`);
//! * the JSON boundary the UniFFI shim speaks: [`parse_to_json`],
//!   [`parse_glossary_to_json`]. [`DefinitionNode`] is recursive, which UniFFI
//!   records cannot express, so the shim returns the formatted result as JSON
//!   and the Kotlin facade maps it into its existing `DefinitionNode` model
//!   (presentation-only mapping — no duplicated walker).
//!
//! Data in / data out: stored `definitions` JSON in; display nodes (or their
//! JSON) out. No `Rc`, no `Cell`, no DB — this module builds under
//! `--no-default-features` and is `Send` by construction.

use std::collections::HashSet;

use crate::models::{DefinitionNode, ExampleNode};

// ---------------------------------------------------------------------------
// Structured-content walker (moved verbatim from `overlay_state.rs`)
// ---------------------------------------------------------------------------

/// The `data-content` classes that are structural blocks rather than
/// inline text. A block never gets a comma spliced in front of it, and its
/// own children keep their own layout.
const BLOCK_CONTENT_CLASSES: &[&str] = &[
    "sense-groups", "sense-group", "sense",
    "forms", "extra-info",
    "example-sentence", "xref", "antonym", "related",
    "sense-note", "info-gloss", "lang-source", "attribution", "graphic",
];

/// Jitendex's `extra-info` boxes: each renders as its own line.
const BOXED_CONTENT_CLASSES: &[&str] = &[
    "xref", "antonym", "related", "sense-note", "info-gloss", "lang-source",
];

/// `data-content` values that stay inline even on a `ul`/`ol`.
const INLINE_LIST_CLASSES: &[&str] = &[
    "glossary", "infoGlossary", "sourceLanguages", "info-gloss", "sense-note",
];

/// Jitendex form-validity cell classes → the glyph upstream draws.
const FORM_MARKERS: &[(&str, &str)] = &[
    ("form-valid", "◇"),
    ("form-rare", "▽"),
    ("form-pri", "★"),
    ("form-irr", "✕"),
    ("form-out", "古"),
    ("form-old", "旧"),
];

/// Yomitan structured-content class on a node (`data.content`), or None.
fn content_class(node: &serde_json::Value) -> Option<String> {
    node.as_object()
        .and_then(|m| m.get("data"))
        .and_then(|d| d.as_object())
        .and_then(|d| d.get("content"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Parse a stored definition payload into displayable nodes.
pub fn parse_definition(data: &serde_json::Value, separator: &str) -> Vec<DefinitionNode> {
    let mut nodes = Vec::new();
    parse_definition_into(data, separator, &mut nodes);
    nodes
}

fn parse_definition_into(
    data: &serde_json::Value,
    separator: &str,
    nodes: &mut Vec<DefinitionNode>,
) {
    match data {
        serde_json::Value::String(s) => {
            let replaced = s
                .replace("\r\n", " ")
                .replace('\n', " ")
                .replace('\r', " ")
                .replace(';', "; ")
                .replace(";  ", "; ");
            let trimmed = replaced.trim().split_whitespace().collect::<Vec<_>>().join(" ");
            if !trimmed.is_empty() {
                nodes.push(DefinitionNode::Text(trimmed));
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                let mut item_nodes = Vec::new();
                parse_definition_into(item, separator, &mut item_nodes);
                if item_nodes.is_empty() {
                    continue;
                }
                // Attach the separator to the PREVIOUS text node where
                // possible, so it cannot wrap onto a line of its own.
                if !nodes.is_empty() && !separator.is_empty() && !is_block(item) {
                    let last = nodes.last().unwrap();
                    let first = item_nodes.first().unwrap();
                    if is_inline_node(last) && is_inline_node(first) {
                        // #88: a ruby run is one word or sentence, not an
                        // enumeration — never splice a separator between
                        // its pieces (the xref 湾外 must not render
                        // "湾, 外").
                        let glue = if matches!(last, DefinitionNode::Ruby { .. })
                            || matches!(first, DefinitionNode::Ruby { .. })
                        {
                            ""
                        } else {
                            separator
                        };
                        if !glue.is_empty() {
                            let last_idx = nodes.len() - 1;
                            if let DefinitionNode::Text(ref t) = nodes[last_idx] {
                                nodes[last_idx] = DefinitionNode::Text(format!("{}{}", t, glue));
                            } else {
                                nodes.push(DefinitionNode::Text(glue.to_string()));
                            }
                        }
                    }
                }
                nodes.extend(item_nodes);
            }
        }
        serde_json::Value::Object(map) => {
            let tag = map.get("tag").and_then(|v| v.as_str());
            let content = map.get("content").or_else(|| map.get("list"));
            let sc_content = get_attr(map, "content");
            let sc_class = get_attr(map, "class");

            let citation = content_class(data).as_deref() == Some("attribution");
            if citation {
                // #88 follow-up: the source line that closes a Jitendex
                // entry; rendered as a faint citation, not definition text.
                nodes.push(DefinitionNode::Citation(citation_text(content)));
            } else if is_example(map) {
                let jp = map
                    .get("japanese")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| content.and_then(|v| v.as_str()).map(|s| s.to_string()));
                let en = map
                    .get("english")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                if let Some(jp) = jp {
                    nodes.push(DefinitionNode::Example(ExampleNode {
                        japanese: Some(jp),
                        english: en,
                        ..Default::default()
                    }));
                } else {
                    let parts = example_parts(content);
                    if let Some(parts) = parts {
                        nodes.push(DefinitionNode::Example(ExampleNode {
                            parts,
                            ..Default::default()
                        }));
                    } else {
                        nodes.push(DefinitionNode::Example(ExampleNode {
                            content: content
                                .map(|c| parse_definition(c, "\n"))
                                .unwrap_or_default(),
                            ..Default::default()
                        }));
                    }
                }
            } else if let Some(glyph) = sc_class
                .as_deref()
                .and_then(|c| FORM_MARKERS.iter().find(|(k, _)| *k == c))
                .map(|(_, g)| *g)
            {
                nodes.push(DefinitionNode::Text(glyph.to_string()));
            } else if sc_class.as_deref() == Some("tag") {
                let text = content
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                nodes.push(DefinitionNode::Tag { text });
            } else if tag == Some("span")
                && sc_content.as_deref().map_or(false, |s| s.ends_with("-info"))
            {
                let text = content
                    .map(|v| {
                        if let Some(s) = v.as_str() {
                            s.to_string()
                        } else {
                            v.to_string()
                        }
                    })
                    .unwrap_or_default();
                nodes.push(DefinitionNode::Tag { text });
            } else if tag == Some("ruby") {
                if let Some(serde_json::Value::Array(ruby_list)) = content {
                    if ruby_list.len() >= 2 {
                        let term = ruby_list[0]
                            .as_str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| ruby_list[0].to_string());
                        let reading = ruby_list
                            .get(1)
                            .and_then(|v| v.as_object())
                            .and_then(|m| m.get("content"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        nodes.push(DefinitionNode::Ruby { term, reading });
                    } else if let Some(c) = content {
                        // Fail open: a malformed ruby still shows its content.
                        parse_definition_into(c, separator, nodes);
                    }
                } else if let Some(c) = content {
                    parse_definition_into(c, separator, nodes);
                }
            } else if tag == Some("table") {
                // Rows -> cells, real content this time (#88). Each cell is
                // itself structured content, so a cell can carry ruby.
                let rows: Vec<Vec<Vec<DefinitionNode>>> = content
                    .and_then(|c| c.as_array())
                    .map(|rows| {
                        rows.iter()
                            .map(|row| {
                                let cells = row
                                    .as_object()
                                    .and_then(|m| m.get("content"))
                                    .unwrap_or(row);
                                cells
                                    .as_array()
                                    .map(|cells| {
                                        cells
                                            .iter()
                                            .map(|cell| {
                                                // Parse the whole cell, not
                                                // just its content, so a
                                                // form-validity class on
                                                // the `td` is seen.
                                                parse_definition(cell, separator)
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default()
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if !rows.is_empty() {
                    nodes.push(DefinitionNode::Table { rows });
                } else if let Some(c) = content {
                    // Fail open: an unshaped table still shows its content.
                    parse_definition_into(c, separator, nodes);
                }
            } else if tag == Some("ul") || tag == Some("ol") {
                if INLINE_LIST_CLASSES.contains(&sc_content.as_deref().unwrap_or("")) {
                    if let Some(c) = content {
                        parse_definition_into(c, separator, nodes);
                    }
                } else {
                    let items: Vec<Vec<DefinitionNode>> = content
                        .and_then(|c| c.as_array())
                        .map(|items| {
                            items
                                .iter()
                                .map(|item| parse_definition(item, separator))
                                .collect()
                        })
                        .unwrap_or_default();
                    if !items.is_empty() {
                        nodes.push(DefinitionNode::ListBlock {
                            items,
                            list_type: sc_content,
                        });
                    } else if let Some(c) = content {
                        parse_definition_into(c, separator, nodes);
                    }
                }
            } else if sc_content
                .as_deref()
                .map_or(false, |c| BOXED_CONTENT_CLASSES.contains(&c))
            {
                // #88 follow-up: an extra-info box gets a block of its own,
                // so it starts a new line instead of being spliced into the
                // sense's inline text run.
                let inner = content
                    .map(|c| parse_definition(c, separator))
                    .unwrap_or_default();
                if !inner.is_empty() {
                    nodes.push(DefinitionNode::Group {
                        nodes: inner,
                        is_inline: false,
                    });
                }
            } else if let Some(c) = content {
                // Fail open: any other tag contributes its content.
                parse_definition_into(c, separator, nodes);
            }
        }
        _ => {}
    }
}

/// #88: split a Jitendex example box into its Japanese and English parts.
/// Returns None when the box does not have that shape.
fn example_parts(content: Option<&serde_json::Value>) -> Option<Vec<Vec<DefinitionNode>>> {
    let children = content?.as_array()?;
    let parts: Vec<Vec<DefinitionNode>> = children
        .iter()
        .filter_map(|child| {
            let map = child.as_object()?;
            match get_attr(map, "content").as_deref() {
                Some("example-sentence-a") | Some("example-sentence-b") => {
                    Some(parse_definition(
                        map.get("content").unwrap_or(child),
                        "",
                    ))
                }
                _ => None,
            }
        })
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts)
    }
}

/// #88: a row's glossary, split for numbering.
pub(crate) fn parse_glossary(data: &[serde_json::Value]) -> ParsedGlossary {
    let mut groups: Vec<ParsedSenseGroup> = Vec::new();
    let mut trailing: Vec<DefinitionNode> = Vec::new();

    fn walk(
        node: &serde_json::Value,
        groups: &mut Vec<ParsedSenseGroup>,
        trailing: &mut Vec<DefinitionNode>,
    ) {
        match node {
            serde_json::Value::Array(arr) => {
                for item in arr {
                    walk(item, groups, trailing);
                }
            }
            serde_json::Value::Object(map) => {
                match content_class(node).as_deref() {
                    Some("sense-group") => {
                        groups.push(split_sense_group(node));
                    }
                    Some("sense") => {
                        groups.push(ParsedSenseGroup {
                            header: Vec::new(),
                            senses: vec![parse_definition(node, ", ")],
                            trailing: Vec::new(),
                        });
                    }
                    Some("forms") | Some("attribution") => {
                        trailing.extend(parse_definition(node, ", "));
                    }
                    _ => {
                        if let Some(content) = map.get("content") {
                            walk(content, groups, trailing);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    walk(
        &serde_json::Value::Array(data.to_vec()),
        &mut groups,
        &mut trailing,
    );

    if groups.is_empty() {
        ParsedGlossary {
            structured: false,
            groups: Vec::new(),
            plain: parse_definition(&serde_json::Value::Array(data.to_vec()), ", "),
            trailing: Vec::new(),
        }
    } else {
        ParsedGlossary {
            structured: true,
            groups,
            plain: Vec::new(),
            trailing,
        }
    }
}

/// Split one Jitendex `sense-group` into its shared header, one entry per
/// `sense` child, and any forms/attribution block inside it. An unexpected
/// shape falls back to the whole group as a single sense, so nothing is
/// dropped.
fn split_sense_group(sense_group: &serde_json::Value) -> ParsedSenseGroup {
    let mut header: Vec<DefinitionNode> = Vec::new();
    let mut senses: Vec<Vec<DefinitionNode>> = Vec::new();
    let mut trailing: Vec<DefinitionNode> = Vec::new();

    fn walk(
        node: &serde_json::Value,
        header: &mut Vec<DefinitionNode>,
        senses: &mut Vec<Vec<DefinitionNode>>,
        trailing: &mut Vec<DefinitionNode>,
    ) {
        match node {
            serde_json::Value::Array(arr) => {
                for item in arr {
                    walk(item, header, senses, trailing);
                }
            }
            serde_json::Value::Object(map) => {
                match content_class(node).as_deref() {
                    Some("sense") => {
                        senses.push(parse_definition(node, ", "));
                    }
                    Some("forms") | Some("attribution") => {
                        trailing.extend(parse_definition(node, ", "));
                    }
                    Some(_) => {
                        header.extend(parse_definition(node, ", "));
                    }
                    None => {
                        // A wrapper node may carry its single child as an
                        // object rather than a one-element array — Jitendex
                        // emits `ol` with one `li[sense]` this way. Descend
                        // into either shape; treating the object form as
                        // header copy put the sense (and its example) in
                        // the unnumbered header and rendered it again as
                        // the fallback sense.
                        let content = map.get("content");
                        if matches!(
                            content,
                            Some(serde_json::Value::Array(_)) | Some(serde_json::Value::Object(_))
                        ) {
                            walk(content.unwrap(), header, senses, trailing);
                        } else {
                            header.extend(parse_definition(node, ", "));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(content) = sense_group
        .as_object()
        .and_then(|m| m.get("content"))
    {
        walk(content, &mut header, &mut senses, &mut trailing);
    }

    if senses.is_empty() {
        senses.push(parse_definition(sense_group, ", "));
    }
    ParsedSenseGroup {
        header,
        senses,
        trailing,
    }
}

/// The plain text of a Jitendex `attribution` block: its link labels joined
/// in order (`JMdict`, or `JMdict | Tatoeba`).
fn citation_text(content: Option<&serde_json::Value>) -> String {
    fn walk(node: &serde_json::Value, parts: &mut String) {
        match node {
            serde_json::Value::String(s) => parts.push_str(s),
            serde_json::Value::Array(arr) => {
                for item in arr {
                    walk(item, parts);
                }
            }
            serde_json::Value::Object(map) => {
                if let Some(c) = map.get("content") {
                    walk(c, parts);
                }
            }
            _ => {}
        }
    }
    let mut parts = String::new();
    if let Some(c) = content {
        walk(c, &mut parts);
    }
    parts
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn get_attr(data: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    data.get("data")
        .and_then(|v| v.as_object())
        .and_then(|m| m.get(key))
        .or_else(|| data.get(&format!("data-{}", key)))
        .or_else(|| data.get(key))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// #88: true only for an actual example container. The old test matched
/// any `data-content` containing "example", which also caught Jitendex's
/// `example-sentence-a`/`-b` divisors and `example-keyword` spans and
/// wrapped each in its own nested box.
fn is_example(data: &serde_json::Map<String, serde_json::Value>) -> bool {
    let sc_content = get_attr(data, "content");
    data.get("type")
        .and_then(|v| v.as_str())
        .map_or(false, |t| t == "sentence" || t == "example")
        || data.contains_key("japanese")
        || sc_content.as_deref() == Some("example-sentence")
        || sc_content.as_deref() == Some("examples")
        || get_attr(data, "class").map_or(false, |c| c.contains("example"))
}

fn is_inline_node(node: &DefinitionNode) -> bool {
    matches!(
        node,
        DefinitionNode::Text(_)
            | DefinitionNode::Ruby { .. }
            | DefinitionNode::Tag { .. }
    )
}

/// Check whether a JSON value represents a block-level element (as opposed
/// to inline). Mirrors the Kotlin `isBlock` helper.
fn is_block(data: &serde_json::Value) -> bool {
    match data {
        serde_json::Value::Array(arr) => arr.iter().any(|item| is_block(item)),
        serde_json::Value::Object(map) => {
            if is_example(map) {
                return true;
            }
            let tag = map.get("tag").and_then(|v| v.as_str());
            let sc_content = get_attr(map, "content");

            if tag == Some("table") {
                return true;
            }
            // Lists are blocks unless they are one of the inline
            // gloss/reference enumerations.
            if tag == Some("ul") || tag == Some("ol") {
                return !INLINE_LIST_CLASSES
                    .contains(&sc_content.as_deref().unwrap_or(""));
            }
            if let Some(cls) = sc_content.as_deref() {
                if BLOCK_CONTENT_CLASSES.contains(&cls) {
                    return true;
                }
            }

            let content = map.get("content").or_else(|| map.get("list"));
            content.map_or(false, |c| is_block(c))
        }
        _ => false,
    }
}

/// The parsed form of one database row's glossary. Jitendex rows carry
/// every sense in one payload, so `structured` is true and `groups` holds
/// each `sense-group`; JMdict/KANJIDIC rows keep the pre-#88 flat shape.
pub(crate) struct ParsedGlossary {
    pub(crate) structured: bool,
    pub(crate) groups: Vec<ParsedSenseGroup>,
    pub(crate) plain: Vec<DefinitionNode>,
    pub(crate) trailing: Vec<DefinitionNode>,
}

/// One Jitendex `sense-group` split for numbering.
pub(crate) struct ParsedSenseGroup {
    pub(crate) header: Vec<DefinitionNode>,
    pub(crate) senses: Vec<Vec<DefinitionNode>>,
    pub(crate) trailing: Vec<DefinitionNode>,
}

// ---------------------------------------------------------------------------
// Redirects (moved verbatim from `overlay_state.rs`)
// ---------------------------------------------------------------------------

/// #65: JMdict pointer entries (variant spellings) carry only `?query=`
/// links; this returns those headwords so lookup can resolve them. Entries
/// with any real definitional content yield nothing (their `see also` links
/// are not redirects). Mirrors `DictionaryRedirects.extractTargets`.
pub fn extract_redirect_targets(definitions_json: &str, max_targets: usize) -> Vec<String> {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(definitions_json) else {
        return Vec::new();
    };
    let mut targets: Vec<String> = Vec::new();
    let mut has_definitional = false;

    fn visit(
        node: &serde_json::Value,
        targets: &mut Vec<String>,
        has_definitional: &mut bool,
    ) {
        match node {
            serde_json::Value::Object(o) => {
                if o.get("tag").and_then(|v| v.as_str()) == Some("a") {
                    let href = o.get("href").and_then(|v| v.as_str()).unwrap_or("");
                    let raw = href
                        .split("?query=")
                        .nth(1)
                        .unwrap_or("")
                        .split('&')
                        .next()
                        .unwrap_or("");
                    if !raw.is_empty() {
                        let target = percent_decode(raw);
                        if !target.trim().is_empty() {
                            targets.push(target);
                        }
                    }
                }
                if let Some(dc) = o
                    .get("data")
                    .and_then(|d| d.as_object())
                    .and_then(|d| d.get("content"))
                    .and_then(|v| v.as_str())
                {
                    if dc != "references" && dc != "refGlosses" {
                        *has_definitional = true;
                    }
                }
                for value in o.values() {
                    visit(value, targets, has_definitional);
                }
            }
            serde_json::Value::Array(arr) => {
                for value in arr {
                    visit(value, targets, has_definitional);
                }
            }
            _ => {}
        }
    }

    visit(&root, &mut targets, &mut has_definitional);
    if has_definitional {
        return Vec::new();
    }
    targets.dedup();
    targets.truncate(max_targets);
    targets
}

/// Decode `%XX` escapes (UTF-8 lossily), as `java.net.URLDecoder` does for
/// the redirect hrefs.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 3 <= bytes.len() {
            if let Some(byte) = std::str::from_utf8(&bytes[i + 1..i + 3])
                .ok()
                .and_then(|h| u8::from_str_radix(h, 16).ok())
            {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------------------
// Plain-text walker (mirrors Android `Definitions.plain`/`plainAll`, #67)
// ---------------------------------------------------------------------------

/// Structured-content keys that describe presentation, carry a link target
/// or hold a tooltip — never the readable definition. Without this the
/// generic fallback (which joins an object's values) would export a Jitendex
/// form cell (`<td class=form-valid><span title=…/></td>`) as
/// "valid form/reading combination, span, form-valid".
const NON_TEXT_KEYS: &[&str] = &[
    "data", "href", "style", "path", "title",
    "tag", "type", "lang", "class", "code", "id",
];

/// Flatten a stored `definitions` blob to plain text: one line per top-level
/// sense, nested glosses joined with ", ". A malformed blob falls back to the
/// raw string rather than losing the senses.
pub fn plain_text(definitions_json: &str) -> String {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(definitions_json) else {
        return definitions_json.to_string();
    };
    if let serde_json::Value::Array(arr) = &root {
        return arr
            .iter()
            .filter_map(render_plain)
            .filter(|s| !s.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
    }
    render_plain(&root).unwrap_or_default()
}

/// Several definition blobs (multiple entries in one dictionary block) as one
/// blob of lines: blank lines dropped, first occurrence kept.
pub fn plain_all_text(rows: &[String]) -> String {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<&str> = Vec::new();
    let rendered: Vec<String> = rows.iter().map(|r| plain_text(r)).collect();
    for text in &rendered {
        for line in text.lines() {
            if !line.trim().is_empty() && seen.insert(line.to_string()) {
                out.push(line);
            }
        }
    }
    out.join("\n")
}

fn render_plain(el: &serde_json::Value) -> Option<String> {
    match el {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Array(arr) => {
            let joined = arr
                .iter()
                .filter_map(render_plain)
                .filter(|s| !s.trim().is_empty())
                .collect::<Vec<_>>()
                .join(", ");
            if joined.trim().is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        serde_json::Value::Object(map) => {
            if let Some(content) = map
                .get("content")
                .or_else(|| map.get("list"))
                .or_else(|| map.get("glossary"))
            {
                return render_plain(content);
            }
            let joined = map
                .iter()
                .filter(|(k, _)| !NON_TEXT_KEYS.contains(&k.as_str()))
                .filter_map(|(_, v)| render_plain(v))
                .filter(|s| !s.trim().is_empty())
                .collect::<Vec<_>>()
                .join(", ");
            if joined.trim().is_empty() {
                None
            } else {
                Some(joined)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JSON boundary for the UniFFI shim
// ---------------------------------------------------------------------------

/// JSON-crossable mirror of [`DefinitionNode`].
///
/// A flat struct (not a tagged enum) so both serde and the Kotlin Gson DTO
/// map it field-for-field with no polymorphism: `kind` selects the variant
/// (`text`, `ruby`, `tag`, `citation`, `example`, `list`, `table`, `group`)
/// and only that variant's fields are set. `list_type` is the `ListBlock`
/// type; `is_inline` is the `Group` flag.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DefinitionNodeJson {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub term: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reading: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub japanese: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub english: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<DefinitionNodeJson>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<Vec<DefinitionNodeJson>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<Vec<DefinitionNodeJson>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<Vec<Vec<Vec<DefinitionNodeJson>>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nodes: Option<Vec<DefinitionNodeJson>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_type: Option<String>,
    #[serde(default)]
    pub is_inline: bool,
}

/// JSON-crossable mirror of [`ParsedSenseGroup`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParsedSenseGroupJson {
    pub header: Vec<DefinitionNodeJson>,
    pub senses: Vec<Vec<DefinitionNodeJson>>,
    pub trailing: Vec<DefinitionNodeJson>,
}

/// JSON-crossable mirror of [`ParsedGlossary`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParsedGlossaryJson {
    pub structured: bool,
    pub groups: Vec<ParsedSenseGroupJson>,
    pub plain: Vec<DefinitionNodeJson>,
    pub trailing: Vec<DefinitionNodeJson>,
}

fn nodes_to_json(nodes: &[DefinitionNode]) -> Vec<DefinitionNodeJson> {
    nodes.iter().map(DefinitionNodeJson::from_node).collect()
}

impl DefinitionNodeJson {
    fn from_node(node: &DefinitionNode) -> Self {
        fn empty() -> DefinitionNodeJson {
            DefinitionNodeJson {
                kind: String::new(),
                text: None,
                term: None,
                reading: None,
                japanese: None,
                english: None,
                content: None,
                parts: None,
                items: None,
                rows: None,
                nodes: None,
                list_type: None,
                is_inline: false,
            }
        }
        match node {
            DefinitionNode::Text(text) => DefinitionNodeJson {
                kind: "text".to_string(),
                text: Some(text.clone()),
                ..empty()
            },
            DefinitionNode::Ruby { term, reading } => DefinitionNodeJson {
                kind: "ruby".to_string(),
                term: Some(term.clone()),
                reading: Some(reading.clone()),
                ..empty()
            },
            DefinitionNode::Tag { text } => DefinitionNodeJson {
                kind: "tag".to_string(),
                text: Some(text.clone()),
                ..empty()
            },
            DefinitionNode::Citation(text) => DefinitionNodeJson {
                kind: "citation".to_string(),
                text: Some(text.clone()),
                ..empty()
            },
            DefinitionNode::Example(ex) => {
                // Mirror the Kotlin `Example` shapes exactly: the
                // `{japanese, english}` pair carries no content list, the
                // split-parts shape carries no content list, otherwise the
                // generic content list (possibly empty) is present.
                let (content, parts) = if ex.japanese.is_some() || ex.english.is_some() {
                    (None, None)
                } else if !ex.parts.is_empty() {
                    (
                        None,
                        Some(
                            ex.parts
                                .iter()
                                .map(|p| nodes_to_json(p))
                                .collect(),
                        ),
                    )
                } else {
                    (Some(nodes_to_json(&ex.content)), None)
                };
                DefinitionNodeJson {
                    kind: "example".to_string(),
                    japanese: ex.japanese.clone(),
                    english: ex.english.clone(),
                    content,
                    parts,
                    ..empty()
                }
            }
            DefinitionNode::ListBlock { items, list_type } => DefinitionNodeJson {
                kind: "list".to_string(),
                items: Some(items.iter().map(|i| nodes_to_json(i)).collect()),
                list_type: list_type.clone(),
                ..empty()
            },
            DefinitionNode::Table { rows } => DefinitionNodeJson {
                kind: "table".to_string(),
                rows: Some(
                    rows.iter()
                        .map(|r| r.iter().map(|c| nodes_to_json(c)).collect())
                        .collect(),
                ),
                ..empty()
            },
            DefinitionNode::Group { nodes, is_inline } => DefinitionNodeJson {
                kind: "group".to_string(),
                nodes: Some(nodes_to_json(nodes)),
                is_inline: *is_inline,
                ..empty()
            },
        }
    }
}

/// [`parse_definition`] over a JSON string, returning the nodes as JSON.
/// An unparseable payload yields `[]` (the walker fails open per call).
pub fn parse_to_json(glossary_json: &str, separator: &str) -> String {
    let value: serde_json::Value =
        serde_json::from_str(glossary_json).unwrap_or(serde_json::Value::Null);
    let nodes = parse_definition(&value, separator);
    serde_json::to_string(&nodes_to_json(&nodes)).unwrap_or_else(|_| "[]".to_string())
}

/// [`parse_glossary`] over a JSON string, returning the glossary as JSON.
/// A non-array payload is treated as a single glossary item; an unparseable
/// payload yields the unstructured empty shape.
pub fn parse_glossary_to_json(glossary_json: &str) -> String {
    let value: serde_json::Value =
        serde_json::from_str(glossary_json).unwrap_or(serde_json::Value::Null);
    let items: Vec<serde_json::Value> = match value {
        serde_json::Value::Array(arr) => arr,
        other => vec![other],
    };
    let parsed = parse_glossary(&items);
    let out = ParsedGlossaryJson {
        structured: parsed.structured,
        groups: parsed
            .groups
            .iter()
            .map(|g| ParsedSenseGroupJson {
                header: nodes_to_json(&g.header),
                senses: g.senses.iter().map(|s| nodes_to_json(s)).collect(),
                trailing: nodes_to_json(&g.trailing),
            })
            .collect(),
        plain: nodes_to_json(&parsed.plain),
        trailing: nodes_to_json(&parsed.trailing),
    };
    serde_json::to_string(&out).unwrap_or_else(|_| {
        "{\"structured\":false,\"groups\":[],\"plain\":[],\"trailing\":[]}".to_string()
    })
}
