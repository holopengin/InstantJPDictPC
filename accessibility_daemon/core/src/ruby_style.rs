//! Ruby style resolution as UI-free data.
//!
//! Port of the ticket-06 rule: body (mini) ruby renders in the surrounding
//! body typeface — white regular base at body size with a gray ruby row —
//! while the term display keeps full-size bold cyan. The values are pinned by
//! the `ruby_style` conformance cases; the iced binary converts [`RubyStyle`]
//! to `iced::Color` at the paint site, so this module is the single
//! implementation both the UI and the tests read.

/// Resolved ruby treatment for one renderer input, as plain data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RubyStyle {
    /// Base-text size in display points.
    pub base_size: f32,
    /// Ruby-text size in display points.
    pub ruby_size: f32,
    /// Base-text colour as linear RGB.
    pub base: [f32; 3],
    /// Ruby-text colour as linear RGB.
    pub ruby: [f32; 3],
    /// Whether the base renders bold.
    pub bold: bool,
}

/// Body typeface sizes the mini ruby inherits (the panel body sizes).
pub const BODY_TEXT_SIZE: f32 = 15.0;
/// Ruby row size for mini (body-flow) ruby.
pub const BODY_RUBY_SIZE: f32 = 9.0;
/// Shared gray ruby row for both modes.
pub const RUBY_GRAY: [f32; 3] = [0.75, 0.75, 0.75];

/// Resolve the ruby treatment: `is_mini = true` is everything the definition
/// body builds (white regular base at body size); `false` is the
/// headword/term display (32pt bold cyan base). Do NOT "fix" mini back to
/// cyan+bold: that reintroduces the ticket-06 symptom on both codebases by
/// design (mobile non-mini `createRubyView`, mini `createBaseTextView`).
pub fn ruby_style(is_mini: bool) -> RubyStyle {
    if is_mini {
        RubyStyle {
            base_size: BODY_TEXT_SIZE,
            ruby_size: BODY_RUBY_SIZE,
            base: [1.0, 1.0, 1.0],
            ruby: RUBY_GRAY,
            bold: false,
        }
    } else {
        RubyStyle {
            base_size: 32.0,
            ruby_size: 13.0,
            base: [0.0, 1.0, 1.0],
            ruby: RUBY_GRAY,
            bold: true,
        }
    }
}
