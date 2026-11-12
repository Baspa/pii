use serde::Serialize;

/// A span in Unicode code points, the unit `mb_substr` counts in.
#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
pub struct CharSpan {
    pub start: u32,
    pub end: u32,
}

/// An immutable text with the map that converts byte offsets to code points.
///
/// Built once per text. Every recognizer works in bytes, because that is what
/// the regex engine reports, and converts here exactly once on the way out.
pub struct Document {
    text: String,
    /// The byte offset of every character boundary, and the end. One longer
    /// than the code-point count.
    boundaries: Vec<usize>,
}

impl Document {
    pub fn new(text: String) -> Self {
        let mut boundaries: Vec<usize> = text.char_indices().map(|(index, _)| index).collect();
        boundaries.push(text.len());
        Self { text, boundaries }
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn char_len(&self) -> usize {
        self.boundaries.len() - 1
    }

    /// Converts a byte range into a code-point span, or None when either end
    /// falls inside a character or the range is reversed.
    pub fn char_span(&self, byte_start: usize, byte_end: usize) -> Option<CharSpan> {
        if byte_start > byte_end {
            return None;
        }
        let start = self.boundaries.binary_search(&byte_start).ok()?;
        let end = self.boundaries.binary_search(&byte_end).ok()?;
        Some(CharSpan {
            start: start as u32,
            end: end as u32,
        })
    }

    /// Extracts the text slice for a code-point span.
    ///
    /// # Panics
    ///
    /// Panics if the span is out of range. This method expects spans produced by
    /// `char_span` on the same document. A span from elsewhere may panic.
    pub fn slice(&self, span: CharSpan) -> &str {
        let start = self.boundaries[span.start as usize];
        let end = self.boundaries[span.end as usize];
        &self.text[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_byte_offsets_equal_code_point_offsets() {
        let doc = Document::new("hello world".to_string());
        assert_eq!(doc.char_span(6, 11), Some(CharSpan { start: 6, end: 11 }));
        assert_eq!(doc.slice(CharSpan { start: 6, end: 11 }), "world");
    }

    #[test]
    fn multibyte_text_shifts_the_offsets() {
        // "Müller": the umlaut is two bytes, so "ller" starts at byte 3 and code point 2.
        let doc = Document::new("Müller".to_string());
        let span = doc
            .char_span(3, 7)
            .expect("byte range must sit on boundaries");
        assert_eq!(span, CharSpan { start: 2, end: 6 });
        assert_eq!(doc.slice(span), "ller");
    }

    #[test]
    fn emoji_counts_as_its_code_points() {
        // A family emoji is several code points joined by zero-width joiners.
        let doc = Document::new("hi 👨‍👩‍👧 there".to_string());
        assert_eq!(doc.char_len(), "hi 👨‍👩‍👧 there".chars().count());
    }

    #[test]
    fn a_byte_range_inside_a_character_is_rejected() {
        let doc = Document::new("Müller".to_string());
        assert_eq!(doc.char_span(2, 3), None);
    }

    #[test]
    fn a_reversed_byte_range_is_rejected() {
        let doc = Document::new("abc".to_string());
        assert_eq!(doc.char_span(2, 0), None);
    }
}
