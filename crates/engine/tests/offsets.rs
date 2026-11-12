use pii_detection_engine::text::Document;
use proptest::prelude::*;

proptest! {
    /// The round trip alone, `slice(char_span(a, b)) == &text[a..b]`, cannot
    /// fail. `char_span` maps a byte offset to an index through the
    /// boundaries table and `slice` maps that index back through the same
    /// table, so the composition is the identity on any sorted table
    /// containing both offsets, whether or not the table means what it says.
    /// A `Document` that stored one entry per byte would answer
    /// `char_span(3, 7)` on "Müller" with `{3, 7}` instead of `{2, 6}` and
    /// still pass it.
    ///
    /// So the counting property is asserted directly. That is the one the
    /// caller depends on: a caller that splices these spans by character, as PHP's
    /// `mb_substr` does, counts code points, so a span is only usable if `start` is the number
    /// of characters preceding `byte_start` in the text. The round trip stays
    /// as well (it is the property `slice` owes callers of this crate), but
    /// it is no longer the whole gate.
    #[test]
    fn a_span_counts_the_code_points_before_it(text in ".{0,200}") {
        let doc = Document::new(text.clone());

        // The number of characters preceding a byte offset is that offset's
        // position in `char_indices`, computed here without consulting the
        // document. The end of the text is the last position and has no
        // character of its own.
        let positions: Vec<(usize, usize)> = text
            .char_indices()
            .enumerate()
            .map(|(preceding, (byte, _))| (byte, preceding))
            .chain(std::iter::once((text.len(), text.chars().count())))
            .collect();

        for (index, &(byte_start, preceding_start)) in positions.iter().enumerate() {
            for &(byte_end, preceding_end) in &positions[index..] {
                let span = doc
                    .char_span(byte_start, byte_end)
                    .expect("boundaries are valid");

                prop_assert_eq!(span.start as usize, preceding_start);
                prop_assert_eq!(span.end as usize, preceding_end);
                prop_assert_eq!(doc.slice(span), &text[byte_start..byte_end]);
            }
        }
    }
}
