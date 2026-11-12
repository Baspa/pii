//! The calibration for the driving licences that have no structure to be
//! decided by, written as a compile error rather than as a comment.
//!
//! `training/reports/driving-licence.md` excluded the Dutch, Belgian, French
//! and Italian licence numbers under a structure-or-nothing bar: a bare run
//! of digits does not qualify, whatever words surround it. That bar was too
//! strict for a service that also reads context, and the route past it is the
//! one `phone.rs` already took -- a graded ladder where each piece of
//! evidence contributes toward a threshold, so that a signal which cannot
//! decide alone can decide together with another.
//!
//! What lives here is only the arithmetic. The recognizers themselves are
//! pack data (`packs/nl.toml`, `packs/fr.toml`, `packs/it.toml`), because a
//! regex and a vocabulary are exactly what a pack is for and splitting one
//! recognizer across two files would be worse than restating five numbers.
//! The restatement is guarded the way `phone.rs`'s restatement of
//! `NEGATIVE_CONTEXT_PENALTY` is: by a test that reads the real packs and the
//! real `thresholds.toml` and fails when they drift
//! (`tests/driving_licence.rs::the_context_carried_ladder_is_the_one_the_packs_commit`).
//!
//! ### Why these weights and not others
//!
//! Each is derived from a bar rather than picked, and the derivation is
//! two-sided -- there is a reason the number is not larger and a reason it is
//! not smaller.
//!
//! ```text
//! T  = 0.70   DRIVING_LICENCE in thresholds.toml
//! P  = 0.25   pii_detection_resolve::NEGATIVE_CONTEXT_PENALTY
//! C  = 0.85   what a *verified* licence scores without a label of its own
//!             (de_fuehrerschein: 0.40 shape + 0.45 check digit)
//!
//!     SHAPE                  <  T          a digit run never decides alone
//!     A_WORD_NEARBY          <  T          a word never decides alone
//!     SHAPE + A_WORD_NEARBY  >= T          together they do
//!     SHAPE + A_WORD_NEARBY  <  T + P      and one word saying "order
//!                                          number" takes them back off
//!     SHAPE + A_WORD_NEARBY  <  C          and neither is ever worth as much
//!                                          as an arithmetic that was checked
//! ```
//!
//! With `SHAPE` fixed at the 0.40 every structured recognizer in these packs
//! already starts from, the last three lines pin `A_WORD_NEARBY` into the
//! open interval `(0.30, 0.45)`. `0.35` is the lowest multiple of 0.05 inside
//! it: the word is worth what it takes to carry the run over the bar and a
//! margin, and not a point more. Landing on `0.30` exactly would put the
//! whole recognizer on f32 equality with the threshold, which is not a
//! calibration, it is a coin toss.
//!
//! ### The property this buys, and the one it deliberately does not
//!
//! **Neither signal reaches the threshold alone.** That is the whole safety
//! argument for masking a bare run of ten digits, and it is asserted below
//! rather than described. A ten-digit order number in a sentence with no
//! licence word scores 0.40 and is dropped; the word `rijbewijs` with no
//! ten-digit run beside it raises no candidate at all, because the pattern
//! never matched.
//!
//! **It is not the phone recognizer's property.** `phone.rs` calibrates so
//! that everything clearing the bar also clears the *penalised* bar -- no
//! suppression term can ever change a phone answer. That is right for a
//! number written as a telephone number, which stays one whatever it is
//! filed under. It is wrong here: a ten-digit run behind the word
//! `ordernummer` **is** an order number, and there is no checksum to say
//! otherwise. So this ladder is calibrated the other way, and the required
//! hard negative -- an order number with a licence word elsewhere in the same
//! 64-byte window -- is the case that decides it.

/// What a licence-shaped run scores on its shape alone.
///
/// The same floor every structured recognizer in these packs starts from
/// (`nl_bsn`, `de_steuer_id`, `de_fuehrerschein` all use 0.40), and
/// deliberately not more: ten bare digits is weaker evidence than nine digits
/// with an eleven-proof over them, so it cannot be worth more than they are.
/// Deliberately not less either -- at zero the shape would be decoration and
/// the recognizer would be a word list.
pub const SHAPE: f32 = 0.40;

/// What a nearby word saying `rijbewijs`, `permis de conduire` or `patente di
/// guida` adds.
///
/// See the module docs for the interval this is pinned into and why it sits
/// at the bottom of it.
pub const A_WORD_NEARBY: f32 = 0.35;

/// `DRIVING_LICENCE` in `crates/resolve/thresholds.toml`, restated so
/// the weights above can be read against the bar they were chosen for.
///
/// Restating a number is how two files drift apart, so the real relationship
/// is pinned through the real resolver reading the real file. This constant
/// is what makes the block below readable where it is written.
pub const CALIBRATED_AGAINST: f32 = 0.70;

/// `pii_detection_resolve::NEGATIVE_CONTEXT_PENALTY`, restated for the same reason
/// and pinned the same way.
pub const NEGATIVE_CONTEXT_PENALTY: f32 = 0.25;

/// The bar a licence has to clear to survive one word saying it is something
/// else. These recognizers are calibrated to stay *below* it; see the module
/// docs.
pub const PENALISED_AGAINST: f32 = CALIBRATED_AGAINST + NEGATIVE_CONTEXT_PENALTY;

/// What `de_fuehrerschein` scores on a shape whose modulo-11 check digit was
/// computed and passed, with no licence word anywhere near it.
///
/// The ceiling on this ladder. Evidence that was *verified* has to outrank
/// evidence that was read, or the word "rijbewijs" would be worth as much as
/// an arithmetic that came out right.
pub const A_CHECKED_LICENCE: f32 = 0.40 + 0.45;

/// The calibration, as a compile error.
///
/// Each line is a bar, and every bar is two-sided somewhere in the block: a
/// weight that grew would break one line and a weight that shrank would break
/// another. That is what makes this a calibration rather than five numbers
/// that happen to work.
const _: () = {
    // Neither signal decides alone. This is the whole reason a bare digit run
    // may be masked at all, and it is the first thing a careless edit would
    // take away.
    assert!(SHAPE < CALIBRATED_AGAINST);
    assert!(A_WORD_NEARBY < CALIBRATED_AGAINST);
    // Together they decide.
    assert!(SHAPE + A_WORD_NEARBY >= CALIBRATED_AGAINST);
    // And one word saying the digits are an order number takes them back off
    // the mask. Unlike a telephone number, a run of digits with no check over
    // it has nothing to argue back with.
    assert!(SHAPE + A_WORD_NEARBY < PENALISED_AGAINST);
    // A word read is never worth as much as an arithmetic checked.
    assert!(SHAPE + A_WORD_NEARBY < A_CHECKED_LICENCE);
    // A penalty that is not a penalty would make the fourth line vacuous.
    assert!(PENALISED_AGAINST > CALIBRATED_AGAINST);
};
