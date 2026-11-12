//! What the `CREDENTIAL` recognizers must and must not mask, through the real
//! packs, the real thresholds and the real resolver.
//!
//! EVERY VALUE IN THIS FILE IS SYNTHETIC AND WORTHLESS. The keys are drawn
//! from the documented alphabet at the documented length and were never
//! issued by any vendor; the PEM bodies are base64 that does not decode to a
//! valid ASN.1 structure, so no TLS stack, SSH client or OpenSSL build will
//! load one. See `training/reports/credentials.md` for the full argument.
//!
//! WHAT THIS FILE IS FOR, BEYOND "IT WORKS". Each pattern below is pinned
//! from *both* sides -- one value that must be found, and one value one
//! character away from it that must not be. A guard that only asserts the
//! happy path passes just as cheerfully against `.*`, and this repository has
//! shipped sixteen guards that claimed more than they checked. The negative
//! half is the half that fails when a length bound or a prefix is loosened.

use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use pii_detection_resolve::Resolver;
use pii_detection_rules::matcher::Matcher;

/// What the whole pipeline returns for `CREDENTIAL` over one text -- matcher
/// and resolver together, which is what a caller of `/analyze` sees. A
/// candidate raised and then dropped is not a detection.
fn credentials(text: &str) -> Vec<pii_detection_engine::candidate::Candidate> {
    let document = Document::new(text.to_string());
    Resolver::new().resolve(
        Matcher::new().scan(&document),
        &[EntityType::Credential],
        "auto",
    )
}

/// The exact substrings the pipeline masked, so a test can assert the span
/// and not merely the count.
fn masked(text: &str) -> Vec<String> {
    let document = Document::new(text.to_string());
    credentials(text)
        .iter()
        .map(|candidate| document.slice(candidate.span).to_string())
        .collect()
}

fn is_masked(text: &str, value: &str) -> bool {
    masked(text).iter().any(|span| span.contains(value))
}

// ---------------------------------------------------------------------------
// Family 1: known-format API keys
// ---------------------------------------------------------------------------

/// One row per vendor: a value of the documented shape that must be masked,
/// and a value one edit away from it -- a truncated tail, a prefix nobody
/// issues, a separator nobody uses -- that must not be.
///
/// MUTATION TARGET. Loosening any length bound or shortening any prefix in
/// `packs/universal.toml` makes one of the `unmasked` rows start matching,
/// and this test names which vendor.
#[test]
fn every_documented_key_format_is_masked_and_its_near_miss_is_not() {
    let rows: &[(&str, &str, &str)] = &[
        (
            "aws long-term",
            "AKIAQ4TZ7XW2NMRJ5KDB",
            // Fifteen characters after the prefix, not sixteen.
            "AKIAQ4TZ7XW2NMRJ5KD",
        ),
        (
            "aws temporary",
            "ASIAQ4TZ7XW2NMRJ5KDB",
            // A role identifier: same shape, not a credential.
            "AROAQ4TZ7XW2NMRJ5KDB",
        ),
        (
            "github classic",
            "ghp_9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3Vt",
            // Thirty-five characters after `ghp_`, one short of the minimum.
            "ghp_9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3V",
        ),
        (
            "github fine-grained",
            "github_pat_11ABCDEFG0aBcDeFgHiJkL_9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTvB4nX7cRj0LpZaYdE6s",
            // Ninety-two characters, one short of the stated ninety-three.
            "github_pat_11ABCDEFG0aBcDeFgHiJkL_9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTvB4nX7cRj0LpZaYdE6",
        ),
        (
            "stripe secret",
            "sk_live_9wKQ2mTvB4nX7cRj0LpZaYdE",
            // Twenty-three characters after the mode, one short.
            "sk_live_9wKQ2mTvB4nX7cRj0LpZaYd",
        ),
        (
            "stripe publishable",
            "pk_live_9wKQ2mTvB4nX7cRj0LpZaYdE",
            // A mode Stripe does not issue.
            "pk_prod_9wKQ2mTvB4nX7cRj0LpZaYdE",
        ),
        (
            "openai legacy",
            "sk-9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTvB4nX7cR",
            // Forty-seven characters after `sk-`, one short of forty-eight.
            "sk-9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTvB4nX7c",
        ),
        (
            "openai project",
            "sk-proj-9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTvB4nX7cRj0LpZ",
            // A body far too short to be a key.
            "sk-proj-9wKQ2mTvB4nX7c",
        ),
        (
            "anthropic api",
            "sk-ant-api03-9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTvB4nX7cRj0LpZ",
            // No generation digits.
            "sk-ant-api-9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTvB4nX7cRj0LpZ",
        ),
        (
            "slack bot",
            "xoxb-2345678901-2345678901-9wKQ2mTvB4nX7cRj0LpZaYdE",
            // The prose collision the segmented shape exists to exclude.
            "xoxo-love-you-always-and-forever",
        ),
        (
            "slack app-level",
            "xapp-1-A04TZ7XW2NM-2345678901-9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            // A signature segment that is not sixty-four hex characters.
            "xapp-1-A04TZ7XW2NM-2345678901-9f86d081884c7d659a2feaa0c55ad015",
        ),
        (
            "google api key",
            "AIzaB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTv-B",
            // Thirty-four characters after the prefix, one short of thirty-five.
            "AIzaB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTv-",
        ),
    ];

    for (vendor, key, near_miss) in rows {
        let found = format!("The deployment notes carry {key} in the header.");
        assert!(
            is_masked(&found, key),
            "{vendor}: a documented key format went unmasked"
        );

        let missed = format!("The deployment notes carry {near_miss} in the header.");
        assert!(
            masked(&missed).is_empty(),
            "{vendor}: a value one edit from the format was masked anyway -- got {:?}",
            masked(&missed)
        );
    }
}

/// The span is the key and nothing around it. A recognizer that swallowed
/// the surrounding words would over-mask every config line it fired on, and
/// the count alone cannot see that.
#[test]
fn the_masked_span_is_exactly_the_key() {
    let text = "Set aws_access_key_id = AKIAQ4TZ7XW2NMRJ5KDB before deploying.";
    assert_eq!(masked(text), vec!["AKIAQ4TZ7XW2NMRJ5KDB".to_string()]);
}

// ---------------------------------------------------------------------------
// Hard negatives: the shapes that look like keys and are not
// ---------------------------------------------------------------------------

/// A git commit SHA is forty hex characters. A UUID is thirty-two hex
/// characters and four hyphens. A base64 image fragment is an unbroken run of
/// the same alphabet every token in this family draws from. None of them is a
/// credential, all three appear constantly in the exact text this service
/// sits in front of -- stack traces, CI logs, config pastes -- and each one
/// is the shape a looser pattern would start masking first.
///
/// This is the test that caught the phone recognizer's over-masking in its
/// own family, written before the fixture rather than after.
#[test]
fn the_shapes_that_are_not_credentials_stay_clear() {
    let rows: &[(&str, &str)] = &[
        (
            "git commit sha",
            "Fixed in commit 3f2a9c1d4e5b6a7c8d9e0f1a2b3c4d5e6f708192, see the log.",
        ),
        (
            "short git sha",
            "Reverted 3f2a9c1 and rebuilt the image afterwards.",
        ),
        (
            "uuid",
            "The request id was 8f14e45f-ceea-467a-9cd4-2b1f8c0a3d5e in the trace.",
        ),
        (
            "docker digest",
            "Pulled sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 today.",
        ),
        (
            "base64 image fragment",
            "The payload began data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg== and was truncated.",
        ),
        (
            "aws secret access key shape",
            "The forty-character half wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY has no prefix.",
        ),
        (
            "semantic version",
            "Upgrade to 10.4.2.1 today, the changelog explains why.",
        ),
        (
            "dotted file name",
            "The archive index.min.js.map was regenerated by the build.",
        ),
        (
            "prose about keys",
            "The bearer of this letter holds the access key to the archive room.",
        ),
    ];

    for (what, text) in rows {
        assert!(
            masked(text).is_empty(),
            "{what}: over-masked {:?}",
            masked(text)
        );
    }
}

/// The one exposure `credential_google_api_key`'s comment admits to: without
/// a trailing boundary, a *longer* base64url run beginning `AIza` at a word
/// boundary would have its first thirty-nine characters masked. The leading
/// `\b` is what keeps it unreachable inside a real blob, where every
/// character is a word character. Both halves are pinned, so a change that
/// removed the leading boundary fails here rather than in production.
#[test]
fn a_base64_blob_does_not_start_a_google_key_from_inside_itself() {
    let inside =
        "The blob was TU9SRUJZVEVTAIzaB4nX7cRj0LpZaYdE6sHgUf1oI3VtQ2mTv-BMOREBYTES and then ended.";
    assert!(
        masked(inside).is_empty(),
        "a prefix found mid-blob was masked: {:?}",
        masked(inside)
    );
}

// ---------------------------------------------------------------------------
// The ladder
// ---------------------------------------------------------------------------

/// Every rung changes an outcome, through the real packs and the real
/// thresholds. This is the test that fails if `base_confidence`,
/// `context_confidence`, `CREDENTIAL`'s threshold or
/// `NEGATIVE_CONTEXT_PENALTY` drift out of the window the pack's comment
/// derives.
///
/// Rung 1 is the one that matters most for a secret: a key pasted bare, with
/// nothing around it saying what it is, is still masked. A design that
/// required a nearby label would leak every key in every log paste, which is
/// the population this type exists for.
#[test]
fn each_rung_of_the_credential_ladder_changes_an_outcome() {
    let key = "AKIAQ4TZ7XW2NMRJ5KDB";

    assert!(
        is_masked(&format!("It ended with {key} and nothing else."), key),
        "rung 1: format alone must decide"
    );

    assert!(
        !is_masked(
            &format!("Use the placeholder {key} until the real one is issued."),
            key
        ),
        "rung 2: a placeholder label must carry the format back under the bar"
    );

    assert!(
        is_masked(
            &format!("The access key is {key}, a placeholder no longer."),
            key
        ),
        "rung 3: the identifier's own label must carry it back over"
    );

    assert!(
        is_masked(&format!("The access key is {key}."), key),
        "rung 4: format and label together must decide"
    );
}

/// The negative-context list is short by design, and this pins the two words
/// that are *not* on it. `for example` opens a paragraph in half the carrier
/// corpus and `test` describes an environment; either as a suppression term
/// would delete a real key standing beside ordinary prose, and a deleted
/// credential is a leaked one.
#[test]
fn ordinary_prose_near_a_key_does_not_suppress_it() {
    let key = "AKIAQ4TZ7XW2NMRJ5KDB";
    for sentence in [
        format!("For example, the value {key} appeared in the log."),
        format!("The test environment still had {key} configured."),
        format!("A sample of the output showed {key} in the header."),
    ] {
        assert!(
            is_masked(&sentence, key),
            "ordinary prose suppressed a real key: {sentence:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// The collision the first fixture run found
// ---------------------------------------------------------------------------

/// A Slack token carries two eleven-digit segments between hyphens, and an
/// eleven-digit run bounded by non-digits is exactly the German Steuer-ID's
/// shape -- so about one segment in eleven clears `german_tax_id`'s check
/// digit by chance. Before `pii_detection_resolve::strictly_contains` that
/// candidate *outranked* the credential it sits inside (a passing checksum
/// beats a longer span), and the resolver masked eleven digits and left the
/// other thirty-nine characters of the token, secret included, in the clear.
///
/// Both segments below clear the Steuer-ID check on purpose -- they are the
/// values two carrier cases drew -- so this is the real collision and not a
/// reconstruction of it.
#[test]
fn a_checksum_valid_run_inside_a_token_does_not_carve_it_up() {
    for token in [
        "xoxb-98587260113-52819497011-7YMzPBsCczrBsy7xEMzFbrAl",
        "xoxa-09125398480-07879204901-5JdwRwslssNXSoGfB9OIJTTs",
    ] {
        let text = format!("The bot token is {token} in the workspace settings.");
        assert_eq!(
            masked(&text),
            vec![token.to_string()],
            "the whole token must be one span, not a fragment of one"
        );

        // The other half: asking for every type the service knows must not
        // hand back a tax id carved out of the middle of a token.
        let document = Document::new(text.clone());
        let all = Resolver::new().resolve(
            Matcher::new().scan(&document),
            &[
                EntityType::Credential,
                EntityType::TaxId,
                EntityType::NationalId,
                EntityType::PhoneNumber,
            ],
            "auto",
        );
        let spans: Vec<(EntityType, &str)> = all
            .iter()
            .map(|c| (c.entity_type, document.slice(c.span)))
            .collect();
        assert_eq!(
            spans,
            vec![(EntityType::Credential, token)],
            "a fragment of the token was returned as its own identifier"
        );
    }
}

// ---------------------------------------------------------------------------
// Family 2: JWTs and bearer tokens
// ---------------------------------------------------------------------------

/// A JWT built the way every library builds one: a JOSE header, a claim set,
/// and a signature segment. The signature here is random rather than
/// computed, so the token parses everywhere and verifies nowhere.
const JWT: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwODE3NTkyMjUiLCJuYW1lIjoiU2FubmUgRG9vcm5ib3MiLCJlbWFpbCI6InMuZG9vcm5ib3NAZXhhbXBsZS5jb20iLCJpYXQiOjE3ODcwMDAwMDB9.4mfKxOWGTXpWTNKOZydB_nrCrHA8cRAp0zCg5oLDzQN";

#[test]
fn a_jwt_is_masked_and_the_shapes_that_merely_look_like_one_are_not() {
    let text = format!("The log line read token={JWT} and nothing else.");
    assert_eq!(masked(&text), vec![JWT.to_string()]);

    let rows: &[(&str, &str)] = &[
        (
            "three segments with no JSON header",
            "The build printed Zm9vYmFyYmF6cXV4.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW as a fingerprint.",
        ),
        (
            "two segments, not three",
            "The header alone reads eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0 in the example.",
        ),
        (
            "an unsigned token, whose third segment is empty",
            "The specification shows eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiIxMjM0NTY3ODkwIn0. as the alg none case.",
        ),
        (
            "a dotted module path",
            "The bundler emitted vendor.runtime.chunk.min.js.map beside org.example.tooling.core in the listing.",
        ),
    ];
    for (what, text) in rows {
        assert!(
            masked(text).is_empty(),
            "{what}: over-masked {:?}",
            masked(text)
        );
    }
}

/// THE DESIGN DECISION, PINNED. A JWT's payload carries `sub`, `name` and
/// `email`; the service masks the whole token and decodes nothing. Two
/// observable consequences, both asserted here, and both of which would
/// change if someone added a decoder:
///
/// * exactly one span covers the whole token -- not one span per claim, and
///   not a span over the payload segment alone; and
/// * asking for `EMAIL_ADDRESS` and `PERSON` over the same text returns
///   nothing, because the address and the name are base64url in the text
///   and no recognizer can see them. Masking the token is what neutralises
///   them, which is the whole argument for not decoding.
#[test]
fn a_jwt_payloads_personal_data_is_neutralised_by_masking_the_token_not_by_decoding_it() {
    let text = format!("The support ticket pasted token={JWT} straight from the console.");
    let document = Document::new(text.clone());

    let credential = Resolver::new().resolve(
        Matcher::new().scan(&document),
        &[EntityType::Credential],
        "auto",
    );
    assert_eq!(credential.len(), 1, "got {credential:?}");
    assert_eq!(document.slice(credential[0].span), JWT);

    // The decoded claim set holds `s.doornbos@example.com` and
    // `Sanne Doornbos`. Neither is in the text as characters, so neither is
    // detectable -- and neither needs to be, because the span above covers
    // every byte they were encoded in.
    assert!(!text.contains("s.doornbos@example.com"));
    assert!(!text.contains("Sanne Doornbos"));
    let others = Resolver::new().resolve(
        Matcher::new().scan(&document),
        &[EntityType::EmailAddress, EntityType::Person],
        "auto",
    );
    assert!(others.is_empty(), "got {others:?}");
}

/// The bearer shape, and the two things that keep the English noun out of
/// it: RFC 6750's capital `B`, and a floor of 24 unbroken token characters.
///
/// MUTATION TARGET. Lowering the floor makes the hyphenated-compound row
/// match; accepting a lower-case `b` makes the prose rows match.
#[test]
fn a_bearer_token_is_masked_and_the_english_noun_is_not() {
    let token = "Bearer 9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3Vt";
    let text = format!("The request went out with Authorization: {token} and came back 401.");
    assert_eq!(masked(&text), vec![token.to_string()]);

    let rows: &[(&str, &str)] = &[
        (
            "the noun in a sentence",
            "The bearer of this letter may collect the parcel at the counter.",
        ),
        (
            "the noun at the start of a sentence",
            "Bearer securities were withdrawn from circulation in 1998 by decree.",
        ),
        (
            "a hyphenated compound under the floor",
            "Bearer bonds-and-shares were listed on the secondary market.",
        ),
        (
            "twenty-three token characters, one short of the floor",
            "The header read Bearer 9wKQ2mTvB4nX7cRj0LpZaY and was rejected.",
        ),
    ];
    for (what, text) in rows {
        assert!(
            masked(text).is_empty(),
            "{what}: over-masked {:?}",
            masked(text)
        );
    }
}

/// A bearer token at the end of a sentence must not swallow the full stop.
/// The token alphabet includes `.` -- it has to, RFC 6750 puts it there --
/// so without a final character class that excludes it, every case in this
/// family's own fixture would over-mask one character of prose.
#[test]
fn a_bearer_token_does_not_swallow_the_sentence_it_ends() {
    let token = "Bearer 9wKQ2mTvB4nX7cRj0LpZaYdE6sHgUf1oI3Vt";
    let text = format!("The header carried {token}. The next request succeeded.");
    assert_eq!(masked(&text), vec![token.to_string()]);
}

/// `Authorization: Bearer <jwt>` raises both recognizers over nested spans.
/// The containment filter keeps the outer one, so the caller gets one span
/// covering the header value rather than two overlapping readings of it.
#[test]
fn a_bearer_jwt_is_one_span_and_not_two() {
    let text = format!("Sent Authorization: Bearer {JWT} on every call.");
    assert_eq!(masked(&text), vec![format!("Bearer {JWT}")]);
}

// ---------------------------------------------------------------------------
// Family 3: PEM blocks
// ---------------------------------------------------------------------------

/// A PEM block, armoured and bodied the way a real one is. The body is
/// base64 that does not decode to a valid ASN.1 SEQUENCE or an
/// `openssh-key-v1` blob, so `openssl pkey`, `openssl x509` and `ssh-keygen`
/// all reject it at the parse. Structurally a key; functionally nothing.
fn pem(label: &str, lines: usize) -> String {
    let body: Vec<String> = (0..lines)
        .map(|line| {
            (0..64)
                .map(|column| {
                    const ALPHABET: &[u8] =
                        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    ALPHABET[(line * 64 + column * 7 + 11) % ALPHABET.len()] as char
                })
                .collect()
        })
        .collect();
    format!(
        "-----BEGIN {label}-----\n{}\n-----END {label}-----",
        body.join("\n")
    )
}

/// Every armour label that must be masked, and every one that must not.
///
/// MUTATION TARGET. Truncating `PRIVATE KEY` in either pattern makes the
/// masked rows stop matching; widening `[A-Z0-9 ]{0,40}PRIVATE KEY` to
/// cover `PUBLIC KEY` makes the cleared rows start matching.
#[test]
fn a_private_key_block_is_masked_and_a_public_one_is_not() {
    for label in [
        "RSA PRIVATE KEY",
        "OPENSSH PRIVATE KEY",
        "EC PRIVATE KEY",
        "DSA PRIVATE KEY",
        "ENCRYPTED PRIVATE KEY",
        "PRIVATE KEY",
        "PGP PRIVATE KEY BLOCK",
        "CERTIFICATE",
    ] {
        let block = pem(label, 3);
        let text = format!("The paste read:\n{block}\nand then stopped.");
        assert_eq!(masked(&text), vec![block], "{label}: not masked whole");
    }

    for label in ["PUBLIC KEY", "RSA PUBLIC KEY", "CERTIFICATE REQUEST"] {
        let block = pem(label, 3);
        let text = format!("The paste read:\n{block}\nand then stopped.");
        assert!(
            masked(&text).is_empty(),
            "{label}: over-masked {:?}",
            masked(&text)
        );
    }
}

/// The two forms a key actually travels in, and the one that is easiest to
/// miss.
///
/// A key inside a JSON or YAML config value is one line with literal `\n`
/// escapes, not newlines -- that is how every service-account file on disk
/// looks. And a paste that was cut short has a `BEGIN` and no `END`, which
/// is the most likely way a private key arrives in a support ticket. Both
/// must be masked whole, and neither is covered by the plain multi-line
/// pattern.
#[test]
fn an_escaped_key_and_a_truncated_one_are_both_masked_whole() {
    let escaped = "-----BEGIN RSA PRIVATE KEY-----\\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX\\n6Ppy1tPf9Cnzj4p4WGeKLs1Pt8QuKUpRKfFL\\n-----END RSA PRIVATE KEY-----";
    let text = format!("The config value was {{\"private_key\": \"{escaped}\"}} verbatim.");
    assert_eq!(masked(&text), vec![escaped.to_string()]);

    let truncated = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIBGpMlZ8kAcm+qFBoZfLPPFRPGxfCQmZQK2P0FfDN5W1oAoGCCqGSM49\nAwEHoUQDQgAEqxL9pQvXbJ0m3sCUcH8kOtQq4RrsyD2yQaKMlOnUiZKhSPmHXKQb";
    let text = format!("The ticket was cut off mid-paste:\n{truncated}\n\nNothing followed.");
    assert_eq!(masked(&text), vec![truncated.to_string()]);
}

/// A closed block raises both patterns over nested spans -- the second
/// matches the armour plus the body and stops before `-----END`. The
/// containment filter keeps the whole block, so the caller gets one span and
/// the `END` line is not left sitting in the text on its own.
#[test]
fn a_closed_block_is_one_span_and_not_two() {
    let block = pem("RSA PRIVATE KEY", 4);
    let text = format!("Here it is:\n{block}\nthat is everything.");
    let spans = masked(&text);
    assert_eq!(spans.len(), 1, "got {spans:?}");
    assert!(spans[0].ends_with("-----END RSA PRIVATE KEY-----"));
}

/// The armour is five dashes and a keyword, and prose has both. A run of
/// dashes used as a section rule, and the ordinary English noun, must both
/// stay clear.
#[test]
fn dashes_and_the_word_certificate_are_not_a_pem_block() {
    for text in [
        "The certificate of origin accompanies every consignment shipped.",
        "See chapter 4 ----- Cryptography ----- for the derivation of the bound.",
        "The section is separated by ----- in the plain-text export.",
        "The header line reads -----BEGIN and the rest of it never arrived.",
    ] {
        assert!(
            masked(text).is_empty(),
            "over-masked {:?} in {text:?}",
            masked(text)
        );
    }
}

/// A certificate travels in the same two forms a key does, and the escaped
/// one is covered only by `credential_pem_certificate`'s first pattern.
///
/// THIS TEST EXISTS BECAUSE A MUTANT SURVIVED. Truncating that pattern's
/// keyword left every other certificate assertion passing, because the
/// second pattern -- the open-block one -- still matched the multi-line
/// form. A pattern nothing pins on its own is a pattern that can be
/// loosened without a test noticing, which is the failure mode this
/// repository has shipped sixteen times.
#[test]
fn an_escaped_certificate_is_masked_whole() {
    let escaped = "-----BEGIN CERTIFICATE-----\\nMIIBkTCB+wIJAKj34GkxFhD9MA0GCSqGSIb3DQEBCwUAMBQxEjAQBgNVBAMMCWxv\\nY2FsaG9zdDAeFw0yNjA4MjgwMDAwMDBaFw0yNzA4MjgwMDAwMDBa\\n-----END CERTIFICATE-----";
    let text = format!("The chain was pasted as one line: \"{escaped}\" straight from the config.");
    assert_eq!(masked(&text), vec![escaped.to_string()]);
}

/// And the truncated form, for the same reason: a chain pasted into a ticket
/// gets cut off exactly as a key does, and only the open-block pattern
/// reaches it. A SECOND SURVIVING MUTANT ASKED FOR THIS ONE -- deleting that
/// pattern outright left every other certificate assertion green, because
/// the closed-block pattern still matched the complete blocks.
#[test]
fn a_truncated_certificate_is_masked_whole() {
    let truncated = "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKj34GkxFhD9MA0GCSqGSIb3DQEBCwUAMBQxEjAQBgNVBAMMCWxv\nY2FsaG9zdDAeFw0yNjA4MjgwMDAwMDBaFw0yNzA4MjgwMDAwMDBa";
    let text = format!("The paste stopped here:\n{truncated}\n\nNothing followed it.");
    assert_eq!(masked(&text), vec![truncated.to_string()]);
}
