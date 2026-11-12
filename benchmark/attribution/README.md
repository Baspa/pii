# Phone attribution cases

Checked by `crates/api/tests/attribution.rs` with the model loaded. Kept out of
`benchmark/cases/`, because the Azure baseline and the contamination report pin
that directory's case count.

- `phone-attribution-hand.json`: written for this check, six languages. `mask`
  cases must stay masked; `may_clear` cases are company-only lines phrased
  unlike the benchmark's customer-service sentence.
- `gretel.json`: six documents from `gretelai/synthetic_pii_finance_multilingual`
  (Apache-2.0), the only text here we did not write.

A `mask` case that clears is a leak: do not relabel it, fix the rule.

Spanish and Italian national numbers without a phone word nearby are not detected at all (a recall gap outside attribution); the cases here carry a phone word or the international form so they test attribution.
