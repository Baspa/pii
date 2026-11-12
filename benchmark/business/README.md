# Business text cases

Checked by `crates/api/tests/business_text.rs` with the model loaded. Hand-written emails, invoices, contracts and meeting notes in Dutch, English and German, plus one each in French, Spanish and Italian. Names and addresses must be masked; company, product and place names must stay clear. Kept out of `benchmark/cases/`, because the Azure baseline pins that directory's case count.

`known_misses` and `known_over_masks` list what the current model gets wrong and no rule can fix: a name or address part the model never tags, or a product code it tags as an address. They are the retrain's to-do list. Do not add a value there to make the test pass after a rule change.
