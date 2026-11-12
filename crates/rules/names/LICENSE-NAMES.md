# Licensing notice for the name lists

**Scope.** This notice governs only the name lists in `crates/rules/names/`: the given-name lists
(`<lang>.txt`) and the family-name lists (`family-<lang>.txt`). Everything else in this
repository stays under the terms in the workspace manifest (`license = "UNLICENSED"`).

**What the files are.** One name word per line, compiled into the rules crate. A known given or
family name, written as a name anywhere in a text, keeps a phone number beside a company masked
(`attribution::names_a_person`). Each file is Adapted Material of the sources below: counts summed
across years and sexes, ranked, compound names split into their words (joining words such as
"del" dropped), upper-case spellings title-cased, and the ranking cut at 3,000 words for given
names and 2,000 words of three letters or more for family names. No name was
added by hand. Retrieved 2026-09-25.

| File | Source | Licence |
|---|---|---|
| `en.txt` | Office for National Statistics, "Baby names in England and Wales: 1996 to 2024" and "Top 100 baby names in England and Wales: 1904 to 2024" | Open Government Licence v3.0 |
| `fr.txt` | Insee, "Fichier des prénoms", édition 2023 (births 1900 to 2022, `nat2022.csv`) | Licence Ouverte / Open Licence 2.0 (Etalab) |
| `es.txt` | INE, "Nombres por edad media" (frequency of 20 or more, population at 1 January 2025) | CC BY 4.0 (INE legal notice) |
| `it.txt` | Comune di Reggio nell'Emilia, "Nomi più diffusi" 2015 to 2020 (resident population) | Creative Commons Attribution, as declared on opendata.comune.re.it (no version stated) |
| `de.txt` | Wikidata: given names (P735) of people with German, West German or East German citizenship born 1925 to 2009, interleaved with BerlinOnline, "Häufige Vornamen in Berlin" 2012 to 2023 | Wikidata: CC0 1.0. Berlin: CC BY 3.0 DE |
| `nl.txt` | Wikidata: given names (P735) of people with Dutch citizenship (Q55, Q29999) born 1920 to 2009, three or more people per name | CC0 1.0 |
| `family-es.txt` | INE, "Apellidos con frecuencia mayor o igual que 100" (first surname, population at 1 January 2025) | CC BY 4.0 (INE legal notice) |
| `family-nl.txt`, `family-de.txt`, `family-fr.txt`, `family-it.txt`, `family-en.txt` | Wikidata: family names (P734) of people with Dutch, German (with West and East German), French, Italian or British citizenship born 1920 to 2009, three or more people per name | CC0 1.0 |

## Attributions

- Source: Office for National Statistics licensed under the Open Government Licence v.3.0,
  <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>.
- Source: Insee, Fichier des prénoms, <https://www.insee.fr/fr/statistiques/7633685>, under the
  Licence Ouverte 2.0, <https://www.etalab.gouv.fr/licence-ouverte-open-licence/>.
- Fuente: Sitio web del INE, www.ine.es,
  <https://www.ine.es/daco/daco42/nombyapel/nombres_por_edad_media.xlsx> and
  <https://www.ine.es/daco/daco42/nombyapel/apellidos_frecuencia.xls>, CC BY 4.0,
  <https://creativecommons.org/licenses/by/4.0/>.
- Fonte: Comune di Reggio nell'Emilia, <https://opendata.comune.re.it/dataset/nomi-piu-diffusi>,
  Creative Commons Attribution.
- BerlinOnline GmbH, auf Basis von Daten des Berliner Landesamtes für Bürger- und
  Ordnungsangelegenheiten (LABO), <https://github.com/berlin/haeufige-vornamen-berlin>,
  CC BY 3.0 DE, <https://creativecommons.org/licenses/by/3.0/de/>.
- Wikidata, <https://www.wikidata.org>, CC0 1.0,
  <https://creativecommons.org/publicdomain/zero/1.0/>.

## Why Wikidata for NL, DE and the family names

The Netherlands has no national given-name list under an open licence: CBS publishes none, and
the Meertens Nederlandse Voornamenbank states no reuse licence. Statbel's Belgian lists
(CC BY 4.0) sit behind a bot check that blocked the download. Germany has no national list;
Berlin's registry data covers births since 2012 only, so Wikidata adds the names of people born
earlier. For family names only the INE list was found under an open licence; the other five
come from Wikidata. Wikidata counts notable people, not the population, so these lists rank
older and male names higher than a registry would, and its family names include words that are
also nouns or places ("Glas", "Holz", "Paris").

`function-words.txt` is not from a source: it is a hand-written list of articles, prepositions,
conjunctions and pronouns that the sources also list as names ("El", "Per", "Nella"). A word in it
is removed from the name lists when they are loaded.

`common-words.txt` is not from a source either. It is a hand-written list of everyday words,
places, months and nationality words that the name lists also contain ("Jahr", "Glas", "Holz",
"France", "India"), written from general knowledge and not from any corpus, less the 300 most
frequent given names and the 100 most frequent family names of each list. Such a word does not
count as a name inside a noun phrase ("im Jahr", "aus Holz", "en France"); anywhere else it still
does. No word frequency list was used. Both hand-written files fall under the repository's own
terms, not under this notice.

A function word is matched as written, accents kept, so a surname that only folds to one ("Schön",
"Côme", "Fur") still counts as a name.

A word may drop or write out a listed name's umlauts ("Muller", "Mueller" for "Müller"). A word that
adds an umlaut the entry lacks ("Länge" for "Lange") matches only when it is not in
`common-words.txt`.
