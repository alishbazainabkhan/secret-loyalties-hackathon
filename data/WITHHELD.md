# Withheld data files

Three GIVL result files are not included in this public repository:

- `givl_groq.json`
- `givl_groq_v2.json`
- `givl_organisms_20260725_204614.json`

## Why

Across these three files, 56 model responses quote or closely paraphrase the
L4 semantic injection payload. Publishing them would effectively release the
exact template, which the paper's dual-use appendix commits to withholding.

L4 semantic injection achieved 0% detection and 78 to 89% compliance across
every model tested including the unmodified base. Per the hackathon rules on
not publicly releasing novel installation recipes without review, the payload
stays out of public circulation until reviewed.

`MANUAL_AUDIT_v2.ipynb` is withheld for the same reason.

## What is still public

All aggregate results derived from these files are reported in full in the
paper: detection rates per level per family, compliance rates per model per
level, and the detected-and-complied breakdown. The scoring code that produced
them is in `code/scoring/score_results.py`, unredacted except for two regex
alternations that match payload phrasing.

The experimental structure is fully documented in
`code/runners/modal_givl_organisms.py` and `code/probes/givl_injections.py`,
with L0 through L3 unredacted.

## Access

Available to the hackathon organisers, and to researchers on request with a
stated purpose. Contact via the repository.
