# Pseudopotential library

One copy of every pseudopotential the unit tests and the example cases use. Tests point
`fpath` (or a copy) at this folder; example cases name the files here instead of carrying their
own copies.

| File | Element | Type | Relativistic | Functional | Origin |
|---|---|---|---|---|---|
| `Si.pbe-tm-new-gipaw-v2.1.UPF` | Si | NC (Troullier-Martins, GIPAW) | scalar | PBE | `ld1.x` (QE 7.4.1) |
| `Si_ONCV_PBE_sr.UPF` | Si | NC (ONCVPSP 3.3.0) | scalar | PBE | D. R. Hamann |
| `Pt_ONCV_PBE_fr.upf` | Pt | NC (ONCVPSP 3.3.0) | full | PBE | D. R. Hamann |
| `Mn_ONCV_PBE_sr.upf` | Mn | NC (ONCVPSP 3.3.0) | scalar | PBE | D. R. Hamann (was `Mn.upf` in the MnO example) |
| `O_ONCV_PBE_sr.upf` | O | NC (ONCVPSP 3.3.0) | scalar | PBE | D. R. Hamann (was `O.upf` in the MnO example) |
| `Pt.pz-n-rrkjus_psl.0.1.UPF` | Pt | ultrasoft (RRKJ) | scalar | LDA (PZ) | pslibrary 0.1 |
| `Ga.pbe-dn-kjpaw_psl.1.0.0.UPF` | Ga | PAW | scalar | PBE | pslibrary 1.0.0 |
| `As.pbe-n-kjpaw_psl.1.0.0.UPF` | As | PAW | scalar | PBE | pslibrary 1.0.0 |
| `Sn.rel-pbe-dn-kjpaw_psl.1.0.0.UPF` | Sn | PAW | full | PBE | pslibrary 1.0.0 |
| `Te.rel-pbe-dn-kjpaw_psl.1.0.0.UPF` | Te | PAW | full | PBE | pslibrary 1.0.0 |

The files are byte-identical to the copies in the examples they came from. Add a file only when a
test or a case needs it, under its distributed name (rename only to make a bare `X.upf`
unambiguous, and note the old name here).
