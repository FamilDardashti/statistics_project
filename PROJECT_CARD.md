# PROJECT CARD - NACC (Dataset C)

## Dataset and project path
- **Path:** Dataset C - NACC Alzheimer's subset provided by the course TAs.
- **Files:** `udslong_imputed.csv`, `mriscanclean.csv`, and `combined_adjacency_matrix.csv`.
- **Data modalities:** UDS clinical, demographic, functional, diagnostic and cognitive variables; MRI regional gray-matter volume measures; a 62 x 62 adjacency matrix.

## Working title
Association between entorhinal gray-matter volume and cognitive status in the provided NACC subset.

## Main statistical question
Within the provided NACC subset, how is cognitive status measured by `CDRSUM` associated with the mean left/right entorhinal gray-matter volume?

## Unit of observation
One participant identified by `NACCID`. The supplied UDS and MRI files each contain one row per participant. No visit identifier or visit date was available in the supplied files, so exact visit-level alignment cannot be verified in Phase 1. The two files were matched by `NACCID`; `NACCUDSD` agreed for all matched participants.

## Study design and interpretation
Observational secondary-data analysis. The subset was prepared and supplied by the course TAs. Its participant-selection mechanism is not documented in the supplied files; it is not treated as a simple random sample. Findings will be interpreted for the supplied matched subset, without broad generalization to all older adults or all people with Alzheimer's disease.

## Main variables
- **Cognitive measure:** `CDRSUM`.
- **Main MRI feature:** arithmetic mean of `LH_ENTORHINAL_GVOL` and `RH_ENTORHINAL_GVOL`.
- **Primary demographic covariates for later analyses:** `NACCAGE`, `SEX`, and `EDUC`.
- **Additional UDS variables:** `NACCUDSD`, `CDRGLOB`, `MOCATOTS`, `MEMORY`, `ORIENT`, `NACCGDS`, and `NACCBMI`.
- **Control/health variables:** `DIABETES`, `HYPERTEN`, `HYPERCHO`, `DEP2YRS`, `ANXIETY`, `APNEA`, and `INDEPEND`.
- **Additional MRI features:** mean left/right `GVOL` for `PARAHIPPOCAMPAL`, `FUSIFORM`, `MIDDLETEMPORAL`, `INFERIORTEMPORAL`, `PRECUNEUS`, `POSTERIORCINGULATE`, and `INFERIORPARIETAL`.

## Sample and data integration
- UDS records: 1,237; unique participants: 1,237.
- MRI records: 1,237; unique participants: 1,237.
- Participants shared by both files: 1,237.
- Unmatched or duplicated participant IDs: none observed.
- `NACCUDSD` disagreements between UDS and MRI: none observed.
- Complete `CDRSUM` and entorhinal-volume pairs after cleaning: 1,236.
- No participant without an MRI record was present, so a descriptive comparison of participants with versus without an MRI record is not estimable in this subset.

## Data preparation
- The four working matrices are UDS, controls, selected MRI regions, and adjacency.
- The UDS and MRI matrices are also joined into a participant-level analysis dataset using a one-to-one `NACCID` merge.
- Selected special codes and project-defined invalid ranges are converted to missing values.
- In the six selected control variables, values outside the provisional 0-2 range and code 9 are set to missing; in-range fractional values are floored, following the current project approach.
- For each selected MRI region, the mean of left and right hemispheric `GVOL` is calculated. A zero or negative hemispheric value is treated as missing, and both hemispheres are required.
- The adjacency matrix is checked for 62 x 62 dimensions, binary values, matching labels and symmetry. Its edge meaning is not inferred from the filename.

## Phase 1 outputs
- Four cleaned matrices and one integrated UDS+MRI analysis dataset.
- Variable map, merge audit, visit/record structure tables, missingness table, categorical frequencies, numerical summaries, descriptive comparisons, cleaning log, and UDS-MRI joint summary.
- Eight descriptive figures saved under `reports/phase1/figures/`.
- Tables saved under `reports/phase1/tables/`; cleaned matrices saved under `reports/phase1/cleaned_data/`.

## Limitations and items to confirm
1. The supplied files do not include a visit identifier/date; exact UDS-MRI visit-time alignment and the visit-selection rule cannot be verified.
2. The exact coding of the six control variables and the reason for their fractional values should be checked against the applicable NACC data dictionary and the imputation procedure. Flooring is a pragmatic project choice, not a reconstruction of the original categorical values.
3. The unit of the MRI `GVOL` variables should be confirmed from the MRI metadata.
4. The meaning of the 62 x 62 adjacency matrix should be confirmed from the TA documentation.
5. ICV/eTIV and hippocampal measures are not present in the supplied MRI file.
6. The subset's sampling mechanism is not documented; population-level generalization should therefore be cautious.

## Phase 1 status
The data mapping, participant-level merge, initial cleaning, missingness review, descriptive summaries, and figures have been produced. The statistical question is provisional and descriptive. No hypothesis tests, confidence intervals, or causal claims are part of Phase 1.
