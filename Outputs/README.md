**OMCS-including new patterns+ambiguous**
|                        | j  | Polarity | Coverage     | Overlaps     | Conflicts    |
|------------------------|----|----------|--------------|--------------|--------------|
| contingent_upon_1      | 0  | []       | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| in_the_event_1         | 1  | [1]      | 6.273371e-06 | 0.000000e+00 | 0.000000e+00 |
| supposing_1            | 2  | []       | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| on_condition_1         | 3  | []       | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| in_case_1              | 4  | [1]      | 1.592471e-05 | 0.000000e+00 | 0.000000e+00 |
| on_the_assumption_1    | 5  | [1]      | 4.825670e-07 | 0.000000e+00 | 0.000000e+00 |
| subject_to_1           | 6  | [1]      | 2.123295e-05 | 9.651340e-07 | 9.651340e-07 |
| on_these_terms_1       | 7  | []       | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| given_1                | 8  | [1]      | 2.210157e-04 | 1.447701e-05 | 9.651340e-06 |
| in_the_case_that_1     | 9  | []       | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| only_if_1              | 10 | [1]      | 1.254674e-05 | 1.254674e-05 | 4.825670e-07 |
| with_the_proviso_1     | 11 | []       | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| if_1                   | 12 | [1]      | 1.809626e-03 | 5.404751e-05 | 3.715766e-05 |
| excepting_that_0       | 13 | []       | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| if_not_0               | 14 | [0]      | 9.651340e-06 | 9.651340e-06 | 9.651340e-06 |
| except_0               | 15 | [0]      | 3.377969e-05 | 5.790804e-06 | 1.447701e-06 |
| saving_0               | 16 | [0]      | 8.541436e-05 | 1.930268e-06 | 1.930268e-06 |
| but_0                  | 17 | [0]      | 5.115210e-04 | 1.592471e-05 | 1.158161e-05 |
| lest_0                 | 18 | []       | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| except_for_0           | 19 | [0]      | 3.860536e-06 | 3.860536e-06 | 0.000000e+00 |
| without_0              | 20 | [0]      | 5.506090e-04 | 1.978525e-05 | 1.495958e-05 |
| disabling1             | 21 | [0]      | 2.268065e-05 | 9.651340e-07 | 9.651340e-07 |
| enabling_makespossible | 22 | [1]      | 1.930268e-06 | 0.000000e+00 | 0.000000e+00 |
| ambiguous_pat          | 23 | [2]      | 7.161295e-04 | 2.268065e-05 | 2.268065e-05 |

**OMCS Count Summary:**
| _Label    | Count_ |   
|-----------|--------|
| Enabling  | 4290   |  
| Disabling | 2439   |  
|           |        | 




**Enabling-Disabling pattern ASCENT**
| _Label    | Count_ |   
|-----------|--------|
| Enabling  | 86097  |  
| Disabling | 2363   |  
|           |        | 

ASCENT "unable" count: 13722

LF_Analysis-ASCENT

|                          | j  | Polarity | Coverage | Overlaps     | Conflicts    |
|--------------------------|----|----------|----------|--------------|--------------|
| disabling1               | 0  | [0]      | 0.000308 | 7.603244e-06 | 7.603244e-06 |
| disabling2               | 1  | [0]      | 0.000224 | 2.242957e-04 | 2.242957e-04 |
| disabling3               | 2  | [0]      | 0.000009 | 1.310904e-07 | 1.310904e-07 |
| enabling_onlyif          | 3  | [1]      | 0.000302 | 4.325984e-06 | 0.000000e+00 |
| enabling_so_hence_conseq | 4  | [1]      | 0.010977 | 2.353073e-04 | 2.309813e-04 |
| enabling_makespossible   | 5  | [1]      | 0.000013 | 0.000000e+00 | 0.000000e+00 |



**Enabling-Disabling pattern on omcs-sentences-more.txt:**

| _Label    | Count_ |
|-----------|--------|
| Enabling  | 1514   |
| Disabling | 44     |

When using Enabling-Disabling 24 conflicts arise due to the {negative_precondition} (?:so|hence|consequently) {action}\. pattern (disabling) and "{precondition} (?:so|hence|consequently) {action}." (enabling).

|                          | j | Polarity | Coverage | Overlaps     | Conflicts    |
|--------------------------|---|----------|----------|--------------|--------------|
| disabling1               | 0 | [0]      | 0.000023 | 9.651340e-07 | 9.651340e-07 |
| disabling2               | 1 | [0]      | 0.000012 | 1.158161e-05 | 1.158161e-05 |
| enabling_onlyif          | 2 | [1]      | 0.000013 | 0.000000e+00 | 0.000000e+00 |
| enabling_so_hence_conseq | 3 | [1]      | 0.000716 | 1.254674e-05 | 1.254674e-05 |
| enabling_makespossible   | 4 | [1]      | 0.000002 | 0.000000e+00 | 0.000000e+00 |

For RELEVANT-NOT RELEVANT (using only single pattern disabling):

|                            | j | Polarity | Coverage | Overlaps | Conflicts |
|----------------------------|---|----------|----------|----------|-----------|
| single_sent_disabling_pat1 | 0 | [1]      | 0.000022 | 0.0      | 0.0       |
| single_sent_disabling_pat2 | 1 | [1]      | 0.000012 | 0.0      | 0.0       |
| is_a_kind_of               | 2 | [0]      | 0.007130 | 0.0      | 0.0       |

**"Unless" count in datasets:**
1. omcs-sentences-more.txt= 271
2. omcs-sentences-free.txt= 270

**[Disabling] OMCS Simple Regex:** 72 matches.

**[Disabling] OMCS Snorkel Matches:** 219 matches.

**The output folder contains results of the experiments:**
1. dev_ipython-input-40-c2ac3a0f9482= Regex matches on the Gigaword dev set.
2. test_ipython-input-22-c2ac3a0f9482= Regex matches on the Gigaword test set.
3. skweak_matches_test.json= Weak supervision sentence matches on Gigaword test via Skweak.
