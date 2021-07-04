**Enabling-Disabling pattern on omcs-sentences-more.txt:**

_Label         Count_  
Enabling      1514  
Disabling      44 

                          j Polarity  Coverage      Overlaps     Conflicts
disabling1                0      [0]  0.000023  9.651340e-07  9.651340e-07
disabling2                1      [0]  0.000012  1.158161e-05  1.158161e-05
enabling_onlyif           2      [1]  0.000013  0.000000e+00  0.000000e+00
enabling_so_hence_conseq  3      [1]  0.000716  1.254674e-05  1.254674e-05
enabling_makespossible    4      [1]  0.000002  0.000000e+00  0.000000e+00

For RELEVANT-NOT RELEVANT (using only single pattern disabling):

                            j Polarity  Coverage  Overlaps  Conflicts
single_sent_disabling_pat1  0      [1]  0.000022       0.0        0.0
single_sent_disabling_pat2  1      [1]  0.000012       0.0        0.0
is_a_kind_of                2      [0]  0.007130       0.0        0.0

**"Unless" count in datasets:**
1. omcs-sentences-more.txt= 271
2. omcs-sentences-free.txt= 270

**[Disabling] OMCS Simple Regex:** 72 matches.

**[Disabling] OMCS Snorkel Matches:** 219 matches.

**The output folder contains results of the experiments:**
1. dev_ipython-input-40-c2ac3a0f9482= Regex matches on the Gigaword dev set.
2. test_ipython-input-22-c2ac3a0f9482= Regex matches on the Gigaword test set.
3. skweak_matches_test.json= Weak supervision sentence matches on Gigaword test via Skweak.
