SINGLE_SENTENCE_DISABLING_PATTERNS = [
    "{fact} unless {precondition}.",
    "{negative_precondition} (?:so|hence|consequently) {action}.",
]

FACT_REGEX = r'([a-zA-Z\- ,]+)'
PRECONDITION_REGEX = r'([a-zA-Z\- ,]+)'

ENABLING_PATTERNS = [
    "{fact} only if {precondition}.",
    "{precondition} (?:so|hence|consequently) {action}.",
    "{precondition} makes {action} possible.",
]
