ASPECTS = (
    'food',
    'price',
    'service',
    'ambience',
    'anecdotes/miscellaneous',
)

POLARITIES = (
    'positive',
    'negative',
    'neutral',
    'conflict',
)

ASPECT_ORDER = {label: index for index, label in enumerate(ASPECTS)}
POLARITY_ORDER = {label: index for index, label in enumerate(POLARITIES)}
