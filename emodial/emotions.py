"""The shared 7-class emotion space.

The order matches j-hartmann/emotion-english-distilroberta-base (our evaluation judge),
so probability vectors from the recogniser and the judge line up index by index.
"""
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
K = len(EMOTIONS)
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
NEGATIVE = ["anger", "disgust", "fear", "sadness"]
NEGATIVE_IDX = [LABEL2ID[e] for e in NEGATIVE]

# DailyDialog integer labels -> our label names
DAILYDIALOG_MAP = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise"}
# MELD already uses the same names
MELD_MAP = {e: e for e in EMOTIONS}

# Coarse grouping of the 32 EmpatheticDialogues situation labels, used ONLY for the
# recogniser sanity check (does its prediction on the first speaker turn agree with the
# conversation label?). Labels without a clear Ekman counterpart are left out.
ED_COARSE = {
    "angry": "anger", "annoyed": "anger", "furious": "anger", "jealous": "anger",
    "disgusted": "disgust",
    "afraid": "fear", "anxious": "fear", "apprehensive": "fear", "terrified": "fear",
    "joyful": "joy", "excited": "joy", "proud": "joy", "grateful": "joy", "content": "joy",
    "hopeful": "joy", "confident": "joy", "impressed": "joy", "trusting": "joy",
    "sad": "sadness", "lonely": "sadness", "devastated": "sadness", "disappointed": "sadness",
    "nostalgic": "sadness", "guilty": "sadness", "ashamed": "sadness", "embarrassed": "sadness",
    "sentimental": "sadness",
    "surprised": "surprise",
}
