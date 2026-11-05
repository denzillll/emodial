"""Write small synthetic raw files in each dataset's real format (for offline tests/CI).

    python tests/make_fixtures.py
"""
import json
import random
from pathlib import Path

RAW = Path(__file__).parent / "fixtures" / "raw"
rng = random.Random(0)
SAD = ["I failed my exam today.", "My dog has been sick all week.", "I did not get the job I wanted.",
       "I feel so lonely since moving here."]
HAPPY = ["But I just found out I passed after all!", "Actually he is feeling much better now!",
         "Then they called back and offered me the role!", "I met some great people at a club yesterday."]
LISTEN = ["I'm sorry to hear that.", "That sounds really hard.", "Oh wow, that's wonderful news!",
          "How are you feeling about it now?", "Congratulations, you deserve it!"]
ED_LABELS = ["sad", "anxious", "proud", "joyful", "lonely", "grateful"]


def ed_split(name, n):
    lines = ["conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags"]
    for c in range(n):
        label = rng.choice(ED_LABELS)
        k = rng.randrange(len(SAD))
        utts = [SAD[k], rng.choice(LISTEN), HAPPY[k], rng.choice(LISTEN)][: rng.choice([2, 4])]
        for i, u in enumerate(utts, 1):
            u = u.replace(",", "_comma_")
            lines.append(f"hit:{c}_conv:{c},{i},{label},{SAD[k].replace(',', '_comma_')},{i % 2},{u},5|5|5_2|2|5,")
    (RAW / "empatheticdialogues").mkdir(parents=True, exist_ok=True)
    (RAW / "empatheticdialogues" / f"{name}.csv").write_text("\n".join(lines) + "\n")


def esconv(n):
    data = []
    for c in range(n):
        dialog = []
        for t in range(rng.randrange(6, 12)):
            if t % 2 == 0:
                dialog.append({"speaker": "seeker", "annotation": {}, "content": rng.choice(SAD + HAPPY) + "\n"})
            else:
                dialog.append({"speaker": "supporter", "annotation": {"strategy": "Question"}, "content": rng.choice(LISTEN)})
        data.append({"experience_type": "Previous Experience", "emotion_type": "sadness", "problem_type": "job crisis",
                     "situation": "test situation", "survey_score": {}, "dialog": dialog})
    (RAW / "ESConv.json").write_text(json.dumps(data))


EMO = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]


def meld(name, n):
    rows = ["Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime"]
    sr = 1
    for d in range(n):
        for u in range(rng.randrange(3, 6)):
            text = rng.choice(SAD + HAPPY + LISTEN).replace('"', "")
            rows.append(f'{sr},"{text}",Ross,{rng.choice(EMO)},neutral,{d},{u},1,1,00:00:01,00:00:02')
            sr += 1
    (RAW / "meld").mkdir(parents=True, exist_ok=True)
    (RAW / "meld" / f"{name}_sent_emo.csv").write_text("\n".join(rows) + "\n")


def dailydialog(name, n):
    d = RAW / "dailydialog" / name
    d.mkdir(parents=True, exist_ok=True)
    texts, emos = [], []
    for _ in range(n):
        k = rng.randrange(3, 6)
        texts.append(" __eou__ ".join(rng.choice(SAD + HAPPY + LISTEN) for _ in range(k)) + " __eou__")
        emos.append(" ".join(str(rng.randrange(7)) for _ in range(k)))
    (d / f"dialogues_{name}.txt").write_text("\n".join(texts) + "\n")
    (d / f"dialogues_emotion_{name}.txt").write_text("\n".join(emos) + "\n")


if __name__ == "__main__":
    for s, n in (("train", 40), ("valid", 12), ("test", 12)):
        ed_split(s, n)
    esconv(30)
    for s, n in (("train", 20), ("dev", 6), ("test", 6)):
        meld(s, n)
    for s, n in (("train", 20), ("validation", 6), ("test", 6)):
        dailydialog(s, n)
    print("fixtures written to", RAW)
