"""
Generates SAMPLE (synthetic) datasets for the two modules whose real datasets
were not supplied: stroke and mental_health.

  data/raw/stroke/stroke_raw.csv                 same schema as Kaggle
                                                 "Stroke Prediction Dataset"
  data/raw/mental_health/mental_health_raw.csv   columns: text, label

!!! These are SYNTHETIC. Use them only to test the pipeline end to end.
!!! For your final report / viva, replace them with the real Kaggle files
!!! (same file names, same columns) and re-run the training pipeline.

Run from the project root:   python scripts/generate_sample_data.py
"""
import os
import random

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
random.seed(42)


# ------------------------------------------------------------------ stroke
def make_stroke(n=5110):
    gender = rng.choice(["Male", "Female"], n, p=[0.41, 0.59])
    age = np.clip(rng.normal(46, 22, n), 0.5, 82).round(0)
    hypertension = (rng.random(n) < (0.03 + 0.20 * (age / 82) ** 2)).astype(int)
    heart_disease = (rng.random(n) < (0.01 + 0.14 * (age / 82) ** 2)).astype(int)
    ever_married = np.where((age > 22) & (rng.random(n) < 0.9), "Yes", "No")
    work_type = np.where(
        age < 14, "children",
        rng.choice(["Private", "Self-employed", "Govt_job", "Never_worked"], n, p=[0.64, 0.18, 0.15, 0.03]),
    )
    residence = rng.choice(["Urban", "Rural"], n)
    glucose = np.clip(rng.normal(95, 25, n) + 60 * (rng.random(n) < 0.15) * rng.random(n) * 2 + 0.2 * age, 55, 272).round(2)
    bmi = np.clip(rng.normal(28, 7, n) + 0.05 * (age - 40), 12, 70).round(1)
    smoking = np.where(
        age < 14, "Unknown",
        rng.choice(["never smoked", "formerly smoked", "smokes", "Unknown"], n, p=[0.37, 0.17, 0.15, 0.31]),
    )

    logit = (
        -8.2 + 0.075 * age + 0.9 * hypertension + 1.0 * heart_disease
        + 0.009 * (glucose - 100) + 0.3 * (smoking == "smokes") + 0.15 * (smoking == "formerly smoked")
    )
    p = 1 / (1 + np.exp(-logit))
    stroke = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "id": rng.integers(1000, 72000, n), "gender": gender, "age": age,
        "hypertension": hypertension, "heart_disease": heart_disease,
        "ever_married": ever_married, "work_type": work_type, "Residence_type": residence,
        "avg_glucose_level": glucose, "bmi": bmi, "smoking_status": smoking, "stroke": stroke,
    })
    # real Kaggle file has ~3.9% missing BMI
    df.loc[rng.random(n) < 0.039, "bmi"] = np.nan
    return df


# ------------------------------------------------------------ mental health
OPENERS = {
    "Normal": ["Had a pretty good day", "Just finished a great workout", "Feeling grateful today",
               "Spent the weekend with friends", "Started a new book", "Work went smoothly",
               "Cooked dinner for my family", "Went for a long walk", "Excited about the trip next month"],
    "Depression": ["I feel empty all the time", "Nothing makes me happy anymore", "I can't get out of bed",
                   "I feel worthless and hopeless", "I have lost interest in everything",
                   "I cry for no reason every night", "I feel so alone and nobody cares",
                   "Everything feels pointless", "I am exhausted and sad all day"],
    "Anxiety": ["My heart is racing and I can't calm down", "I keep worrying about everything",
                "I had another panic attack today", "I am constantly afraid something bad will happen",
                "I can't stop overthinking every little thing", "My chest feels tight and I feel nervous",
                "I am scared to go outside", "I feel on edge all the time", "I worry so much I can't breathe"],
    "Stress": ["I am overwhelmed with deadlines", "Too much pressure at work and I can't cope",
               "I have so many things to do and no time", "My boss keeps piling on work",
               "I am burned out from exams", "Money problems are stressing me out",
               "I am under so much pressure lately", "I am juggling too many responsibilities",
               "There is no time to rest and I am drained"],
}
SHARED = ["lately", "honestly", "this week", "right now", "for a while", "again today", "these days", "tonight", ""]
TAILS = {
    "Normal": ["Life is good.", "Looking forward to tomorrow.", "Feeling calm and happy.", "Can't complain.", "Things are fine."],
    "Depression": ["I don't see the point.", "I just want to sleep forever.", "Nobody would notice.", "I can't sleep either.", "I feel numb."],
    "Anxiety": ["I can't sleep because of it.", "What if something goes wrong?", "I feel sick with worry.", "My hands keep shaking.", "I just want it to stop."],
    "Stress": ["I need a break.", "I can't keep up.", "My head hurts from it all.", "I'm running on coffee.", "I need a vacation."],
}
# a little label noise through cross-class "bleed" sentences
BLEED = ["I feel tired.", "I couldn't sleep well.", "It is what it is.", "Not sure what to do.", "Talked to a friend about it."]


def make_text(n_per_class=700):
    rows = []
    for label in OPENERS:
        for _ in range(n_per_class):
            parts = [random.choice(OPENERS[label]) + " " + random.choice(SHARED)]
            parts.append(random.choice(TAILS[label]) if random.random() < 0.8 else random.choice(BLEED))
            if random.random() < 0.3:
                parts.append(random.choice(BLEED))
            text = " ".join(p.strip() for p in parts).replace("  ", " ")
            # ~4% label noise to keep metrics realistic
            lab = label if random.random() > 0.04 else random.choice(list(OPENERS))
            rows.append((text, lab))
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])


if __name__ == "__main__":
    os.makedirs("data/raw/stroke", exist_ok=True)
    os.makedirs("data/raw/mental_health", exist_ok=True)
    s = make_stroke()
    s.to_csv("data/raw/stroke/stroke_raw.csv", index=False)
    m = make_text()
    m.to_csv("data/raw/mental_health/mental_health_raw.csv", index=False)
    print("stroke:", s.shape, "stroke rate:", round(s.stroke.mean() * 100, 2), "%")
    print("mental_health:", m.shape, m.label.value_counts().to_dict())
