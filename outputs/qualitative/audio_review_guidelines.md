# Audio review guidelines: captions against the recordings (E19)

These guidelines describe the second pass over the qualitative sample: the same 50 recordings and 200 captions as in E17–E18, this time judged against the audio. One reviewer reviews every item.

## 1. What is being judged

The question is **whether a generated caption is correct about its recording**. The references are not used here. E18 asked how well a caption agrees with the references; a caption can disagree with them and still describe the recording correctly, or agree with them and still be wrong. The two passes are reported separately.

Judge each caption on its own against the audio. Do not compare captions A–D with each other, do not try to remember the references or your E18 category for an item, and do not try to work out which model wrote a caption.

## 2. Materials

| File | Use |
|---|---|
| `audio_review.html` | Listening page: for each recording code, an audio player and captions A–D with their disputed phrases. Open it in a browser; it plays the copies in `audio/`. Local only, never committed. |
| `audio_review_sheet.md` | The same text without audio |
| `audio_annotations_reviewer_1.csv` | Where caption ratings are entered (one row per caption, 200 rows) |
| `disputed_verification_reviewer_1.csv` | Where disputed phrases are checked (one row per phrase from E18 `disputed_details`) |

Do not open `review_items.csv`, `review_sheet.md`, `annotations_reviewer_1.csv`, `blinding_key.csv`, `qualitative_annotations.csv`, the metric score tables or the evaluation reports while reviewing. Do not change the `item_id`, `recording_code`, `caption_label`, `caption`, `phrase_id` or `phrase` columns; the analysis checks that they are unchanged.

## 3. Listening

- Listen to each recording **in full** before rating its captions, with headphones, at a comfortable fixed volume. Replay as often as needed.
- Judge only what can be heard. A caption element is audible if a careful listener would recognise it in the recording; it does not need to be loud or in the foreground.
- As in E18, ignore style: framing ("The audio features…"), length, wording, and hedges (a hedged claim counts as a claim). A more general term is correct if the specific sound is consistent with it ("a vehicle" for a truck).
- The **principal** sound source and event are the ones a listener would name first to describe the recording: the dominant or most salient sound, or the sound the recording is evidently about. **Secondary** sounds are the rest (background, brief or incidental sounds).

## 4. Overall rating

Assign exactly one rating per caption.

| Code | Rating | Use when |
|---|---|---|
| 1 | Accurate | Every sound source and event in the caption is audible, and the described attributes and order match the recording. Omitting secondary sounds is acceptable. |
| 2 | Minor errors | The principal sound source and event are correct, but the caption has a wrong attribute or order, an unsupported secondary detail, or omits a secondary sound. |
| 3 | Major error or wrong sound source | The principal sound source or event is wrong, a principal source is invented, or the dominant sound is omitted. |

**Tie-breaks**

- Between 1 and 2: an omitted secondary sound alone does not prevent 1. Any wrong attribute, wrong order or unsupported secondary detail makes it 2.
- Between 2 and 3: if the principal source and event are right, choose 2 however many secondary errors there are, and describe them in the notes. If the principal source or event is wrong, or the caption misses the dominant sound, choose 3, even if the secondary details are right.
- A principal source the caption names wrongly but plausibly (for example "rain" for frying) is wrong: 3. If the sound is genuinely ambiguous between the two, choose 2 and explain in `notes_sources`.

## 5. Disputed phrases

Each row of `disputed_verification_reviewer_1.csv` is one phrase that E18 marked as unsupported by every reference. Check it against the recording and enter one verdict.

| Code | Verdict | Use when |
|---|---|---|
| S | Supported | The recording contains what the phrase describes. |
| U | Unsupported | The recording does not contain what the phrase describes, or contradicts it. |
| A | Ambiguous or unverifiable | Listening can neither confirm nor rule it out (masked, too faint, or not decidable from sound alone). |

- Judge the phrase in the context of its caption (a "distant" sound is the caption's sound being distant).
- Use A for claims that sound cannot settle: an unseen object's material, a speaker's intention, a location ("in a kitchen") that no sound confirms or rules out. Use U when the sound is clearly absent or clearly different.
- The phrase verdicts and the overall rating are separate judgements: rate the caption as a whole by section 4 even if the phrase rows are already filled in.

## 6. Columns to fill in

`audio_annotations_reviewer_1.csv`:

| Column | Content |
|---|---|
| `audio_rating` | One code, 1–3 |
| `notes_sources` | What is right or wrong about the sound sources |
| `notes_attributes` | What is right or wrong about acoustic attributes |
| `notes_temporal` | What is right or wrong about order, overlap or change over time |
| `notes_other` | Anything else: uncertain decisions, ambiguous sounds, audible content the caption omits |

`disputed_verification_reviewer_1.csv`:

| Column | Content |
|---|---|
| `verdict` | One code: S, U or A |
| `note` | Optional; explain A verdicts and any uncertain decision |

Notes are short and optional, except that an uncertain rating or an A verdict should be explained.

## 7. Worked examples

These examples are invented for illustration.

Recording (invented): a small dog barks repeatedly in the foreground; cars pass on a wet road throughout; a car horn sounds once near the end.

| Caption (invented) | Rating | Why |
|---|---|---|
| A dog barks while cars drive past on a wet road. | 1 | Every element is audible; the horn is a secondary sound and may be omitted. |
| A large dog barks while cars pass, then a horn honks twice. | 2 | Principal content right; "large" and "twice" are wrong attributes. |
| A dog barks while traffic passes and a man shouts. | 2 | Principal content right; the shout is an unsupported secondary detail (phrase `a man shouts`: U). |
| Cars pass on a wet road and a horn honks. | 3 | Misses the dominant sound, the barking dog. |
| Heavy rain falls on a metal roof. | 3 | Wrong principal source and event. |

## 8. Recording the work

- Work through the recordings in code order. For each recording, listen, rate all four captions, then check their disputed phrases, before moving on.
- Save both templates as CSV in UTF-8. Do not add, remove or reorder rows or columns.
- When every caption has a rating and every phrase a verdict, the labels are final: the analysis (`make audio-review`) checks both templates, unblinds them with the key and summarises them per model and against the E18 categories. Labels are not changed after unblinding.
