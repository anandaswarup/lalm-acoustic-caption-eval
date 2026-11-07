# Annotation guidelines: model–reference agreement (E17)

These guidelines describe how each generated caption in the qualitative sample is labelled. The sample has 50 recordings and 4 captions per recording, 200 items in total (E16). One reviewer labels every item.

## 1. What is being judged

The question is **how well a generated caption agrees with the five human references** of its recording. It is not whether the caption is correct about the audio. Audio is not used here: a caption can disagree with the references and still be right about the recording, and that distinction is examined separately in E18.

Judge each caption on its own against the references. Do not compare captions A–D with each other, and do not try to work out which model wrote a caption.

## 2. Materials

| File | Use |
|---|---|
| `review_sheet.md` | Reading view: for each recording code, the five references and captions A–D |
| `annotations_reviewer_1.csv` | Where labels are entered (one row per item) |
| `categories.csv` | The six categories (also in section 5) |

Do not open `blinding_key.csv`, the metric score tables or the evaluation reports while annotating. Do not change the `item_id`, `recording_code`, `caption_label`, reference or caption columns of the template; the analysis checks that they are unchanged.

## 3. What to compare

Compare the content of the caption with the content of the references along four aspects:

1. **Sound sources**: what produces sound (a dog, a car engine, rain, people talking).
2. **Events and actions**: what happens (barking, a door closing, a vehicle passing).
3. **Acoustic attributes**: loudness, pitch, timbre, rhythm, distance, background or foreground, the acoustic environment (indoors, reverberant, outdoors).
4. **Temporal structure**: order, overlap, onset and ending, change over time ("then", "while", "gradually louder").

Ignore style:

- framing such as "The audio features…", "In this recording…", "A sound of…";
- length, sentence structure and wording, as long as the meaning is the same;
- hedges ("possibly", "appears to be"): treat a hedged claim as a claim;
- synonyms and near-synonyms ("automobile" and "car"; "people chatting" and "a crowd talking").

A more general or a more specific term for the same thing counts as agreement ("a vehicle" and "a truck"; "birds" and "a crow"), unless the specific term contradicts the references ("a crow" when the references say "seagulls").

## 4. Definitions

- **Principal content**: a sound source or event that **at least two of the five references** mention, in any wording. References often disagree in detail; the principal content is what they share.
- **Covered**: a caption element supported by at least one reference. A detail that only one reference mentions counts as covered, not as an addition.
- **Additional detail**: a caption element that **no reference supports**: a source, event, attribute or temporal claim the references do not mention. An addition that contradicts the references is a mismatch, not an addition.
- **Missing detail**: principal content that the caption does not mention.
- **Mismatch**: the caption identifies a principal source or event as something different ("rain" where the references describe frying), or describes a different scene altogether.

## 5. Categories

Assign exactly one primary category per caption.

| Code | Category | Use when |
|---|---|---|
| 1 | Strong agreement | The caption covers the principal content, adds nothing substantive and omits nothing important. Differences are only in wording. |
| 2 | Mostly aligned | The main content agrees, with minor differences in emphasis or detail: for example, it omits a minor element or adds a trivial one, without changing the description. |
| 3 | Mostly aligned with additional details | The main content agrees, but the caption adds one or more substantive details (sources, events, attributes) that no reference supports. |
| 4 | Partially aligned with missing details | The caption captures some principal content but omits other principal content, and does not add much. |
| 5 | Partially aligned with additional details | The caption captures some principal content but also introduces sources, events or descriptions not in the references, and these change the overall description. |
| 6 | Weak agreement or sound-source/event mismatch | The caption shares little with the references, or misidentifies a principal source or event, or appears to describe a different recording. |

## 6. Decision procedure

1. List the principal content (section 4) from the references.
2. Decide the agreement level:
   - **All** principal content covered → step 3.
   - **Some** principal content covered → step 4.
   - **None** covered, or a principal source or event contradicted → **6**.
3. All principal content covered:
   - no substantive additional detail → **1** if the descriptions are essentially equivalent, **2** if they differ in minor ways;
   - substantive additional detail → **3**.
4. Some principal content covered (missing details):
   - substantive additional detail as well → **5**, or **4** if the omissions clearly dominate (see tie-break);
   - no substantive additional detail → **4**.

**Substantive** means that a listener reading the caption would imagine a different or richer scene: an extra sound source, an extra event, or a specific attribute such as "metallic", "distant" or "echoing" that no reference mentions. Rewording, generic intensifiers ("clearly audible") and framing are not substantive.

**Tie-breaks**

- Between 1 and 2: choose 2 if any minor element differs. Use 1 only for captions that a reader could swap with the references without noticing a difference in content.
- Between 2 and 3: a single minor attribute (for example "loud") is 2; a new source or event, or several new attributes, is 3.
- Between 4 and 5 when a caption both omits and adds: choose by which changes the description more. If they are about equal, choose 5, and record the omission in the notes.
- Between 5 and 6: if at least one principal source or event is correctly covered and none is contradicted, choose 5; otherwise 6.
- When the references themselves disagree strongly and there is no content that two of them share, judge against the content most references share, and say so in `notes_other`.

## 7. Columns to fill in

| Column | Content |
|---|---|
| `category` | One code, 1–6 |
| `disputed_details` | The caption's additional or contradicting phrases, as short quotes from the caption, separated by `;` (for example `metallic clanking; distant thunder`). Leave empty for categories 1 and 4 unless something is disputed. These are checked against the audio in E18. |
| `notes_sources` | Agreement or disagreement about sound sources |
| `notes_attributes` | Agreement or disagreement about acoustic attributes |
| `notes_temporal` | Agreement or disagreement about order, overlap or change over time |
| `notes_other` | Anything else: uncertain decisions, conflicting references, ambiguous wording |

Notes are short and optional, except that an uncertain category decision should be explained in `notes_other`.

## 8. Worked examples

These examples are invented for illustration; none is taken from the sample.

References (invented):

1. A dog barks repeatedly while cars drive past on a wet road.
2. Traffic passes by as a dog barks loudly nearby.
3. Cars are driving on a rainy street and a dog is barking.
4. A dog keeps barking and vehicles go by.
5. Several vehicles pass while a small dog barks.

Principal content: a barking dog; passing vehicles. Covered by at least one reference: wet or rainy road, loud barking, a small dog.

| Caption (invented) | Category | Why |
|---|---|---|
| A dog barks as cars pass by on a wet street. | 1 | Both principal elements, plus one covered detail; nothing added or missing. |
| A dog barks while traffic moves in the distance. | 2 | Both principal elements; "in the distance" is a minor attribute not in the references. |
| A dog barks while cars pass, and a man shouts and a car horn honks twice. | 3 | Both principal elements; adds two sources and events (`disputed_details`: `a man shouts; a car horn honks twice`). |
| A dog barks repeatedly. | 4 | Omits the vehicles. |
| A dog barks near a construction site with drilling and hammering. | 5 | Covers the dog, omits the vehicles, adds a different scene (`disputed_details`: `construction site; drilling and hammering`). |
| Heavy rain falls on a metal roof. | 6 | Neither principal element; describes a different scene. |

## 9. Recording the work

- Work through the recordings in code order. Label all four captions of a recording before moving on.
- Save the template as CSV in UTF-8. Do not add, remove or reorder columns.
- When every item has a category, the labels are final: the analysis (`make annotations`) checks the template, unblinds it with the key and summarises the categories per model. Labels are not changed after unblinding.
