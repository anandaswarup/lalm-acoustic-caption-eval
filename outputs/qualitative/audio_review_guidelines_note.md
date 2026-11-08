# Note on the audio-review guidelines (2026-11-08)

`audio_review_guidelines.md` is kept unchanged, because the completed ratings were made with it and `outputs/reports/audio_review.json` records its SHA-256.

Its rating table lists "omits a secondary sound" under rating 2, while rating 1 ("Omitting secondary sounds is acceptable") and the first tie-break ("an omitted secondary sound alone does not prevent 1") allow it. The tie-break governed the completed ratings. The effective rule is:

- 1 Accurate: every sound source and event in the caption is audible, and attributes and order match. An omitted secondary sound alone does not lower the rating.
- 2 Minor errors: the principal source and event are correct, but the caption has a wrong attribute or order, or an unsupported secondary detail.
- 3 Major error or wrong sound source: the principal source or event is wrong or invented, or the dominant sound is omitted.
