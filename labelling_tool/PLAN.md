# Labelling tool: the agreed plan

Agreed 2026-11-11. Nothing is built yet. This file is the record of what was decided and why.

## Goal
Every piece gets a human-checked `text` (native script) and `text_roman` (romanized). The tool shows a draft made by
the model; the reviewer listens, corrects and marks the piece verified. Verified pieces are the new training data.
**The reviewer decides when to stop and fine-tune**; after a fine-tune, the new model redrafts the pieces that are still
unverified, and review goes on.

## Order of work
1. **v3 drafts everything on Kaggle**: Bangalore (done), Kolkata (running), Maharashtra, then **Hyderabad**: its
   segments are uploaded by hand, cut by the same pause-only rule, and go through the same notebook.
   Progress: `kaggle_runs/TRANSCRIPTION_LOG.md`.
2. **Build the tool on the laptop** (it can start with the pieces already transcribed).
3. **Review**: the verified folder fills by itself; fine-tune (v4) whenever the reviewer decides; the new model
   redrafts the unverified pieces; repeat.

## What goes into the tool: ONE folder, `labelling_data/`
Everything the tool needs is in `labelling_data/`, old and new together. It is built and updated by
`labelling_tool/build_data.py`, and its `README.md` holds the current counts.

**The audio is hard-linked, not copied.**
- The same file shows up in `labelling_data/audio/` and in its old place, so it uses no extra disk space. Only
  17 GB is free, and the pieces alone are about 21 GB.
- The old folders and dashboards keep working.
- Deleting one name leaves the other.

**What the folder holds, as of the first build (2026-11-11):**

| Source | Calls | Pieces | Text to check |
|---|---|---|---|
| CRM Bangalore | 3,155 (125.5 h) | 18,374 | v3 drafts for 17,905; 47 calls the CRM refused on Kaggle have none yet |
| CRM Kolkata | 1,695 (79.1 h) | 11,533 | v3 drafts for 5,570 so far (Kaggle running) |
| CRM Maharashtra | 3,752 (156.7 h) | 22,888 | none yet (Kaggle next) |
| CRM Hyderabad | 2,825 (108.3 h) | 15,900 | none yet (upload to Kaggle by hand) |
| Old calls | 2,546 (313.3 h) | 6,979 clips (45.7 h) | the old labels, plus whole-call text for every old call |

- CRM calls shorter than 30 s are left out, the same rule as the Kaggle run.
- 3 old clips are left out because their audio was deleted when they were dropped from training.

**Files in the folder:**
- `calls.csv`: one row per call.
- `pieces.csv`: one row per piece.
- `drafts.jsonl`: the text to check for each piece.
- `call_texts.jsonl`: whole-call texts.
- `audio/`, `incoming/` (new Kaggle zips go here), `verified/` (written only by the tool).

**The V4 folder is part of the old calls** (the earlier open question).
- Its 1,811 calls are the old Karnataka calls that were never cut into clips: 1,752 are the very same files as in
  `data/raw/audio` (same md5), and 59 exist only in the V4 folder.
- They are **already transcribed**, but as one text per whole call, with no times. Kept: the old `text_native` /
  `text_romanized`, plus the V4 folder's Agent/Customer-split and English texts.
- **Most of their 267.7 h has little text.** Measured against the ~150 words a minute of normal calls:
  - **434 calls (226.1 h)** have under 50 words a minute; the longest call is 5.9 h. Their text covers only a small
    part of the audio, so it can't serve as a label. They're flagged in `calls.csv`.
  - **1,377 calls (41.5 h)** look complete: English 916, Hindi 409, Kannada 52.
- **To use the 1,377 complete ones for training,** cut them by the same pause-only rule, then **split their existing
  text at the cut points** by aligning words to the audio. v3 drafts aren't needed, since that would throw the existing
  text away.
- **Low priority:** they're English and Hindi, which we already have the most of. Decided 2026-11-11: not now.
- 371 of them have "x1.25" in the file name. Check whether they are sped up before using them.

The audit's repair list (`Downloads/crm_asr_v3_repairs.csv`) is no longer in Downloads, so the old clips it marked as bad
can't be flagged yet. The audit's numbers were 102 to drop and 31 to relabel.

## Features
1. **Pick anything, skip anything.** Filter by state, language, call, source and status, and jump to any call or piece.
   - For an old call, its whole-call text is shown beside the pieces as a reference.
2. **Call view.**
   - The full call's waveform with the piece edges marked, and the list of pieces.
   - Play the whole call or one piece.
3. **Player.**
   - Drag to seek.
   - Speed from 0.5× to 2× with the pitch kept.
   - Loop a piece, step back 2 s, volume boost.
   - Keyboard shortcuts: Space play/pause, ←/→ seek, `[` `]` speed, Enter save and next.
4. **Two text boxes side by side**, native and romanized.
   - Editing one **highlights the other box's border** ("needs update") until it is edited or confirmed.
   - A suggested transliteration fills the other script.
5. **Languages of a piece.**
   - One **main language**: the model needs one language as its prompt.
   - **"Also contains"** tags for every other language heard, even a single word.
6. **Only two states: Unverified and Verified.** A skipped or unclear piece stays unverified, so it never enters the
   dataset.
7. **History.** The model's draft is never overwritten. Every save stores:
   - the old text and the new text;
   - the word-level changes (which word became what);
   - who saved it and when;
   - which model made the draft.

   Every save can be undone.
8. **Edge fix** (rarely needed, since cuts are made on silence): move a piece's start or end, split, or merge. The
   export re-cuts the audio from the full call.
9. **A one-page spelling guide, shown in the tool.** It starts from what the old 45.7 h already does, so old and new
   labels match:
   - **English words inside Indian-language speech are written in that language's script.** Only 0.3–3.9% of
     `text_native` words are in Latin letters (Marathi 0.3%, Hindi 2.9%, Bengali 3.9%).
   - **Numbers as digits.** 35–55% of clips contain digits.
   - Still to decide: toh/to, nahi/nahin, हां/हाँ, BHK and brand names, number words versus digits when spoken as
     words.
10. **Progress.**
    - Verified hours per language and state.
    - The model's error rate per language, measured from the corrections, to show where it is weakest.
11. **The verified folder updates itself on every save.**
    - Each verified piece goes in at once: its audio, both texts and its languages.
    - A call shows as **complete** when all its pieces are verified.
    - The format is the old `dataset.jsonl` one, so training code needs no change.
12. **Train/test split at the end, by lead.** All calls of one lead go on one side, so the test part is fully hidden
    from training.
    - **Check before splitting** whether any old-dataset call is also among the CRM calls: the old files are `.m4a`
      named like `bn_wb_0003`, so a match needs an audio check, not a file name.
    - If a call is in both, its old clips and new pieces go on the same side.
13. **New model drafts never touch verified pieces.** When v4 (or later) redrafts, only unverified pieces get the new
    text; verified pieces stay as they are.

## Screen design
The goal is a calm, quiet screen for hours of listening and typing. The text and the audio are the focus; everything
else stays out of the way.

### Layout (one screen, no page changes)
```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ TOP BAR  Labelling   [State ▾] [Language ▾] [Source ▾] [Status ▾]  🔍 call id   ▓▓▓░ 12.4 h verified │
├───────────────┬──────────────────────────────────────────────────────────────────────────┤
│ CALL LIST     │ CALL HEADER  187080_4808749 · Bangalore · kn · 7:02 · 16 pieces · 3 verified │
│ (left, 280px) │ ──────────────────────────────────────────────────────────────────────── │
│ ● 187080_48…  │ WAVEFORM of the whole call, piece edges as thin lines, verified pieces     │
│   7:02 · kn   │ tinted green, current piece highlighted; click anywhere to play from there │
│   3/16 ✓      │ ▶ 01:24 / 07:02   ⟲2s   0.75× ━━●━━ 2×   🔊 boost   🔁 loop piece        │
│ ○ 187080_48…  │ ──────────────────────────────────────────────────────────────────────── │
│   2:31 · hi   │ PIECES (scrolling list, the current one open, the others one line each)    │
│   0/6         │  ┌ s003 · 00:48–01:12 · 24 s · kn ───────────────────── Unverified ┐ │
│ …             │  │ Native   [ editable text, large                                   ] │ │
│               │  │ Roman    [ editable text, large                                   ] │ │
│               │  │ Languages: kn (main)  + en   [+ add]                              │ │
│               │  │ v3 draft ▸ (fold-out: original text and word changes)            │ │
│               │  │                                   [Skip]   [✓ Verify  Enter]     │ │
│               │  └──────────────────────────────────────────────────────────────────┘ │
│               │  s004 · 01:12–01:40 · "namaskara sir…" (first words)    ✓ Verified     │
│               │                                                                          │
│               │ REFERENCE (old calls only, folded): whole-call text, Agent/Customer split │
└───────────────┴──────────────────────────────────────────────────────────────────────────┘
```
- **Call list (left):** each call shows its id, length, main language and "verified / pieces".
  - A dot marks status: empty = nothing verified, half = some, full green = all.
  - The list follows the filters in the top bar.
- **Call area (right):**
  - header, then the waveform, then the player bar, then the pieces;
  - **only the current piece is open for editing**; the others are one line each, so the call reads like a list.
- **The spelling guide** opens as a side drawer from the top bar (`?` key) and never covers the text boxes.
- **Phone width:** the call list becomes a drawer, and everything else stacks in one column.

### Colours
Light theme by default and a dark theme for night work, both with the same roles. Only one accent colour, so the
status colours stand out.

| Role | Light | Dark | Used for |
|---|---|---|---|
| Background | `#F7F7F5` (warm off-white) | `#16181B` | page |
| Surface | `#FFFFFF` | `#1F2226` | panels, the open piece |
| Border | `#E4E4E0` | `#2E3238` | lines between parts |
| Text | `#1D1F21` | `#E8E9EA` | main text |
| Muted text | `#6B6F76` | `#9AA0A8` | times, ids, labels |
| Accent | `#2F6FEB` (calm blue) | `#5B8DEF` | play button, current piece, focus ring, links |
| Verified | `#1F9D55` (green) | `#3CC27A` | ✓ badge, verified pieces on the waveform, progress bar |
| Unverified | `#8A8F98` (grey) | `#7D838C` | badge of pieces not yet verified |
| Needs update | `#E8A317` (amber) | `#F2B43C` | **border of the other script's box** after one box is edited |
| Changed word | `#FFF3C4` background (light yellow) | `#4A3F14` | words changed against the draft, in the fold-out |
| Danger | `#D64545` | `#E86464` | undo of a verification, delete |
| Waveform | `#B8C4D6`, played part `#2F6FEB` | `#3A4A63`, played part `#5B8DEF` | the call's waveform |

- **Language tags** have muted, distinct colours, the same everywhere (list, header, piece):
  - hi `#7C5CFF`, en `#2F6FEB`, kn `#C2410C`, te `#0E9F6E`, mr `#B45309`, bn `#BE185D`;
  - always shown as a small rounded tag with its code, so colour is never the only signal.
- **Contrast:** text and buttons meet WCAG AA (4.5:1) in both themes.

### Type and spacing
- **Fonts:**
  - **Inter** for the interface;
  - **Noto Sans** + the Noto family for the scripts (Devanagari, Kannada, Telugu, Bengali), so every script renders
    cleanly and at the same height;
  - numbers and times in tabular figures.
- **Sizes:**
  - **18 px** for the text being edited, with line height 1.6;
  - 14 px for the interface;
  - 12 px for ids and times.
- **Spacing:** an 8 px grid; 16 px padding in panels; 12 px between pieces.
- **Shape:** corners 8 px for panels, 6 px for buttons and tags. Shadows only on the open piece, and very light
  (`0 1px 3px rgba(0,0,0,.08)`).
- **Motion:** none beyond short fades (120 ms). Nothing moves while audio plays, except the play line on the waveform.

### Behaviour that keeps it fast
- **Keyboard first**, with every action shown on hover:
  - Space play/pause; ←/→ seek 2 s; `[` `]` speed;
  - `L` loop the piece; `Enter` verify and open the next piece; `Ctrl+Enter` verify and stay;
  - `S` skip; `Alt+↑/↓` previous/next piece; `?` spelling guide;
  - `Ctrl+Z` undo.
- **Typing doesn't break the audio:** keyboard shortcuts use Alt or work only outside the text boxes, except Space,
  which is taken over only when a text box isn't focused.
- **Opening a piece** plays it automatically at the chosen speed. Speed and volume boost are remembered.
- **Autosave** two seconds after typing stops. A small "Saved" note appears next to the badge; nothing pops up.
- **Verifying** turns the badge green and tints that piece green on the waveform. The next unverified piece opens.
- **Undo verification** asks once ("Mark as unverified?"), as it's the only action that removes work.
- **Empty states say what to do**, for example "No calls match these filters", with a button to clear them.
- **Progress** in the top bar:
  - verified hours out of the total for the current filters;
  - a click opens the per-language table: verified hours, and the model's error rate measured from the corrections.

### What it deliberately doesn't have
- No pop-up dialogs while working.
- No third status.
- No speaker tags.
- No colour that means two things.

## Storage and safety
- **Storage:** a SQLite database on the laptop, backed up every day. It is text only, a few MB, so a copy can also go
  to a private cloud folder without using much data.
- **The audio stays on the laptop.** The laptop is now the **only copy** of some calls: the CRM has already dropped
  some, for example 26 Bangalore calls that are no longer on their lead's timeline. Keep one copy on an external
  drive.
- **More reviewers later, through ngrok:**
  - **A login per person is required**, because an ngrok link can be opened by anyone on the internet and these are
    customer recordings.
  - The reviewer's name is stored with every save.
- **Audio over ngrok uses the laptop's mobile data.**
  - The FLAC pieces are **44 MB per hour of audio**.
  - Sent as Opus at about 24 kbps (no audible loss on phone audio) and cached by the browser, they are about
    **11 MB per hour**, 4× less.

## Not now (decided)
- Agent and customer speaker tags.
- Typing test labels without seeing the draft: labels start from the model draft.
- Masking names or phone numbers in labels: to revisit before any data is shared outside.
