# Stage 2 — AAAI 2026 AuthorKit Compliance Notes

This note captures the concrete constraints we enforced during the Dec 12, 2025 session, referencing the AAAI 2026 AuthorKit LaTeX guidance.

---

## Sources consulted

- `AuthorKit26/CameraReady/LaTeX/Formatting-Instructions-LaTeX-2026.tex`
- `AuthorKit26/AnonymousSubmission/LaTeX/anonymous-submission-latex-2026.tex`
- `AuthorKit26/CameraReady/LaTeX/aaai2026.sty`

---

## Key constraints that influenced edits

### 1) No margin overflow

- **Rule**: “Nothing is permitted to intrude into the margin or gutter” and you must fix overfull boxes.
- **Enforcement**: We iterated until `paper/sokrates.log` reported **no `Overfull \hbox`**.

### 2) No negative spacing near floats/captions/sections

- **Rule**: Negative `\vspace` / `\vskip` (and similar) is disallowed in proximity to figures/tables/captions/headings.
- **Enforcement**: Removed negative spacing hacks around tables and used column width + ragged-right + `\arraystretch` instead.

### 3) Table caption placement

- **Rule**: AuthorKit states table captions must appear under tables (see “Table Captions” section).
- **Enforcement**: Moved `\caption{...}` to appear after the `tabular` block in all paper tables.

### 4) Table font size constraints

- **Rule**: Tables should be 10pt roman; 9pt is allowed if necessary. Avoid shrinking the entire table with `\resizebox`.
- **Enforcement**: Avoided `\scriptsize` tables and relied on column formatting instead.

### 5) Disallowed packages / commands

- **Rule**: `hyperref`, `geometry`, `titlesec`, etc. are forbidden; many layout-altering commands are forbidden.
- **Enforcement**: We did not add any disallowed packages; used only `array` for table columns.

---

## Submission workflow note: PDF-only vs source archive

- **Anonymous submission**: the AuthorKit anonymous-submission template indicates the submission stage may require **PDF-only**.
- **Camera-ready**: the camera-ready instructions explicitly describe submitting a **source archive** in addition to the PDF, and include the “single `.tex` file, no `\input`” rule.

To support the camera-ready workflow without refactoring the paper structure, we added:

- `paper/tools/inline_tex.py` to generate `paper/sokrates_single.tex` (no `\input{...}`).


