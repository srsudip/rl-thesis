# assets/ — Static Frontend Assets

Dash automatically serves everything in this directory at `/assets/`.

## Files

| File | Purpose |
|------|---------|
| `style.css` | Global CSS overrides: typography, card styling, KNEC grade-level colour classes, HITL badge styles, pathway colour tokens |

## CSS Conventions

Grade-level colour classes follow the KNEC 8-level scheme:

| Class | Level | Colour |
|-------|-------|--------|
| `.grade-ee1`, `.grade-ee2` | Exceeding Expectations | Green shades |
| `.grade-me1`, `.grade-me2` | Meeting Expectations | Blue shades |
| `.grade-ae1`, `.grade-ae2` | Approaching Expectations | Amber shades |
| `.grade-be1`, `.grade-be2` | Below Expectations | Red shades |

Pathway colour tokens:

| Token | Pathway |
|-------|---------|
| `--stem-color` | STEM (#1a73e8) |
| `--ss-color` | Social Sciences (#34a853) |
| `--arts-color` | Arts & Sports Science (#ea4335) |

## Adding Assets

Drop any `.css`, `.js`, or image file here and Dash will serve it automatically. No configuration required.
