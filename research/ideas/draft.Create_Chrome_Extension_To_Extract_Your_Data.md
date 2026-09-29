# Chrome Extension to Export Your Own Data from Closed Platforms

## Status
- **Status**: draft
- **Complete Specs**: 15%
- **Assignee**: TBD

## Core Idea
- Many platforms (e.g., Instagram) make it hard to programmatically export
  your own data even though data-portability regulation (GDPR Art. 20, CCPA)
  entitles you to it, and their official "download your data" flows are often
  slow, incomplete, or missing fields available in the logged-in UI
- Build a Chrome extension that runs in the user's own authenticated session
  and extracts their own data (posts, likes, messages, metadata) into a
  structured, portable format (JSON/CSV): strictly self-data-export, not
  scraping other users' data or bypassing access controls
- Interesting angle: compare what the official export API gives you vs. what
  the rendered UI shows, and quantify the gap

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples
- **Instagram**: export your own posts, captions, likes-received counts, and
  comment threads into a structured archive richer than the official ZIP
  export
- **Failure mode**: platform changes its DOM/internal API and silently breaks
  the extractor: worth designing for graceful detection of breakage rather
  than silently producing empty/wrong data

## Questions
1. What data is available in the authenticated UI/internal API that is
   missing from the platform's official data-export tool, and why?
2. How do you build an extractor that fails loudly (vs. silently) when the
   platform's frontend changes?
3. What's the right legal/ethical boundary to design around (self-data-only,
   rate-limited, no bypassing of auth) so this stays squarely in "data
   portability tool," not "scraper"?

## Research Topics
- Browser extension architecture (content scripts, background workers,
  message passing) for authenticated-session data extraction
- Data-portability regulation (GDPR Art. 20) as a design constraint
- Robustness to frontend changes (detecting extractor breakage)

## Next steps
- [ ] Pick one target platform (start with Instagram) and inventory what data
  is visible in the UI vs. included in the official export
- [ ] Build a minimal content-script extractor for one data type (e.g. posts)
- [ ] Design a breakage-detection mechanism (schema/shape checks)
- [ ] Document the legal/ethical boundary explicitly before expanding scope

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References
- GDPR Article 20: Right to data portability
