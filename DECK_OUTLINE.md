# Insight Excavator — 10-Slide Deck Outline

> **For:** Microsoft Build AI hackathon (HackerEarth) · theme **AI Meets Data**.
> **Format rules:** PDF only, **max 10 slides**, ≤20 MB, filename `TeamName_Deck.pdf`.
> **Required elements (all must appear):** problem statement, solution overview, architecture diagram, AI integration details, demo screenshots, team introduction. All six are covered below.

## Design rules (judged under Communication/Presentation/UX — 15%)
- One idea per slide. Max ~3 lines of text per slide — the deck supports the talk, it isn't the talk.
- Lead with the problem and the hook, not the tech.
- Sentence case everywhere. One consistent color palette. Big readable type.
- Use the rendered pipeline diagram on slide 5 and real product screenshots on slide 7 — no stock art.
- Keep your one-liner visible: **"Most data tools guess. This one proves."**

---

### Slide 1 — Title / hook
- **Headline:** Insight Excavator
- **Sub:** "Most data tools guess. This one proves."
- **Footer:** team name · theme: AI Meets Data · Microsoft Build AI (HackerEarth)
- *Purpose:* set the frame in 5 seconds. The tagline is your whole thesis.

### Slide 2 — Problem statement *(required: problem statement)*
- **Headline:** Insight is rare, even though data is everywhere.
- **Body:** Teams sit on messy data they can't mine. Analysts spend hours cleaning before they ask a single question. Most never find the non-obvious signal.
- *Defends:* Problem Depth & Product Clarity (10%). Make the pain concrete and human.

### Slide 3 — The gap / why now
- **Headline:** And "AI for data" today just makes things up.
- **Body:** Pipe a spreadsheet into an LLM and you get a confident summary — with no proof it's true. Hallucinated analytics are worse than no analytics. That's the trust gap.
- *Purpose:* this is the wedge that makes your solution necessary. Sets up the differentiator.

### Slide 4 — Solution overview *(required: solution overview)*
- **Headline:** An agent that discovers insights — and proves every one.
- **Body:** Drop in messy data → it profiles, cleans, then runs a discovery loop where an LLM *proposes* hypotheses and statistical tests *verify* them. You only see findings that are real, each with a chart and the actual numbers.
- **Callout:** the "I had no idea that was in our data" moment.
- *Defends:* Product Clarity (10%).

### Slide 5 — How it works / architecture *(required: architecture diagram)*
- **Visual:** the pipeline diagram (screenshot the one rendered in chat, or rebuild it):
  `Messy data → Profiler (no LLM) → Cleaning agent → Insight engine [Hypothesis gen → Verifier → Significance gate, looping] → Narrative + dashboard → Verified insights`
- **One line:** deterministic stages bracket the LLM stages, so the smart part never fights bad input.
- *Defends:* System Architecture & Engineering Quality (25%). This is your single most important slide for that bucket.

### Slide 6 — AI integration deep-dive *(required: AI integration details)*
- **Headline:** The verify loop — the LLM explores, math decides.
- **Body (3 beats):**
  1. Hypothesis generator (LLM) proposes structured questions about the data.
  2. Verifier runs a real statistical test (correlation, group difference, chi-square, trend, anomaly) in code.
  3. Significance gate keeps a finding only if it's statistically real; a grounded narrator phrases the verified numbers and is forbidden from adding new claims.
- **Pull-quote:** No insight is shown unless it's statistically verified.
- *Defends:* AI Integration & Intelligence Design (25%). This is your highest-value slide overall — spend the most polish here.

### Slide 7 — Demo / it works *(required: demo screenshots)*
- **Visual:** 2–3 real screenshots — upload screen, the insight cards, one card opened showing the chart + p-value + effect size + sample size.
- **Caption:** the surprising verified finding from your demo dataset.
- *Defends:* Prototype Readiness & Scalability (15%). Show the product running, not a mockup.

### Slide 8 — Engineering & scalability
- **Headline:** Built to be trusted and to scale.
- **Bullets (short):** unit-tested analysis library · deterministic, stateless pipeline · swappable LLM provider (Azure OpenAI) · deployed public link · cached calls.
- *Defends:* the rest of Architecture (25%) + Prototype Readiness (15%).

### Slide 9 — Market & differentiation
- **Headline:** Who needs this, and why us.
- **Body:** Every data, ops, and product team that has data but not answers. Versus ChatGPT-on-a-CSV (unverified) and BI dashboards (you must already know what to ask), we *find* the insight and *prove* it.
- *Defends:* Market Understanding & Product Fit (10%).

### Slide 10 — Team & close *(required: team introduction)*
- **Headline:** The team.
- **Body:** each member, role, and one-line strength.
- **Footer:** live link (the deployed Streamlit URL) + the tagline once more.
- *Purpose:* close on the hook and hand the judges the working link.

---

## Slide → required-element checklist
- Problem statement → slides 2–3
- Solution overview → slide 4
- Architecture diagram → slide 5
- AI integration details → slide 6
- Demo screenshots → slide 7
- Team introduction → slide 10

## Slide → scoring-weight coverage
- AI Integration 25% → slide 6 (primary), 4
- Architecture & Engineering 25% → slides 5, 8
- Communication/Presentation/UX 15% → whole deck, esp. 1, 4, 7
- Prototype Readiness & Scalability 15% → slides 7, 8
- Problem Depth & Clarity 10% → slides 2, 3, 4
- Market Understanding & Fit 10% → slide 9
