---
name: recap
description: distill insights and next steps from past research reports.
---

## Goal

Review past research reports (specially most recent ones), reason about the implications and connections, and suggest next steps to improve ML models. **Include the new baseline recommendation based on recent model iterations**. Later research runs can read from it to establish their baseline.

## Output

Generate 2 output:
* Create a new folder under `recaps/local` directory for each recap. The folder should be the recap datestr. This recap output file should focus on summarizing recent research and insights and suggesting next steps.
* Create (or update) a file called `memory.md` under `recaps/local`. This file is different than the first recap file in that it should have a more comprehensive coverage of all historical research results, which will be used for next session's agents to pick up the historical context.
