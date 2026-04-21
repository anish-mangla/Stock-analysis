---
inclusion: manual
---

# Event Labeling Workflow

When the user says "let's do the next batch" or "classify this" or pastes ChatGPT output
for event labeling, follow the workflow in `___LABELING_WORKFLOW___.md`:

1. Read `___LABELING_WORKFLOW___.md` to find the current batch number and status
2. If user pasted ChatGPT output:
   - Save raw response to `outputs/chatgpt_research/batch_NNN_raw.txt`
   - Classify each event using our 11-type taxonomy
   - Save to `outputs/stock_level_labels_v2/batch_NNN_classified.csv`
   - Update `___LABELING_WORKFLOW___.md`: increment completed count, update current batch, generate next prompt
3. If user says "next batch" without pasting:
   - Read the "Next Prompt" section from the workflow file
   - Show it to the user to copy into ChatGPT

The CSV schema and event types are defined in the workflow file.
Always preserve the full ChatGPT response text in the store_description column.
