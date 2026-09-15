# Evaluation

The evaluation has two parts. The **verdicts** decide whether a final answer
is correct. The **diagnostics** explain a run: how much of the candidate set it
read, and whether its quotations come from data it read. No language model
takes part in any of them.

| Tool | Question | Needs the graph |
|---|---|---|
| `score.py` + `canonical.py` | Is the answer exactly correct? | No |
| `grading.py` (through `score.py`) | Which fields are correct, and which entities are missing? | No |
| `coverage.py` | Did the run read the complete candidate set? | Yes |
| `grounding.py` | Did the run read the text that it quotes? | Yes |
| `audit_gold.py` | Does the graph confirm the reference answers? | Yes |
| `summary.py` | What are the totals of a batch? | No |

The harness also enforces provenance during the run: an identifier in `FINAL`
must come from a read or from the question (see
[02 Architecture](02-architecture.md#provenance-and-shape-checks)).

## Exact verdict

`score.py` parses the final answer as JSON and compares it with
`answer.expected` after canonicalization (`canonical.py`):

- Every list is a set. The order of items does not change the verdict.
- A sequence whose order has meaning, such as a path, is written as
  `{"ordered": [...]}`. Its order is kept.
- QIDs and PIDs are normalized to upper case.
- Strings are trimmed, and repeated white space becomes one space.
- Typographic punctuation is folded: curly quotes to straight quotes, dashes
  to hyphens, non-breaking spaces to spaces.
- Accents and letter case in text are not folded. In six languages they
  distinguish real values.
- A float with an integer value equals the integer.

`exact` is true only when the complete canonical answer equals the complete
canonical reference answer. This is the strict correctness verdict.

## Content-exact verdict

Some answer formats contain a field that counts the candidate set, not the
answer. Example: `"reviewed": 245` in "among all works filmed in Berlin, which
work has the most cast members?". The task declares these fields in
`answer.bookkeeping`. 20 tasks declare one.

`content_exact` is true when every field is correct except the bookkeeping
fields. It measures the answer itself. `exact` also requires the correct count
of the candidate set.

The scorer reports both verdicts. The bookkeeping fields are still scored and
shown in the field detail.

## Graded detail

`grading.py` adds detail beside the verdict. It never changes the verdict.

- **List fields.** Rows are matched by identity: `qid`, `id`, `entity`,
  `item`, `language` or `lang`, when every row has the key and its values are
  unique. The detail gives precision, recall, F1, the missing identities and
  the extra identities. A list without an identity uses multiset overlap.
- **Scalar and object fields.** Each field is correct or not correct.
- **`set_f1`** is the mean F1 over the list fields of the answer.

With this detail, a run that found 11 of 12 entities is different from a run
that found 3, and the report names the missing entity.

## Coverage

`coverage.py` measures completeness from the read log. It does not read the
task file.

1. It collects every `edges` call: anchor entity, direction and property.
   Pages of the same call are combined.
2. For each call, it counts the distinct entities that the run obtained.
3. It asks the graph for the true number of matching edges (`count_edges`).
4. It reports the widest candidate set that the run touched, and the share of
   that set that the run read.

A coverage of 1.0 on the widest set shows that the run read the complete pool.
The model decides how to explore. The scorer decides whether it explored.

## Grounding

`grounding.py` checks each quotation in a final answer. A quotation is a text
value under `evidence`, `text`, `description` or `subject_evidence`, attributed
to an entity in the same answer object.

| Verdict | Meaning |
|---|---|
| `GROUNDED` | The text is a stored description of the entity, and the run read that description. |
| `BLIND` | The text is a stored description of the entity, but the run did not read it. |
| `NOT_VERBATIM` | The text is not identical to a stored description of the entity: for example, stored text inside added prose, changed punctuation, or text not in the graph. |
| `UNCHECKED` | The quotation has no entity to attribute it to. |

The calls that show description text are `descriptions`, `describe`,
`description`, `entity` and `search_entity`. `edges` and `name` show labels
only, so they cannot ground a description.

`BLIND` is important for a training corpus. The answer is correct, but the
trajectory cannot account for it: it teaches a model to answer from memory. The
grounding report lists the `BLIND` and `NOT_VERBATIM` quotations, so a person
can read each one.

## Reference answer audit

`audit_gold.py` checks every reference answer value that the graph can
confirm. It reports findings. It does not change a task.

| Check | Finding |
|---|---|
| `qid_absent` | A QID in the reference answer is not in the graph. |
| `name_vs_label_en` | A `name` differs from the stored English label. |
| `evidence_not_stored` | An `evidence` text differs from the stored description in its language. |
| `evidence_lang_absent` | The entity has no description in the stated language. |
| `evidence_unattributed` | An `evidence` text without a language matches no stored description. |
| `lang_not_available` | A language code is not available for the entity. |
| `not_in_graph` | A string value exists nowhere in the entity's labels or descriptions. It is a convention, and the answer format must state it. |
| `count_mismatch` | A `count` field does not equal the length of the list that it counts. |

The report goes to stdout and to `results/gold_audit.json`.

## Batch summary

`summary.py` reads `scores.json` and gives the batch totals and one row per
execution pattern:

| Column | Source |
|---|---|
| exact, content exact | Number of runs with the verdict true |
| reached FINAL | Runs with status `final`: the harness accepted a `FINAL` |
| max_iters | Runs that used all turns without an accepted `FINAL` |
| turns | Sum of the turns used |
| graph reads | Sum of the charged reads. A repeated read of cached data is free. |
| total tokens | Sum of the prompt and output tokens of root turns and sub-queries |

## Reproducibility

Every batch records the git revision, the model name, the generation settings
and the task manifest in `batch.json`. Every run records its generation
settings in `metadata.json`. `compare.py` matches two batches by task and
lists the tasks whose verdict differs.
