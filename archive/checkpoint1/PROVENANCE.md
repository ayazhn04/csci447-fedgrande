# Provenance of archived Checkpoint 1 evidence

Archived: 2026-11-07T04:50:55Z on macOS 26.4.1 (arm64).
Hashes: see SHA256SUMS (verify with `shasum -a 256 -c SHA256SUMS` from this folder).

| File | Kind | Source | Byte-preserved? |
|---|---|---|---|
| speeddating_checkpoint_loader.py | ORIGINAL | ~/Downloads/speeddating_checkpoint_loader.py (mtime 2026-11-06 13:27), copied with `cp -p -n` | Yes. SHA256 equals the source file at archive time. |
| Q14_GRANDE_execution_evidence.txt | ORIGINAL | ~/Downloads/Q14_GRANDE_execution_evidence.txt (mtime 2026-11-06 13:27), copied with `cp -p -n` | Yes. SHA256 equals the source file at archive time. |
| checkpoint1_moodle_answers_from_chat_paste.txt | RECONSTRUCTION (text) | Text the user copied from the NU Moodle quiz review page and pasted into the project's development-tool session on 2026-11-06. Extracted programmatically (no manual edits) from user message line 4 of the session transcript 739aa795-c84f-4474-a09c-1b3ad59812d5.jsonl, as the exact characters between the third `<pasted_content>` open/close tag pair, UTF-8, no trailing newline added. | No. No original Moodle export file exists on this machine. This is the faithful text of the paste, not a downloaded file. Formatting from the web page (tables, attachments, images) is lost, and its hash is NOT the hash of any Moodle export. |

Transcript SHA256 at extraction time (it is an append-only log and will change afterwards):
b06095ce15133ec5bcede4fd5ccbc8b25551e77786439549770ef808464a8bad

Originals in ~/Downloads were not modified. The archived files are made read-only. Corrections are recorded only in ERRATA.md and never by editing the evidence.
