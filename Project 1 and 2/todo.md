# Comp_Sci_Projects TODO

> Master checklist for completing, testing, and maintaining the repository.
> Check items off as they are completed. Do not redo completed work unless a later change requires it.

---

# Project 1 & 2

## Dataset Pipeline

- [X] Install `huggingface_hub` into the project's `.venv`
- [ ] Verify `huggingface_hub` imports from the project's Python
- [ ] Verify Hugging Face dataset sync works
- [ ] Verify the dataset is downloaded into the expected `dataset/` directory
- [ ] Verify `coco_builder.py` works with the synced dataset
- [ ] Verify `sync_dataset.pull()` runs before `build_coco_dataset()`
- [ ] Move dataset startup code into `main()`
- [ ] Make dataset sync failure stop execution
- [ ] Add clear error handling for Hugging Face failures
- [ ] Add test for sync → COCO build order
- [ ] Add test confirming COCO build does not run when sync fails
- [ ] Run the complete Project 1 & 2 playtest
- [ ] Run GitHub Actions
- [ ] Fix any issues found by CI
- [ ] Open PR
- [ ] Review PR
- [ ] Merge PR

## Project 1 & 2 Code Cleanup

- [ ] Review `main.py` for unused code
- [ ] Review `sync_dataset.py` for unused/dead code
- [ ] Review `coco_builder.py` for unused/dead code
- [ ] Review `downloader.py` for unused/dead code
- [ ] Review imports across Project 1 & 2
- [ ] Review error handling
- [ ] Review generated files
- [ ] Review `.gitignore`
- [X] Decide whether `image_classifier.keras` should remain in Git(in .gitignore)
- [ ] Split main.py into modules as follows 
Project 1 and 2/
├── main.py
├── model.py
├── training.py
├── settings.py
├── dataset.py
├── sync_dataset.py
└── coco_builder.py
- [ ] Run full playtest after cleanup
- [ ] Open cleanup PR
- [ ] Merge cleanup PR

---

# Website

- [ ] Remove dead `"Avery Chen"` replacement code

---

# Repository-Wide Project Audit

## For Image recognition 

- [ ] Identify how the project is supposed to work
- [ ] Run the project from a clean environment
- [ ] Play through every menu
- [ ] Test every option
- [ ] Test normal user workflows
- [ ] Test invalid user input
- [ ] Look for crashes
- [ ] Look for options that do nothing
- [ ] Look for unreachable code
- [ ] Look for unused/dead code
- [ ] Look for missing error handling
- [ ] Look for missing dependencies
- [ ] Add tests for important functionality
- [ ] Fix confirmed bugs
- [ ] Run tests again
- [ ] Run the complete playtest again
- [ ] Open PR
- [ ] Review PR
- [ ] Merge PR

---

# Repository Quality

- [ ] Every project has documented dependencies
- [ ] Every project can be run from a clean environment
- [ ] Important functionality has automated tests
- [ ] GitHub Actions pass
- [ ] No known broken functionality remains
- [ ] No obvious dead code remains
- [ ] No unnecessary generated files are committed
- [ ] `.gitignore` files are appropriate
- [ ] README/documentation is accurate
- [ ] Repository structure is organized
- [ ] Final repository-wide playtest
- [ ] Final repository cleanup
- [ ] Final GitHub Actions run
- [ ] Repository is ready to show/share