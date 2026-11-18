# Hermes Frontier Review Test

Test repository for Frontier Review Phase 3D integration testing.

## Purpose
This repository is used exclusively for automated integration testing of the 
Frontier Review autonomous correction loop with live GitHub PRs.

- Tests draft PR discovery via real GitHub API
- Verifies mock review → Coder correction → re-review pipeline
- Validates SHA tracking, credential isolation, and negative test paths
