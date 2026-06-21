# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# Git
- Do not add Co-Authored-By or self-attribution lines to commit messages. Confidence: 0.95
- Check open GitHub issues and use them to plan implementation work. Confidence: 0.75
- Use `bravo1goingdark` as the git author name (not the real name) for commits. Confidence: 0.95

# Workflow
- Research the codebase and check whether features are already implemented before building new ones. Confidence: 0.75
- Deliver production-grade quality (proper error handling, edge cases, optimization) rather than quick prototypes. Confidence: 0.80
- Read existing docs (CHANGELOG.md, future.md) to understand current state and remaining work before planning. Confidence: 0.70

# Documentation
- Update CHANGELOG.md under the "Unreleased" section when making changes. Confidence: 0.75

# Architecture
- afp is an audio-only SDK; do not add matcher or storage layers — those are handled by the user. Confidence: 0.95
