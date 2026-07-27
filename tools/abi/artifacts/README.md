# ABI artifacts

Maintainer: **JamePeng**

Place trusted Windows, Linux, and macOS build artifacts here for local ABI
inspection. Filenames do not need to follow a fixed convention.

```text
tools/abi/artifacts/
├── windows-x86_64/
├── linux-x86_64/
├── macos-arm64/
└── macos-x86_64/
```

Downloaded and copied content is ignored by both the local and repository
`.gitignore`; only this README and `.gitignore` are tracked. `git add -f` can
still deliberately override ignore rules.

See `tools/abi/README.md` for commands and artifact provenance requirements.
