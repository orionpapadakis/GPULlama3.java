---
name: update-langchain4j-integration
description: Update the LangChain4j GPULlama3 integration to a new GPULlama3.java release. Use when bumping io.github.beehive-lab:gpu-llama3, adapting the adapter to API or capability changes, enabling inherited integration tests, validating JDK/TornadoVM combinations and demos, or updating the module documentation.
---

# Update the LangChain4j GPULlama3 integration

Treat a version bump as an integration migration. Discover current repository state and preserve
unrelated changes.

## Inputs

Determine or ask for:

- GPULlama3 base version
- GPULlama3.java, LangChain4j, and demo checkout paths
- exact GGUF paths for chat and capability-specific tests
- supported JDK/TornadoVM backend combinations

Never guess paths, versions, models, flags, or backends.

## 1. Inspect the release

Read the GPULlama3 changelog, release diff, POM, and changed public APIs. Confirm both artifacts:

- `gpu-llama3:<version>-jdk21`
- `gpu-llama3:<version>-jdk25`

For a published release:

```bash
/path/to/GPULlama3.java/.claude/skills/update-langchain4j-integration/scripts/inspect-release.sh <version>
```

Otherwise build and inspect both JARs locally. Among others check model loading/generation, metrics,
`ChatFormat`, tool/thinking support, stop tokens, and TornadoVM lifecycle APIs.

## 2. Inspect and update LangChain4j

Locate the integration rather than assuming its current layout:

```bash
rg -n "gpu-llama3|langchain4j-gpu-llama3|jdk21|jdk25" pom.xml langchain4j-gpu-llama3
```

Read the module POM, adapter classes, inherited IT subclasses, module README, root JDK profiles,
and BOM entry.

Keep changes minimal and backend agnostic:

- use `TORNADOVM_HOME`; leave backend configuration to the selected SDK
- select matching `-jdk21`/`-jdk25` dependencies from JDK-specific profiles
- use `MODEL` for tests; never hardcode a GGUF path
- preserve inherited ITs and enable only implemented capabilities
- keep unsupported feature combinations disabled with accurate reasons

For a new capability, implement both request and response mappings, including conversation
history, metadata, finish reasons, and streaming callbacks. Add deterministic unit tests where
possible, then enable inherited ITs incrementally.

Tool calling does not imply forced/named tool choice or structured-output support. Verify calls
with arguments, without arguments, multiple calls, tool results followed by final answers, and
sync/streaming callbacks separately.

## 3. Validate

Use focused tests while iterating, preferably in fresh TornadoVM processes when diagnosing shared
device state. Classify failures as adapter, model behavior, GPULlama3, TornadoVM, or test harness.

Completion requires the entire sequence in
[whole-chain-validation.md](references/whole-chain-validation.md) for every claimed JDK/backend:

1. set up Java and TornadoVM
2. build the required LangChain4j modules
3. run the full module ITs
4. build the demos
5. run the required demos

Compilation or focused tests alone do not complete the update.

## 4. Curate and report

Update only affected documentation with commands that actually passed. Document supported
versions, matching SDKs, preview requirements, `TORNADOVM_HOME`, `MODEL`, and unsupported
capabilities. Remove stale versions, machine paths, generated classpaths, and obsolete commands.

Finish with:

```bash
git diff --check
./mvnw -pl langchain4j-gpu-llama3 spotless:check
git status --short
```

Report the API changes, exact JDK/backend/models tested, every whole-chain result, failures,
disabled capabilities, publication compatibility, and files that must not be committed. Remember
that source profiles do not make one flattened, published LangChain4j POM dual-JDK compatible.
Never check a PR checklist item for tests that were not run.
