# Whole-chain validation

This is the verification source of truth. Run all five stages for every claimed JDK/backend.

## Inputs

```bash
export LANGCHAIN4J_DIR=/path/to/langchain4j
export DEMO_DIR=/path/to/gpullama3-langchain4j-demo
export MODEL=/exact/path/to/model.gguf
```

Use an exact tool-capable model path for tool ITs when needed. Never commit model paths.

## 1. Set up Java and TornadoVM

```bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk use java <java-candidate>
sdk use tornadovm <matching-jdk-and-backend-candidate>

java -version
test -n "$TORNADOVM_HOME"
"$TORNADOVM_HOME/bin/tornado" --version
"$TORNADOVM_HOME/bin/tornado" --devices
test -f "$MODEL"
```

The Java and TornadoVM JDKs must match. Obtain any required native-library environment from the
selected SDK; do not put backend-specific paths in project configuration.

## 2. Build the required LangChain4j modules

```bash
cd "$LANGCHAIN4J_DIR"
./mvnw \
  -pl langchain4j-bom,langchain4j-agentic,langchain4j-gpu-llama3 \
  -am clean install \
  -DskipTests \
  -DskipITs \
  -Drevapi.skip=true

./mvnw -pl langchain4j-gpu-llama3 dependency:tree \
  -Dincludes=io.github.beehive-lab:gpu-llama3
```

Confirm the resolved GPULlama artifact matches the active JDK.

## 3. Run the integration tests

```bash
/path/to/GPULlama3.java/.claude/skills/update-langchain4j-integration/scripts/validate-langchain4j-integration.sh \
  "$LANGCHAIN4J_DIR"
```

This stage passes only when tests are discovered, at least one starts, JUnit reports zero failures,
and the command exits successfully. Use isolated tests only for diagnosis, then rerun this stage.

## 4. Build the demos

```bash
cd "$DEMO_DIR"
mvn clean package dependency:build-classpath \
  -Dmdep.outputFile=cp.txt \
  -DskipTests

mvn dependency:tree \
  -Dincludes=dev.langchain4j:langchain4j-gpu-llama3,io.github.beehive-lab:gpu-llama3
```

Confirm the intended local artifacts were resolved. Do not commit `cp.txt`.

## 5. Run the demos

Read current commands and classes first:

```bash
cd "$DEMO_DIR"
rg -n "tornado|--params|public static void main|TicTacToe" README.md src/main/java
```

Run the documented TornadoVM commands for:

1. basic chat
2. streaming chat
3. the capability changed by this release, when applicable
4. Tic-Tac-Toe through completion

Do not reuse old class names, JAR names, model parameters, or JVM options without verifying them.
Startup alone is not a pass; each demo must perform inference and reach its expected end.

## Record

Record Java/TornadoVM versions, backend/device, resolved dependencies, model, IT counts, build
results, and every demo command/result.
