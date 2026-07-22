# LiveKit Agents — Architecture & Design Deep Dive

A ground-up analysis of this framework (v1.5.0, core package ≈ 46k LOC + ~60 provider
plugins): high-level design, low-level design, and the patterns holding it together.
File references are relative to `livekit-agents/livekit/agents/` unless stated otherwise.

---

## Table of Contents

1. [Bird's-Eye View](#1-birds-eye-view)
2. [High-Level Design](#2-high-level-design)
3. [Process & Orchestration Layer (Worker / Jobs / IPC)](#3-process--orchestration-layer)
4. [The Voice Runtime (AgentSession / AgentActivity)](#4-the-voice-runtime)
5. [Model Abstraction Layer (LLM / STT / TTS / VAD / Realtime)](#5-model-abstraction-layer)
6. [I/O & Transport Layer (RoomIO / Console / Avatar / Recorder)](#6-io--transport-layer)
7. [Plugin Ecosystem & Packaging](#7-plugin-ecosystem--packaging)
8. [Telemetry, Metrics & the Testing/Eval Harness](#8-telemetry-metrics--testingeval-harness)
9. [End-to-End Flows](#9-end-to-end-flows)
10. [Design Patterns Catalog](#10-design-patterns-catalog)
11. [Latency Engineering Techniques](#11-latency-engineering-techniques)
12. [Takeaways for Building Your Own Framework](#12-takeaways-for-building-your-own-framework)

---

## 1. Bird's-Eye View

LiveKit Agents is a **realtime voice-AI application server**. A single Python program
(`AgentServer`) registers with a LiveKit SFU, receives "jobs" (one per room/call),
executes each job in an isolated subprocess, and inside each job runs an
**AgentSession** — a streaming pipeline that turns user audio into agent speech:

```
audio in → VAD ─┬→ turn detection → LLM → tool calls → TTS → audio out
                └→ STT ────────────┘         ↑ interruption logic cuts across everything
```

Three largely independent concerns are layered on top of each other:

| Plane | What it does | Key modules |
|---|---|---|
| **Orchestration plane** | worker registration, job dispatch, process pool, supervision, hot reload | `worker.py`, `job.py`, `ipc/`, `cli/` |
| **Conversation plane** | turn taking, speech scheduling, interruptions, tools, handoffs | `voice/` |
| **Model plane** | provider-agnostic LLM/STT/TTS/VAD interfaces + 60 plugins | `llm/`, `stt/`, `tts/`, `vad.py`, `inference/`, `livekit-plugins/` |

Plus a **transport plane** (`voice/io.py`, `voice/room_io/`) that makes the conversation
plane agnostic to *where* audio comes from and goes to (WebRTC room, local mic/speaker,
avatar worker, test harness).

The single most important architectural property: **every seam is an abstract,
composable streaming interface**. Models are async-iterator streams, I/O endpoints are
chainable sources/sinks, and the conversation engine only ever talks to abstractions.
That is what lets the same `AgentSession` run against a WebRTC room in production, a
laptop mic in `console` mode, and fake models in unit tests — with zero changes.

### What user code looks like

```python
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, function_tool, inference
from livekit.plugins import silero

@function_tool
async def lookup_weather(context: RunContext, location: str):
    """Used to look up weather information."""
    return {"weather": "sunny", "temperature": 70}

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="..."),
    )
    agent = Agent(instructions="You are a friendly voice assistant.", tools=[lookup_weather])
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user")

if __name__ == "__main__":
    cli.run_app(server)
```

The mental model is small: `AgentServer` (process) → `JobContext` (one call) →
`AgentSession` (the pipeline + state machines) → `Agent` (instructions + tools +
optional per-agent model overrides). Everything else is machinery beneath these four.

---

## 2. High-Level Design

### 2.1 Process model

The framework is **multi-process by design** (one Python process cannot host many
concurrent voice pipelines — CPU-bound audio/inference work would starve the event loop):

```
┌────────────────────────── main process ───────────────────────────┐
│  AgentServer (worker.py)                                          │
│   ├─ WebSocket ↔ LiveKit server  (register/availability/assign)   │
│   ├─ ProcPool (ipc/proc_pool.py) — pre-warmed job executors       │
│   ├─ InferenceProcExecutor — ONE shared inference subprocess      │
│   ├─ HttpServer (health), Prometheus, load monitor thread         │
│   └─ CLI / hot-reload watcher (dev mode)                          │
└──────┬──────────────────────┬─────────────────────────┬───────────┘
       │ socketpair duplex    │                         │
┌──────▼───────┐      ┌───────▼──────┐          ┌───────▼─────────────┐
│ job process  │      │ job process  │   ...    │ inference process   │
│ (1 job each) │      │              │          │ (shared local ML:   │
│ JobContext   │      │              │          │  turn detector etc.)│
│ AgentSession │      │              │          │ ThreadPoolExecutor  │
└──────────────┘      └──────────────┘          └─────────────────────┘
```

- **Job executors** are a strategy: `ProcJobExecutor` (subprocess, default) or
  `ThreadJobExecutor` (Windows / console mode), both behind the `JobExecutor` protocol
  (`ipc/job_executor.py:9`). The pool doesn't care which.
- **Processes are pre-warmed** (model loading via `prewarm_fnc` happens before a job
  arrives), and the number of idle warm processes is *continuously retargeted* from
  measured CPU headroom (`worker.py:750`).
- The **inference process** loads heavy local models (end-of-utterance turn detector)
  exactly once; job processes reach it via request/response IPC relayed through the
  main process (`ipc/inference_proc_executor.py`). Light local models (Silero VAD)
  instead run in-process on a thread executor.

### 2.2 Domain-object hierarchy (conversation plane)

```
AgentSession  ── public facade, owns I/O + options + chat history + state machines
   │ 1..1 (current)
AgentActivity ── the engine: binds ONE Agent to the session while it's active;
   │             owns recognition loop, speech scheduler, interruption logic
   │ 1..1
Agent         ── declarative: instructions, tools, lifecycle hooks, node overrides
   │
SpeechHandle  ── one scheduled utterance; the synchronization primitive for
                 authorization, interruption, playout, and multi-step tool loops
```

- A **handoff** (tool returning another `Agent`) swaps the `AgentActivity`, not the
  session — chat history, I/O, and options persist across agents.
- `AgentTask[T]` is an *awaitable* Agent used as a sub-conversation that returns a
  typed result (collect an email, confirm a credit card) and then resumes the parent.

### 2.3 Uniform model contract

All four model types (LLM, STT, TTS, VAD — plus RealtimeModel) share one shape:

```
Model(ABC, EventEmitter["metrics_collected" | "error"])
  └─ .stream() / .chat() / .synthesize()  →  Stream(ABC)
        input:  push_frame()/push_text()  →  aio.Chan (input channel)
        body:   abstract _run() wrapped in a shared retry loop (_main_task)
        output: aio.Chan (event channel), tee'd → [caller, metrics monitor task]
```

A plugin author implements **only `_run()`** (and `_recognize_impl()` for batch STT);
retries, timeouts, metrics, telemetry spans, resampling, and channel lifecycle are
inherited. This is the framework's biggest leverage point — see §5.

### 2.4 Pluggable I/O

`AgentSession` reads from an `AudioInput` (async iterator) and writes to
`AudioOutput`/`TextOutput` sinks (`voice/io.py`). Concrete bindings:

| Binding | Input | Output |
|---|---|---|
| Production | `RoomIO` → WebRTC track | published track + transcription text streams |
| `console` | sounddevice mic + AEC | sounddevice speaker |
| Avatar | (same) | byte-stream to avatar worker, RPC playback signaling |
| Recording | `RecorderAudioInput` wrapper | `RecorderAudioOutput` wrapper (chained) |
| Tests | fake inputs | `QueueAudioOutput` / fakes |

Sinks form **chains** (`next_in_chain`): e.g. transcript-synchronizer → recorder →
room output. Playback events bubble *up* the chain; frames flow *down*.

---

## 3. Process & Orchestration Layer

### 3.1 `AgentServer` (`worker.py`)

**Registration & dispatch protocol.** `_connection_task` (`worker.py:981`) maintains a
WebSocket to the LiveKit server (reconnect loop with capped backoff, JWT with
`agent=true` grant). Message flow:

1. worker sends `register` (type, permissions, agent_name) → server replies with worker id.
2. Server sends `availability` request for a job → worker answers after checking load
   and running the user's `request_fnc`.
3. On accept, worker registers a `Future` in `_pending_assignments` and waits ≤7.5 s
   for an `assignment` message carrying the room URL + join token
   (`_answer_availability`, `worker.py:1232`).
4. Worker launches the job on a warm process from the pool.
5. On reconnect, the worker sends the list of running jobs so the server re-adopts them.

**Load control** is reservation-based: `_get_effective_load()` (`worker.py:1207`) =
measured CPU load + `reserved_slots × per-job load estimate`. The slot is reserved the
moment an availability request starts processing, closing the race where two
simultaneous requests both look "cheap" because neither job has started yet. The default
load function is a daemon-thread CPU sampler feeding a 2.5 s moving average.

**Dev/prod duality.** `ServerEnvOption[T]` (`worker.py:134`) is a `(dev_default,
prod_default)` pair: in dev mode load threshold is ∞ and no processes are pre-warmed;
in prod threshold is 0.7 and idle processes ≈ CPU count. One config type, two profiles.

**Drain semantics.** `drain()` first waits for in-flight availability tasks to finish
*launching* accepted jobs, then waits (≤30 min) for every process with a running job to
exit — new work is refused via `WS_FULL` status the whole time.

### 3.2 Job lifecycle (`job.py`, `ipc/job_proc_lazy_main.py`)

`JobContext` is the per-call API: `ctx.connect()` joins the room, `ctx.room`,
`ctx.api` (lazy server API client), `add_shutdown_callback`, participant entrypoints,
SIP helpers, and `get_job_context()` via a `ContextVar` so deeply nested code can find
it. Inside the child process, `_run_job_task` (`job_proc_lazy_main.py:291`) runs the
user entrypoint under an OTel span, warns if `connect()` was never called, and —
importantly — **crash paths still resolve the shutdown future**, so shutdown callbacks
and crash-log flushing always run.

`JobProcess.userdata` is the prewarm stash: `prewarm_fnc(proc)` loads models into it
once per *process*, entrypoints read them once per *job*.

### 3.3 IPC internals (`ipc/`)

- **Transport**: plain `socket.socketpair()` duplexes; framing = 4-byte length prefix +
  4-byte message id + hand-rolled big-endian binary fields (`ipc/channel.py`). No
  pickle for control messages, no shared memory. The same protocol runs over four
  transports: job subprocess, job thread, inference subprocess, and the hot-reload
  channel.
- **`SupervisedProc`** (`ipc/supervised_proc.py:76`) is a template-method base class
  that fixes the child lifecycle: initialize (with timeout → kill), ping/pong liveness
  (unresponsive → stack-dump + kill), optional psutil memory warn/limit kill, graceful
  `ShutdownRequest` → escalating `SIGUSR1` (dump) → `kill()`. Subclasses fill in
  `_create_process` and `_main_task` only.
- **Signal discipline**: children `SIG_IGN` INT/TERM *at import time* (parent owns
  shutdown); the parent masks Ctrl-C around `Process.start()` with a refcounted
  context manager so a mid-spawn SIGINT can't kill a half-initialized child
  (`supervised_proc.py:33`).
- **Log forwarding**: a `LogQueueHandler` in the child pickles log records (stripped of
  non-picklable fields) onto a queue drained by a forwarder thread into the log socket;
  the parent's `LogQueueListener` re-injects them with job/pid context fields.
- **ProcPool** (`ipc/proc_pool.py`): a 0.1 s control loop keeps
  `max(target_idle, jobs_waiting)` warm processes, with a semaphore capping concurrent
  initializations at ~CPU count. `launch_job` pops a warm proc (3 attempts, closing
  broken ones).
- **Inference RPC round-trip**: job code → `InferenceRequest` over job socket → main
  process relays to inference process → ONNX runs in a thread pool → `InferenceResponse`
  matched by `request_id` futures at each hop.

### 3.4 CLI (`cli/`)

- **`dev` + hot reload** (`cli/watcher.py`): a parent `WatchServer` uses `watchfiles`
  to run the worker as a restartable subprocess, watching the main file *and any
  editable-installed livekit packages* (detected via `direct_url.json`). Before
  restarting, it asks the old worker for its active jobs; the new worker **re-adopts
  the same rooms** with JWTs re-minted (+1 h expiry) from the original grants — a code
  reload does not drop live calls (`worker.py:1139`).
- **`console`**: runs the server on a background thread, `unregistered` (no LiveKit
  server at all), forces the thread executor, and calls `simulate_job(fake_job=True)`
  with an autospec'd mock `rtc.Room`. Local audio goes through `sounddevice` at 24 kHz
  with a real `rtc.AudioProcessingModule` (echo cancellation / noise suppression / AGC),
  bridged into asyncio with `call_soon_threadsafe`. Ctrl-T toggles text-only mode by
  simply reassigning `session.input.audio = None` — a live demo of the I/O abstraction.
- **`connect`**: same as console but joins a *real* named room, bypassing the dispatcher.
- **`download-files`**: iterates `Plugin.registered_plugins`, calling each plugin's
  `download_files()` (model weights prefetch).

---

## 4. The Voice Runtime

The heart of the framework (`voice/`). Read this section twice.

### 4.1 Responsibilities split

- **`AgentSession`** (`agent_session.py:193`) — facade + wiring. Owns global chat
  context, I/O holders, options, usage collector, and the two state machines. Public
  verbs (`say`, `generate_reply`, `interrupt`, `commit_user_turn`, `update_agent`) are
  thin delegations to the current activity. Also pumps input:
  `async for frame in audio_input: activity.push_audio(frame)`.
- **`AgentActivity`** (`agent_activity.py:109`) — the engine bound to one `Agent`. Owns
  the `AudioRecognition` loop, the speech scheduler, the realtime-model session (if
  any), interruption state, preemptive generation, tool orchestration, and draining.
  It implements `RecognitionHooks`, so the recognition loop calls *back into it*
  (`on_start_of_speech`, `on_end_of_turn`, `on_interruption`, …) — inversion of control
  between the input loop and the conversation engine.
- **`Agent`** (`agent.py:36`) — declarative. Per-agent overrides of
  stt/llm/tts/vad/turn-detection resolve as "agent value if given, else session value."
  Lifecycle hooks: `on_enter`, `on_exit`, `on_user_turn_completed`.

### 4.2 The "node" customization seam

Every pipeline stage is an overridable async method on `Agent`:

```
stt_node(audio) → SpeechEvents      llm_node(chat_ctx, tools) → ChatChunks
tts_node(text) → AudioFrames        transcription_node(text) → text
realtime_audio_output_node(audio) → audio
```

Each instance method delegates to a static default under `class default:`
(`agent.py:386-514`). Defaults transparently insert adapters — e.g. a non-streaming
STT gets wrapped in `stt.StreamAdapter` with the VAD; a non-streaming TTS gets a
sentence tokenizer + `tts.StreamAdapter`. Overriding a node lets users rewrite the LLM
stream, intercept TTS text, filter transcripts, etc., without touching the engine.
**This is the framework's primary extension point below the model interfaces.**

### 4.3 Anatomy of one turn (pipeline mode)

1. **Frames in.** `activity.push_audio(frame)` forwards to `AudioRecognition`, which
   fans each frame into three channels: STT, VAD, adaptive-interruption
   (`audio_recognition.py:384`). VAD *always* sees frames, even when STT input is
   discarded (e.g. during uninterruptible agent speech).
2. **VAD** emits `START_OF_SPEECH` / `INFERENCE_DONE` / `END_OF_SPEECH`. Start-of-speech
   opens a `user_turn` telemetry span and cancels any pending end-of-turn task.
3. **STT** streams interim → preflight → final transcripts; finals accumulate into the
   pending user turn and fire preemptive generation (§4.6).
4. **End-of-turn detection** (`_run_eou_detection`, `audio_recognition.py:809`): build a
   candidate chat context with the pending user message; if a semantic turn-detector
   model is configured, ask it for an end-of-turn probability — below the threshold the
   endpointing delay stretches from `min_delay` to `max_delay` ("user probably isn't
   done"). Then sleep the delay in a **cancellable task** — the user resuming speech
   cancels it. If it survives, call `on_end_of_turn`.
5. **Turn commit.** `on_end_of_turn` is deliberately synchronous (so the recognition
   loop can't cancel it mid-flight) and spawns the turn-completion task: interrupt
   current speech, run `Agent.on_user_turn_completed` (which may mutate the context or
   `raise StopResponse` to veto a reply), then adopt the preemptive generation or start
   a fresh one.
6. **Generation pipeline** (`generation.py`): the LLM stream is split into a **text
   channel** and a **function-call channel**. Text is tee'd to (a) TTS and (b) the
   transcription output. TTS output frames are resampled and pushed into the
   `AudioOutput`. All of this *starts computing immediately* but blocks on an
   **authorization barrier** before any output escapes (§4.4).
7. **Tools** execute as one asyncio task per call (parallel), shielded from
   cancellation so interrupts wait for in-flight tools. A tool returning an `Agent`
   triggers handoff; returning a value marks "reply required."
8. **Playout & commit.** First output frame flips agent state to `speaking` and records
   TTFT/TTFB/e2e latency. After forwarding completes, the task awaits
   `audio_output.wait_for_playout()`, then writes the assistant message (with
   per-message latency metrics) into chat history.
9. **Multi-step loop.** If tools requested a reply, a new pipeline task runs (step
   count capped by `max_tool_steps`; the final step forces `tool_choice="none"` so the
   agent *must* answer with speech instead of looping tools forever).

### 4.4 Speech scheduling — warm-then-authorize

`SpeechHandle` (`speech_handle.py`) models one utterance with four futures:
`scheduled`, `authorized` (+ per-step generation futures), `interrupted`, `done`.

The scheduler (`agent_activity.py:1153`) pops from a **max-heap priority queue**
(`(-priority, monotonic_ns, handle)`) and performs the handshake:

```python
speech._authorize_generation()       # reply task may now emit output
await speech._wait_for_generation()  # scheduler blocks until playout done
```

The trick: reply tasks are created and start LLM/TTS inference **as soon as the turn
ends**, but sit at the authorization barrier before emitting anything. Playout is
serialized; inference is not. Every await point in the pipeline goes through
`speech_handle.wait_if_not_interrupted(...)` — a race between the work and the
interrupt future — which is how an interruption unwinds a half-built pipeline cleanly
at any stage. A 5 s watchdog force-cancels a speech that doesn't wind down after being
interrupted.

### 4.5 State machines

- **User**: `listening ↔ speaking`, plus `away` (armed by a timer only when both sides
  are listening; a final transcript snaps back).
- **Agent**: `initializing → listening → thinking → speaking → listening` (back to
  `thinking` if tools run). Transitions are emitted as events and mirrored into room
  participant attributes (`lk.agent.state`) so client UIs can render them.

### 4.6 Preemptive generation

With `preemptive_generation=True`, a *final/preflight transcript* — i.e. **before**
endpointing delay elapses — speculatively starts the LLM+TTS pipeline
(`schedule_speech=False`, so no audio can escape). When the turn actually commits, the
snapshot is validated (same transcript, equivalent chat context, same tools); if valid
the pre-warmed handle is simply scheduled — the endpointing delay overlapped with
inference. If `on_user_turn_completed` changed anything, it's discarded and a fresh
generation starts. This is the single biggest perceived-latency win in the framework.

### 4.7 The three-layer interruption model

1. **Audio-activity (VAD) interruption** — cheap and immediate: user speech longer than
   `min_duration` (and `min_words` if STT is available) interrupts the agent. Gated off
   temporarily right after the agent starts speaking (echo protection + AEC warmup).
2. **Adaptive/ML interruption** — an `AdaptiveInterruptionDetector` classifies
   overlapping speech as *real interruption* vs *backchannel* ("uh-huh"). While the
   verdict is pending, STT transcripts are **held in a buffer** and later released or
   discarded — backchannels never reach the LLM. Falls back to VAD mode on failure.
3. **False-interruption pause/resume** — if the output sink supports `pause()`, the
   agent's audio is *paused* (not destroyed) on suspected interruption. A timer waits
   for a real user turn; if none arrives, playback **resumes** where it left off and an
   `AgentFalseInterruptionEvent(resumed=True)` is emitted.

On a committed interruption the final `PlaybackFinishedEvent` carries the
**synchronized transcript** — exactly the words the user actually heard — and the
assistant chat message is truncated to it and marked `interrupted=True`. Chat history
never lies about what was spoken.

### 4.8 Turn detection & endpointing

`TurnDetectionMode` = `"stt" | "vad" | "realtime_llm" | "manual"` or a model
implementing `predict_end_of_turn`. Auto-selected and validated against available
components (invalid choices downgrade with warnings). Endpointing is pluggable:
`BaseEndpointing` (fixed min/max delays) or `DynamicEndpointing`
(`endpointing.py:49`), which *learns the user's pausing behavior* with two exponential
moving averages — one over intra-utterance pauses, one over user→agent turn gaps — and
adapts the delays per user in real time.

### 4.9 Multi-agent: handoffs and AgentTask

- **Handoff-by-return**: a `@function_tool` returns the next `Agent` (optionally with a
  string result for the LLM). The activity drains, `session.update_agent()` builds a
  new `AgentActivity`, `on_exit`/`on_enter` hooks fire, and chat history carries over.
- **`AgentTask[T]`**: awaited inline (inside a tool or `on_enter`), it pauses the
  parent activity, locks the parent speech against interruption, runs its own
  sub-conversation until `self.complete(result)`, merges its chat context back, and
  resumes the parent. `beta/workflows/` ships prebuilt tasks (GetEmailTask,
  GetPhoneNumberTask, credit card, DOB, address, warm transfer) — each a
  self-contained validated data-capture flow.

### 4.10 Concurrency toolkit patterns used throughout

- `aio.Chan[T]` — Go-style channel; the universal producer/consumer seam.
- `aio.itertools.tee` — fan a stream to N consumers **with exception propagation to
  all peers** (fixes the classic silent-swallow bug of naive tee).
- `cancel_and_wait` / `gracefully_cancel` — cancellation that awaits completion.
- `asyncio.shield` wherever draining must survive cancellation (tool gather,
  generation futures, scheduler pause).
- `contextvars` for ambient state (`current job`, `current activity`, `current speech
  handle`, mocked tools) instead of threading parameters through every call.
- OTel context is **captured and re-attached across `create_task` boundaries**
  (`agent_activity.py:476`) so trace trees stay correct.

---

## 5. Model Abstraction Layer

### 5.1 The uniform stream lifecycle

Every model stream follows one shape (LLMStream `llm/llm.py:158`, RecognizeStream
`stt/stt.py:259`, ChunkedStream/SynthesizeStream `tts/tts.py`, VADStream `vad.py:98`):

```
push side  ──►  input aio.Chan  ──►  _main_task:
                                       for attempt in range(max_retry+1):
                                           await self._run()        # ← plugin implements ONLY this
                                       (499 → silent stop; retryable? → sleep, retry;
                                        else emit "error" event + raise)
                                     ──►  output aio.Chan ──tee──► caller
                                                              └──► metrics monitor task
```

Key decisions worth copying:

- **Retryability is data, not policy.** `APIError.retryable` (4xx auto-false) drives
  one shared retry loop. HTTP 499 (client closed) always exits silently.
- **Metrics are a parallel consumer** of the same stream (via tee), never inline —
  TTFT/TTFB/duration/usage computed without touching the hot path.
- The base stream handles **input resampling** (STT), **timestamp linearity across
  reconnects** (`start_time_offset`), and channel close-on-task-done.
- `APIConnectOptions` (frozen dataclass: `max_retry=3, retry_interval=2.0,
  timeout=10.0`) flows through every call; the first retry happens after only 0.1 s.
- `NOT_GIVEN` / `NotGivenOr[T]` sentinel distinguishes "unset" from `None` across the
  entire API surface.

### 5.2 ChatContext (`llm/chat_context.py`)

A time-ordered list of discriminated-union items: `ChatMessage | FunctionCall |
FunctionCallOutput | AgentHandoff | AgentConfigUpdate`, all pydantic with `id` +
`created_at`. Notable semantics:

- **Insert by timestamp**, not append — late-arriving items (e.g. a slow transcript)
  land in the right chronological place.
- `copy(tools=...)` drops function calls/outputs not in the active toolset — this is
  what keeps handoff targets from seeing tools they don't have.
- `truncate(max_items)` strips orphaned tool calls and re-injects the system prompt;
  `_summarize(llm)` compresses old turns into a summary message, preserving the tail.
- `ChatMessage` carries a per-message **latency report** (transcription delay,
  end-of-turn delay, LLM TTFT, TTS TTFB, e2e) — observability baked into the data model.
- Agents get a `_ReadOnlyChatContext` view; mutation must go through
  `copy()` + `update_chat_ctx()` (immutability by policy).
- `Instructions` is a `str` subclass carrying **audio vs text modality variants** —
  prompts adapt to whether the agent is speaking or typing.
- `to_provider_format("openai" | "anthropic" | "google" | "aws" | "mistralai")`
  dispatches to `_provider_format/` converters. A shared normalization step groups
  parallel tool calls and prunes orphaned calls/outputs before conversion — malformed
  request prevention is provider-agnostic; provider quirks (Anthropic separate system
  prompt, dummy-user-message injection) live in each converter.

### 5.3 Tools (`llm/tool_context.py`, `llm/utils.py`)

- `@function_tool` builds a JSON schema **from the Python signature + docstring**
  (`get_type_hints(include_extras=True)`, `Annotated[T, Field(...)]` support,
  `RunContext` params skipped). `raw_schema=` bypasses it for provider-native schemas.
- Tools implement the **descriptor protocol** so decorated *methods* bind `self`
  correctly and drop it from the schema.
- Execution: parse/validate args through a generated pydantic model (with None→default
  coercion required by strict schemas), inject `RunContext`, run, then validate output
  is JSON-serializable. `ToolError` surfaces its message to the LLM; `StopResponse`
  suppresses the reply — **control flow via typed exceptions**.
- **MCP**: `MCPServer` (HTTP/SSE or stdio) lists remote tools and wraps each as a
  `RawFunctionTool` whose body is `client.call_tool(...)` — MCP tools are
  indistinguishable from local ones downstream.

### 5.4 Realtime models (`llm/realtime.py`)

A separate ABC pair for speech-to-speech models (OpenAI Realtime, Gemini Live):
`RealtimeModel` (capability flags: server turn detection, truncation support, audio
output, …) → `RealtimeSession`, an EventEmitter with `push_audio`, `generate_reply`,
`interrupt`, `truncate`. Generations surface as **streams of streams**:
`GenerationCreatedEvent → MessageGeneration{text_stream, audio_stream}` +
`function_stream`. `AgentActivity` reconciles the capability flags with local turn
detection so pipeline mode and realtime mode expose identical session semantics.

### 5.5 Adapters — capability synthesis and resilience

- **`stt.StreamAdapter`**: non-streaming STT + VAD → streaming STT (buffer speech
  between VAD start/end, then batch-recognize).
- **`tts.StreamAdapter`**: non-streaming TTS + sentence tokenizer → streaming TTS.
- **`SentenceStreamPacer`** (`tts/stream_pacer.py`): batches sentences to the TTS based
  on *remaining buffered audio*, sending the first sentence immediately — fewer wasted
  synthesis calls on interruption, more context per call.
- **Fallback adapters** (LLM/STT/TTS): a list of providers with health state. Attempts
  use `max_retry=0` (fallback replaces retry); on failure the provider is marked
  unavailable, an availability-changed event fires, and a **background recovery task
  probes it** until healthy. Guards: LLM won't switch providers after chunks streamed
  (no duplicated half-answers); TTS resamples across providers with different sample
  rates; STT fallback feeds recovering streams live traffic and requires a real final
  transcript to declare recovery.
- **`AudioEmitter`** (`tts/tts.py:671`): the one component every TTS plugin pushes raw
  bytes into. Handles codec detection (PCM vs Opus/MP3 via `AudioStreamDecoder`),
  reframing to 200 ms chunks, segment bookkeeping, delayed-flush pacing when synthesis
  is slower than realtime, timed-transcript propagation via `frame.userdata`, and debug
  WAV dumps. Plugins stay ~trivial because this exists.

### 5.6 Hosted inference gateway (`inference/`)

`inference.LLM("openai/gpt-4.1-mini")`, `inference.STT("deepgram/nova-3")`,
`inference.TTS("cartesia/sonic-3")` speak to LiveKit Cloud's model gateway with
short-lived JWTs minted from the LiveKit API key — one bill, one wire protocol
(OpenAI-compatible for LLM; WebSocket + `ConnectionPool` for STT/TTS), server-side
fallback lists. Same base classes, so they're interchangeable with plugins.

`ConnectionPool[T]` (`utils/connection_pool.py`) is the latency seam: pooled
WebSockets with max session duration, deferred reaping, and `prewarm()` — the
`Model.prewarm()` chain opens a connection before the first user turn.

---

## 6. I/O & Transport Layer

### 6.1 The contract (`voice/io.py`)

- Inputs are **pull** (async iterators) with a chainable `.source` and
  attach/detach lifecycle hooks.
- Outputs are **push** sinks with `capture_frame` → `flush()` (segment boundary) →
  `clear_buffer()` (interrupt), chained via `next_in_chain`; playback events bubble up.
- The universal completion primitive: sinks count segments and must call
  `on_playback_finished(playback_position, interrupted, synchronized_transcript)`
  exactly once per segment; `wait_for_playout()` awaits `finished == segments`.
  Every backend — room, console, avatar — implements the same accounting, which is
  why interruption logic is transport-agnostic.

### 6.2 RoomIO (`voice/room_io/`)

Binds a session to an `rtc.Room`:

- **Input**: subscribes to the linked participant's mic; the Rust-side
  `rtc.AudioStream.from_track` resamples/reframes to 24 kHz mono 50 ms frames and
  applies noise cancellation. Handles track unpublish/republish and hot-swapping the
  linked participant. After a track ends it pushes **0.5 s of silence** so the STT
  emits its final transcript.
- **Pre-connect audio**: clients buffer mic audio locally *before* WebRTC connects and
  ship it over a byte stream (`lk.agent.pre-connect-audio-buffer`); RoomIO decodes it
  and **prepends it to the live stream** — the user's first words are never lost.
- **Output**: publishes a `roomio_audio` track; frames pass through an
  `AudioByteStream` re-chunker into an `rtc.AudioSource` (200 ms queue). Interruption
  computes `playback_position = pushed − queued` so the reported position is what was
  actually *heard*. Supports pause/resume (used by false-interruption recovery).
- **Transcription output** writes both the legacy `publish_transcription` API and
  text streams (topic `lk.transcription`) with segment/final attributes.
- **State sync**: agent state is mirrored to participant attribute `lk.agent.state`.

### 6.3 Transcript synchronization (`voice/transcription/synchronizer.py`)

Captions paced word-by-word to match audio playback. Three timing strategies, best
available wins:

1. fixed speech rate (3.83 hyphens/sec) as fallback;
2. **live speaking-rate estimation** — a DSP detector computing spectral flux (STFT,
   1 s window / 0.1 s hop) over the actual TTS audio, integrated into "speaking units";
3. exact word timestamps when the TTS provides `TimedString` alignments.

On interruption, `synchronized_transcript` returns only the words actually voiced —
this is the source of truth for truncated chat history (§4.7).

### 6.4 Avatar transport (`voice/avatar/`)

The avatar worker is a separate participant. The agent's `AudioOutput` becomes a
`DataStreamAudioOutput`: TTS audio is shipped over a byte stream; the worker renders
lip-synced video and publishes audio+video via an `rtc.AVSynchronizer`. Control flow is
RPC: `lk.clear_buffer` (interrupt) and `lk.playback_finished` (remote playback state
drives the agent's `on_playback_finished`), with a 2 s timeout fabricating a finish
event if the worker never replies — no deadlock on a dead avatar.

### 6.5 Recorder & background audio

- **RecorderIO**: wraps input and output with chained pass-through recorders, writing a
  stereo OGG (left = user, right = agent) via PyAV in a dedicated encode thread,
  reconstructing pauses as silence and clamping to actually-played positions.
- **BackgroundAudioPlayer**: publishes its own track and mixes ambient/"thinking"
  sounds (state-driven: plays while agent state is `thinking`), with per-source volume
  and probabilistic selection.

---

## 7. Plugin Ecosystem & Packaging

### 7.1 Two-level plugin design

`Plugin` (`plugin.py:13`) is deliberately tiny — metadata + registration + optional
`download_files()`. The real polymorphism is subclassing the capability ABCs
(STT/TTS/LLM/VAD/RealtimeModel). Registration happens at import time and must run on
the main thread (children re-import and re-register). The worker uses registered
package names for forkserver preload; the dev watcher uses them to watch editable
installs.

### 7.2 Reference implementations

- **openai**: kitchen sink (LLM, TTS, STT, Realtime, Responses API, embeddings); also
  the base for a dozen OpenAI-compatible providers (Azure, Groq, Ollama, …).
- **deepgram**: the canonical streaming-WebSocket STT. Pattern: `_run()` spawns
  send/recv/keepalive tasks; a `_reconnect_event` raced against the tasks lets
  `update_options()` trigger reconnection with new query params; unexpected socket
  closes raise `APIStatusError` so the *base class* retry loop handles them. Raw
  aiohttp, no vendor SDK.
- **silero**: local VAD; ONNX model **bundled in the wheel** (importlib.resources),
  inference on `run_in_executor` (thread), full VAD state machine in the plugin.
- **turn-detector**: heavy local model using the **inference process**: registers an
  `_InferenceRunner` keyed by a method string; job-side client calls
  `do_inference(method, bytes)` over IPC. Weights come from HuggingFace via
  `download-files`; runtime is `local_files_only=True` with actionable errors.

The split is instructive: *light* local model → thread executor in-process; *heavy*
local model → shared inference subprocess.

### 7.3 Packaging

Each plugin is an independent pip package in a uv workspace, sharing the
`livekit.plugins.*` **PEP 420 namespace** (no `__init__.py` at the namespace levels).
Uniform hatchling pyproject; version in `version.py`; assets via wheel `shared-data`;
core dependency `livekit-agents>=x` with extras. ~60 providers, independently
versioned and released, one import namespace.

---

## 8. Telemetry, Metrics & Testing/Eval Harness

- **Metrics**: pydantic models per stage — `LLMMetrics` (TTFT, tokens/s, cached
  tokens), `STTMetrics`, `TTSMetrics` (TTFB), `VADMetrics`, `EOUMetrics` (end-of-turn
  delay, transcription delay), `RealtimeModelMetrics` — emitted on each model's
  `metrics_collected` event and aggregated by a session `UsageCollector`.
- **Tracing**: OpenTelemetry throughout, mixing LiveKit `lk.*` attributes with standard
  GenAI semconv. Span tree: job entrypoint → user turn → llm/tts request → per-retry
  attempt child spans. Chat context is attached as span events. A swappable tracer
  provider ships to LiveKit Cloud (OTLP + session report upload) or anywhere else.
- **Testing is a first-class exported surface**:
  - `tests/fake_{stt,tts,llm,vad}.py` subclass the *real* ABCs with scripted behavior
    and observable channels (assert on retries, timing, transcripts).
  - `session.run(user_input=...)` returns an awaitable `RunResult` with an ordered
    event log; `result.expect.next_event().is_function_call(name=..., arguments=...)`
    is a cursor-style fluent assertion API; `.judge(llm, intent=...)` is a built-in
    LLM-as-judge; `mock_tools()` swaps tool implementations via ContextVar.
  - Network-fault testing uses a toxiproxy harness against fallback/retry paths.

---

## 9. End-to-End Flows

### 9.1 Boot → job running

```
cli.run_app(server) → server.run()
  ├─ start inference process (if local runners registered), initialize models once
  ├─ ProcPool.start() → pre-warm N job processes (each runs prewarm_fnc)
  ├─ HTTP health server + Prometheus
  └─ WS connect → register → (loop) availability? → reserve slot → request_fnc
        → accept → await assignment (≤7.5 s) → pop warm proc → StartJobRequest
              child: build JobContext → user entrypoint → ctx.connect() → rtc.Room
                     AgentSession.start() → RoomIO.start() → link participant
                     → AgentActivity(agent) → on_enter → generate_reply(greeting)
```

### 9.2 One spoken exchange (condensed)

```
user speaks → VAD start → STT interim/final ┬→ preemptive LLM+TTS start (speculative)
                                            └→ EOU model + endpointing delay
turn commits → on_user_turn_completed → adopt speculation (or regenerate)
→ speech handle scheduled → authorized by scheduler → audio escapes
→ first frame: agent=speaking, latency metrics
→ tools run in parallel (maybe) → follow-up LLM step (≤ max_tool_steps)
→ playout drains → assistant message (with metrics) committed → agent=listening
user barges in at any point → interrupt future → unwind pipeline → clear buffers
→ truncated message committed with synchronized transcript
```

### 9.3 Hot reload (dev)

```
file saved → watchfiles → ask old worker for active jobs → restart subprocess
→ new worker re-registers → re-adopts jobs with re-minted JWTs → calls continue
```

---

## 10. Design Patterns Catalog

| Pattern | Where | Notes |
|---|---|---|
| **Template method** | `SupervisedProc` (`ipc/supervised_proc.py:76`); all model streams (`_main_task` wraps abstract `_run()`) | subclasses fill one method; lifecycle/retry owned by base |
| **Strategy** | `JobExecutor` protocol (proc vs thread); `TurnDetectionMode`; endpointing (fixed vs dynamic); noise-cancellation selector; transcript timing strategies | |
| **Adapter** | STT/TTS `StreamAdapter`; fallback adapters; `_ChunkedStreamFromStream`; provider-format converters | capability synthesis is a first-class concept |
| **Chain of responsibility / decorator** | `AudioOutput.next_in_chain`, `AudioInput.source` (`io.py:41,131`) | recorder + synchronizer + transport compose; events bubble up |
| **Facade** | `AgentSession` over activity/recognition/scheduling | public API is ~8 verbs |
| **Observer (EventEmitter)** | models, session, room I/O, plugins | typed event literals per class |
| **Producer/consumer channels** | `aio.Chan` everywhere; LLM text/function channels; scheduler queue | Go-style CSP in asyncio |
| **Inversion of control (hooks)** | `RecognitionHooks` (recognition loop → activity); Agent lifecycle hooks; node overrides | |
| **Command + priority queue** | `SpeechHandle` + max-heap scheduler (`agent_activity.py:1122`) | warm-then-authorize handshake |
| **Future-as-gate** | speech handle's 4 futures; pending assignments; participant-available; pre-connect buffers | one-shot synchronization everywhere |
| **Registry** | `Plugin.registered_plugins`; `_InferenceRunner.registered_runners`; IPC message-id table | import-time registration |
| **Descriptor protocol** | `@function_tool` methods binding `self` (`tool_context.py:173`) | schema drops the instance param |
| **Sentinel types** | `NOT_GIVEN`, `FlushSentinel`, `AudioSegmentEnd`, `_FlushSegment` | distinguish "unset"/"boundary" from data |
| **ContextVar ambient state** | job context, activity, speech handle, mock tools | no parameter threading |
| **Health-checked failover** | fallback adapters with background recovery probes | fallback ≠ retry (`max_retry=0` per attempt) |
| **Bridge/proxy over IPC** | job → main → inference process, request-id futures at each hop | one binary protocol, four transports |
| **State machines** | user/agent states; `AudioEmitter` segments; VAD state | explicit literals + events, not implicit flags |

---

## 11. Latency Engineering Techniques

Worth cataloging separately — this is where voice UX lives or dies:

1. **Preemptive generation** — LLM+TTS start on the final transcript, *before*
   endpointing delay elapses; the delay overlaps inference (§4.6).
2. **Warm-then-authorize scheduling** — inference runs eagerly; only *playout* is
   serialized (§4.4).
3. **Semantic endpointing** — turn-detector model shortens the wait when the user is
   clearly done (`min_delay`) and stretches it when they aren't (`max_delay`);
   `DynamicEndpointing` adapts both to the individual user.
4. **Process pre-warming** — models loaded before the job exists; warm-pool size
   tracks CPU headroom.
5. **Connection pre-warming** — `prewarm()` chain opens WebSockets/HTTP connections
   before the first turn.
6. **Pre-connect audio buffering** — no lost first words while WebRTC negotiates.
7. **Sentence pacing to TTS** — first sentence sent immediately; later sentences
   batched by remaining audio runway (fewer wasted calls on interruption).
8. **First-retry after 0.1 s** — transient blips recover almost instantly.
9. **AudioEmitter delayed flush** — keeps playback fed when synthesis is slower than
   realtime.
10. **Latency baked into data** — every assistant `ChatMessage` records transcription
    delay, EOU delay, TTFT, TTFB, and e2e latency for offline analysis.

---

## 12. Takeaways for Building Your Own Framework

Opinionated distillation, given the goal of designing a new voice-agent framework
(and comparing against Pipecat next):

1. **Separate the three planes from day one.** Orchestration (how agent code gets
   deployed/scaled), conversation engine (turn taking), and model interfaces are
   independently replaceable here. LiveKit's orchestration is tightly coupled to their
   SFU dispatch protocol — yours doesn't have to be — but the *seam* is what matters.
2. **The conversation engine is the hard 20%.** VAD/STT/LLM/TTS plumbing is
   commoditized; the real IP is in `agent_activity.py` + `audio_recognition.py`:
   endpointing, three-layer interruption, false-interruption resume, held-transcript
   backchannel suppression, preemptive generation, speech scheduling. Budget most of
   your design effort there.
3. **Standardize a stream contract early.** The "input channel → abstract `_run()` in
   a shared retry loop → output channel tee'd to metrics" shape makes 60 providers
   maintainable. Make `retryable` a property of errors, not call sites.
4. **Playback accounting is the keystone abstraction.** The segment-counting
   `AudioOutput` contract (`capture/flush/clear` + exactly-one `on_playback_finished`
   with position + synchronized transcript) is what makes interruptions
   transport-agnostic and chat history truthful. Design this before designing outputs.
5. **SpeechHandle-style handles beat implicit state.** One object carrying
   scheduled/authorized/interrupted/done futures gives you priority queues, barge-in,
   `await handle`, and testability for free.
6. **Decouple "agent" (declarative) from "activity" (runtime binding).** It makes
   multi-agent handoff nearly free: swap the engine binding, keep session state.
7. **Hooks + node overrides beat subclassing the engine.** Users override small async
   generators (`llm_node`, `tts_node`) rather than the orchestrator; adapters make
   non-streaming providers look streaming.
8. **Process isolation is a feature, not plumbing.** Prewarm, ping/pong kills, memory
   limits, escalating shutdown, cross-process logs, and reload-without-dropping-calls
   are the difference between a demo and an operable server.
9. **Make testing part of the core API.** Fakes subclassing real ABCs + a recorded
   event log + fluent assertions + LLM-as-judge means agent behavior is regression-
   testable. Retrofitting this later is painful.
10. **Instrument the data model.** Per-message latency metrics and OTel spans wired
    through context propagation cost little early and are near-impossible to bolt on
    later.
