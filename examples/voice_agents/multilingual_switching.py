import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    inference,
    stt,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("multilingual-switching")

load_dotenv()

# Silent multilingual switching: the adapter runs a language-pinned primary STT plus an
# always-on multilingual detector STT. When the caller switches language mid-conversation
# the primary connection is moved to the new language without losing any transcript —
# during the transition the detector's transcripts are used instead.
multilingual_stt = stt.MultilingualAdapter(
    # the primary can be any streaming STT; Deepgram supports in-place language reconnects
    primary=deepgram.STT(model="nova-3", language="en"),
    # the detector must be a multilingual model that reports the detected language
    detector=deepgram.STT(model="nova-3", language="multi"),
    # mix & match instead: keep the primary provider language-pinned via a factory
    # (used automatically when the primary has no update_options(language=...))
    # primary_factory=lambda lang: deepgram.STT(model="nova-3", language=str(lang)),
    initial_language="en",
    options=stt.LanguageSwitchOptions(
        # only switch between languages the rest of the stack (TTS voice, prompt) supports
        languages=["en", "hi"],
        # set to False for a telemetry-only rollout: evidence + suppressed events are
        # emitted but only manual (function tool) switches are performed
        auto_switch=True,
    ),
)


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Reply in the language the "
            "user speaks. If the user asks to continue the conversation in a different "
            "language, call the set_language tool.",
        )

    @function_tool
    async def set_language(self, context: RunContext, language: str):
        """Called when the user explicitly asks to continue in a different language.

        Args:
            language: BCP-47 code of the requested language, e.g. "en" or "hi"
        """
        try:
            await multilingual_stt.switch_language(language)
        except stt.LanguageNotAllowedError as e:
            return f"Sorry, {language} is not supported. Supported languages: {e.allowed}"
        except stt.LanguageSwitchFailedError:
            return "Could not switch the transcription language, please try again."

        return f"Transcription switched to {language}."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=multilingual_stt,
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    @multilingual_stt.on("language_switch_started")
    def _on_switch_started(ev: stt.LanguageSwitchStartedEvent):
        # fired before the switch completes: retune TTS voice / prompt here so the
        # agent's next reply already matches the user's language
        logger.info(
            "language switch started: %s -> %s (%s)", ev.old_language, ev.new_language, ev.initiator
        )

    @multilingual_stt.on("language_switched")
    def _on_switched(ev: stt.LanguageSwitchedEvent):
        logger.info(
            "language switched: %s -> %s via %s in %.2fs (trigger: %r)",
            ev.old_language,
            ev.new_language,
            ev.executor,
            ev.latency,
            ev.trigger_transcript,
        )

    @multilingual_stt.on("language_switch_suppressed")
    def _on_suppressed(ev: stt.LanguageSwitchSuppressedEvent):
        logger.info("language switch suppressed: %s (%s)", ev.target_language, ev.reason)

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
