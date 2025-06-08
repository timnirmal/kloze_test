import os
from dotenv import load_dotenv

# load all the vars in .env into os.environ
load_dotenv()

# pull them into variables
LIVEKIT_WS_URL = os.getenv("LIVEKIT_WS_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_CRED_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    google,
    silero,
    noise_cancellation,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=google.STT(
            model="chirp_2",
            spoken_punctuation=False,
            credentials_file=GOOGLE_CRED_PATH,
            # project=GOOGLE_PROJECT,
            location=GOOGLE_LOCATION,
            # use_streaming=False
        ),
        llm=google.LLM(
            model="gemini-2.0-flash",
            api_key=GOOGLE_API_KEY,
            vertexai=True,
            project=GOOGLE_PROJECT,
            location=GOOGLE_LOCATION,
        ),
        tts=google.TTS(
            gender="female",
            voice_name="en-US-Chirp3-HD-Leda",
            credentials_file=GOOGLE_CRED_PATH,
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        # room_input_options=RoomInputOptions(
        #     noise_cancellation=noise_cancellation.BVC()
        # ),
    )

    await ctx.connect()
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=LIVEKIT_WS_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
    )
