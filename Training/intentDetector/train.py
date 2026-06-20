from __future__ import annotations

import json
import os
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split as sk_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 1 ─ CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR   = "./intent_classifier"
CLINC_PATH   = "./datasets/clinc_oos"
BANKING_PATH = "./datasets/banking77"
SNIPS_PATH      = "./datasets/snips"               # optional – skipped if path absent
ATIS_PATH       = "./datasets/atis"                # optional – skipped if path absent
MASSIVE_PATH    = "./datasets/massive"             # optional – skipped if path absent
DIFFUSIONDB_PATH  = "./datasets/diffusionDB"   # 5k generate_image prompts
READALOUD_PATH    = "./datasets/readAloud"      # 3.5k read_aloud (CNN + Wiki)
MUSICCAPS_PATH    = "./datasets/musiccaps"      # 2.5k generate_music (MusicCaps)

# MAX_LEN is computed automatically from data; this is the hard ceiling.
MAX_LEN_CEIL   = 64
BATCH_SIZE     = 128
EPOCHS         = 8
LR             = 2e-5
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.1
FOCAL_GAMMA    = 2.0
LABEL_SMOOTHING = 0.1
SEED           = 42

# Balancing: upsample minority classes to MIN_CLASS_FLOOR,
# cap the dominant class at MAX_CHAT_CAP.
# Weighted focal loss handles the residual imbalance.
MIN_CLASS_FLOOR = 500
MAX_CHAT_CAP    = 5000

LABELS   = ["generate_image", "generate_music", "read_aloud", "chat_with_llm"]
LABEL2ID = {lbl: idx for idx, lbl in enumerate(LABELS)}
ID2LABEL = {idx: lbl for idx, lbl in enumerate(LABELS)}

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 2 ─ HARD-CODED EXAMPLES  (100 per class – emphasis on linguistic diversity)
# ══════════════════════════════════════════════════════════════════════════════

HARDCODED: dict[str, list[str]] = {

    "generate_image": [
        # core creation verbs
        "Create an image of a sunset over calm ocean water",
        "Generate a picture of a golden retriever playing fetch",
        "Draw me a cartoon rocket ship heading to the moon",
        "Make an illustration of a cozy autumn forest path",
        "Paint a digital watercolour of the Eiffel Tower at dusk",
        "Render a 3D model of a futuristic sports car",
        "Design a logo that features a running wolf",
        "Produce a photorealistic image of a bowl of ramen",
        "Create concept art for a fantasy castle on a cliff",
        "Generate anime-style artwork of a samurai warrior",
        # visualise / show / depict
        "Show me what a dragon would look like in a city skyline",
        "I want to see a tropical beach with crystal-clear water",
        "Visualise a steampunk city from a bird's-eye view",
        "Give me a picture of the Milky Way over a desert",
        "Can you depict a peaceful Japanese zen garden?",
        "I'd like an image of a cozy coffee shop interior",
        "Draw what Mars might look like after terraforming",
        "Show me an astronaut floating in outer space",
        "I want to see a majestic lion at golden hour",
        "Generate an image of autumn leaves on a winding trail",
        # stylistic
        "Create a pixel-art sprite of a treasure chest",
        "Make a Van Gogh-style painting of a starry night cityscape",
        "Generate a black-and-white photo of a 1920s jazz bar",
        "Draw a minimalist icon set for a weather app",
        "Produce a surrealist painting in the style of Dalí",
        "Create a watercolour portrait of an old fisherman",
        "Make an infographic-style image showing the water cycle",
        "Generate a vintage poster for a fictional space mission",
        "Design a book cover for a thriller novel",
        "Create low-poly art of a mountain range",
        # specific scenes
        "Generate an image of two cats sitting on a windowsill",
        "Draw a knight battling a sea monster in rough waters",
        "Make a holiday-themed picture with snowflakes and pine trees",
        "Create an image of a robot chef cooking in a kitchen",
        "Produce artwork showing a phoenix rising from ashes",
        "Generate a scene of a futuristic metropolis at night",
        "Draw a group of penguins on an iceberg",
        "Create an image of a fairy-tale cottage in the woods",
        "Make a picture of a Viking ship on stormy seas",
        "Generate art of a little girl reading under a giant mushroom",
        # professional / product
        "Design a banner image for my tech startup website",
        "Create a product mockup image for a water bottle",
        "Generate a social media header featuring tropical flowers",
        "Make a thumbnail image for a cooking YouTube video",
        "Design a pattern that can be used as a phone wallpaper",
        "Generate a marketing image for a fitness app",
        "Create a cover photo for a photography portfolio",
        "Make an event poster for a music festival",
        "Generate a background image for a meditation app",
        "Design a business card template for a freelance designer",
        # short / imperative
        "Image of a cyberpunk cat hacker",
        "Picture of cherry blossoms in Tokyo",
        "Draw a map of a fictional island",
        "Generate a space battle scene",
        "Create art of a mermaid underwater city",
        "Render a snowy mountain cabin at night",
        "Make an image of a glowing portal in a dark cave",
        "Generate abstract art using warm colours",
        "Draw a robot playing chess",
        "Create a panoramic view of the Grand Canyon",
        # conversational
        "Can you generate an image for me of a sunflower field?",
        "I need you to draw a wolf howling at the moon",
        "Could you make me a picture of a futuristic spaceship interior?",
        "Please create an image of a child flying a kite",
        "Would you be able to draw a comic-book style explosion?",
        "Help me visualise what a black hole looks like up close",
        "I'd love to see an image of a neon-lit ramen shop at night",
        "Make me a digital painting of a wise old oak tree",
        "Can you sketch out a floor plan diagram for me?",
        "I was hoping you could generate artwork of an ancient temple",
        # reimagine / edit framing
        "Generate a photo of a car but make it look vintage",
        "Create an image similar to the Mona Lisa but in a sci-fi setting",
        "Draw a reimagined Statue of Liberty on another planet",
        "Make a realistic render of my logo idea: a crescent moon with stars",
        "Generate an artistic interpretation of chaos theory",
        "Draw the solar system as a subway map",
        "Make a detailed technical illustration of a jet engine",
        "Generate a fashion sketch of a futuristic outfit",
        "Create an image of a clock melting like in surrealist art",
        "Design an avatar for my gaming profile",
        # diverse phrasings real users type
        "Turn this description into a piece of art",
        "Visualize this concept as a picture",
        "Make a meme about a programmer's daily life",
        "Create a thumbnail for my YouTube cooking video",
        "Render this scene as an oil painting",
        "Design a Twitch stream banner with fire and lightning",
        "Generate a sticker of a cute cartoon bear",
        "Illustrate this children's book page for me",
        "Make a graphic I can post on Instagram",
        "Turn my idea into pixel art",
        "Generate an album cover for my indie band",
        "Make a logo from just the letter A",
        "Render a scene from my fantasy novel",
        "Create a custom emoji of a laughing bear",
        "Illustrate the concept of entropy for a science poster",
        "Generate an architectural rendering of this house design",
        "Make a gif-style image of a bouncing ball",
        "Create a poster promoting recycling for schools",
        "Design a t-shirt graphic with a wolf and mountains",
        "Generate a profile picture that looks like a superhero",
    ],

    "generate_music": [
        # explicit generate / compose
        "Compose a relaxing piano melody for me",
        "Generate a lofi hip-hop beat I can study to",
        "Create some ambient background music with nature sounds",
        "Write a jingle for my bakery",
        "Produce an upbeat pop track for a fitness video",
        "Make a dramatic orchestral score for an action scene",
        "Generate a blues guitar riff in the key of E",
        "Compose a lullaby for a baby",
        "Create a jazz improvisation over a minor chord progression",
        "Make epic trailer music with big drums and brass",
        # genre / style
        "Generate some reggaeton with tropical vibes",
        "Compose classical music in the style of Beethoven",
        "Create a techno track with a hard-hitting bassline",
        "Make country music about a road trip",
        "Generate a samba rhythm for a Brazilian carnival",
        "Compose a Celtic folk melody with fiddle and flute",
        "Create an R&B groove with smooth vocals",
        "Generate death-metal riffs with fast double-kick drums",
        "Make an 80s synthwave track dripping with nostalgia",
        "Compose Indian classical music for a meditation session",
        # mood / purpose
        "I need background music for my podcast intro",
        "Create something calming for a yoga class",
        "Generate energetic music for a sports highlight reel",
        "Make spooky Halloween music with eerie strings",
        "Compose romantic music for a wedding scene",
        "Generate sad background music for a dramatic film moment",
        "Create happy, upbeat music for a children's show",
        "Make suspenseful music for a thriller scene",
        "Generate music that feels like flying through clouds",
        "Compose something that sounds like an ancient ritual",
        # song-writing
        "Write a catchy chorus for a pop song about summer",
        "Create a verse melody for a love song",
        "Generate chord progressions for a chill indie song",
        "Compose a bridge section that builds emotional tension",
        "Make a hook for a hip-hop track about perseverance",
        "Write a drum pattern for a funk song",
        "Generate a bass line that complements a jazz standard",
        "Create a guitar solo in a blues pentatonic scale",
        "Compose a string arrangement for an orchestral piece",
        "Make a synth arpeggio for a sci-fi soundtrack",
        # casual / conversational
        "Can you make me a beat?",
        "I want some music to work out to",
        "Could you create a short melody for my app notification sound?",
        "Help me come up with a musical theme for my YouTube channel",
        "I'd love some lo-fi music to chill and study",
        "Generate some music that matches a rainy-day mood",
        "Make music that feels underwater and dreamlike",
        "Can you compose something short and sweet?",
        "I need a quick musical stinger for a video transition",
        "Write me music in the style of Hans Zimmer",
        # technical / instrument
        "Generate a fingerpicking guitar pattern in D major",
        "Create a four-on-the-floor kick drum pattern at 128 BPM",
        "Compose a trumpet solo for a jazz big band arrangement",
        "Make a minimalist piano piece using only three notes",
        "Generate an electric guitar power-chord riff",
        "Compose a string quartet movement in sonata form",
        "Create a percussion loop using only African rhythms",
        "Generate a vocal melody over a G-C-D-Em progression",
        "Make a 16-bar melody for a children's nursery rhyme",
        "Compose a theme-and-variation for solo violin",
        # short / imperative
        "Make a beat for me",
        "Write me a song",
        "Compose some music",
        "Generate an audio track",
        "Create background music for my video",
        "Make me a jingle",
        "Produce a melody",
        "Generate some tunes",
        "Create an original song",
        "Write me a musical piece",
        # soundscape / audio
        "Generate sound effects for a forest scene",
        "Create ASMR audio with gentle rain sounds",
        "Make a soundscape of a busy city street",
        "Generate white noise to help me sleep",
        "Create ambient audio that sounds like outer space",
        "Make relaxing ocean wave sounds",
        "Generate a binaural beat for deep focus",
        "Compose an audio logo for my brand",
        "Create music that gradually builds in energy",
        "Generate a cheerful ringtone melody",
        # diverse phrasings real users type
        "Make a soundtrack for my short film",
        "Create background tracks for my podcast episode",
        "Turn this poem into a song please",
        "Produce a hip-hop instrumental for me",
        "Generate an acoustic version of a pop song vibe",
        "Make some dance music I can post on TikTok",
        "Compose a theme song for my YouTube brand",
        "Make a retro 8-bit chiptune for my game",
        "Create music for a documentary about the ocean",
        "Generate experimental electronic music with glitchy sounds",
        "Produce a workout playlist vibe track",
        "Make elevator music for my waiting room",
        "Create an intro jingle under 5 seconds long",
        "Make music that sounds like a fantasy RPG",
        "Compose a waltz for a dance class",
        "Generate a baroque harpsichord piece",
        "Make a meditation bell soundscape",
        "Create a remix of this chord progression",
        "Generate a lo-fi beat with vinyl crackle",
        "Compose something in 3/4 time for a ballet class",
    ],

    "read_aloud": [
        # direct TTS
        "Please read this text out loud",
        "Can you say this paragraph for me?",
        "Read the following sentence aloud",
        "Convert this text to speech",
        "Speak this message out loud",
        "Narrate the passage below for me",
        "Vocalize this sentence please",
        "Read out loud what I'm about to type",
        "Turn this text into audio",
        "Say these words for me",
        # content narration
        "Read aloud: The quick brown fox jumps over the lazy dog",
        "Please narrate this article introduction",
        "Can you say the following in a natural voice?",
        "Read out my grocery list for me",
        "Speak the contents of this email aloud",
        "Please read my notes back to me",
        "Narrate this recipe step by step",
        "Read out the instructions below",
        "Say this quote out loud for me",
        "Please vocalise this announcement",
        # accessibility / utility
        "I need you to read this document aloud for me",
        "Can you narrate this text so I don't have to read it?",
        "Read this while I'm driving so I can keep my eyes on the road",
        "Say the title and author of this book out loud",
        "Read the news headline aloud",
        "Please read my message before I send it",
        "Say this caption out loud so I can hear it",
        "I want to hear this text spoken, not read it silently",
        "Read the description out loud so I can listen",
        "Can you speak the subtitles for me?",
        # voice / style
        "Read this with a calm, soothing voice",
        "Narrate this story in a dramatic tone",
        "Say this text in a cheerful manner",
        "Read this slowly and clearly please",
        "Speak this text in a British accent style",
        "Narrate this in the style of a news anchor",
        "Read this like you're telling a bedtime story",
        "Say this in a friendly, conversational tone",
        "Read this passage with emotional expression",
        "Narrate this as if you're a documentary narrator",
        # classroom / learning
        "Read aloud the vocabulary words for me",
        "Say each word slowly so I can learn the pronunciation",
        "Read this foreign phrase out loud",
        "Pronounce this word for me",
        "Read the phonetic transcription aloud",
        "Say this sentence so I can check my understanding",
        "Read the definition of this term aloud",
        "Narrate the lesson content for me",
        "Please read this poem line by line",
        "Say this tongue twister out loud",
        # short / imperative
        "Read that back to me",
        "Say it out loud",
        "Please narrate this",
        "Speak this text",
        "Read it aloud",
        "Vocalise this for me",
        "Turn this into audio speech",
        "Read the following passage",
        "Narrate this for me",
        "Say these words aloud",
        # conversational
        "Could you read this paragraph back to me?",
        "I'd like to hear this text spoken aloud",
        "Would you mind narrating this for me?",
        "Can you help me listen to this text?",
        "I need this read out loud — can you do that?",
        "Is it possible for you to speak this text?",
        "I want to hear this content read aloud, not silently",
        "Please narrate the following text for me",
        "Help me by reading this passage out loud",
        "Can you convert the following into spoken words?",
        # specific content
        "Read this bedtime story to me",
        "Narrate the chapter I just wrote",
        "Read the warning label aloud",
        "Say this address clearly for me",
        "Read aloud the product description",
        "Narrate this tutorial step by step",
        "Read the feedback comments out loud",
        "Say this thank-you note aloud for me",
        "Please read the terms and conditions out loud",
        "Narrate this slide content for a visually impaired user",
        # diverse real-user phrasings
        "Play this text back to me",
        "Use TTS to read this paragraph",
        "Speak these cooking instructions aloud while I work",
        "Can you use your voice to say this for me?",
        "Voice this message for me",
        "Give this text a voice",
        "Dictate this back to me out loud",
        "Read this chapter to me while I commute",
        "Voice-over this script for me",
        "Please use speech synthesis on this block of text",
        "I need this text to be spoken aloud",
        "Read this form field aloud so I can verify it",
        "Narrate the steps in this tutorial",
        "Can you voice act this dialogue for me?",
        "Speak this error message so I know what went wrong",
        "Read the notification message aloud",
        "Make this text talk so I can listen hands-free",
        "Audio output for the following text please",
        "Speak the subtitle text out loud",
        "Read this recipe to me while I cook",
    ],

    "chat_with_llm": [
        # general knowledge
        "What is the capital of Australia?",
        "How does photosynthesis work?",
        "Explain the theory of relativity in simple terms",
        "Who was Napoleon Bonaparte?",
        "What causes earthquakes?",
        "Tell me about the history of the Roman Empire",
        "How do vaccines work?",
        "What is machine learning?",
        "Why is the sky blue?",
        "Explain how black holes are formed",
        # creative writing
        "Write me a short story about a time traveller",
        "Help me draft a poem about loneliness",
        "Continue this story: She opened the door and gasped",
        "Write a haiku about winter",
        "Create a plot summary for a sci-fi novel",
        "Write a limerick about a clumsy wizard",
        "Draft the opening paragraph of a mystery novel",
        "Help me brainstorm characters for my fantasy story",
        "Write a funny anecdote about a talking cat",
        "Create a short script for a comedy sketch",
        # advice / recommendations
        "What are some good habits for better sleep?",
        "How should I start learning Python?",
        "Give me tips for improving my public speaking",
        "What is the best way to budget my monthly expenses?",
        "How can I be more productive working from home?",
        "Recommend some books on philosophy",
        "What are the pros and cons of intermittent fasting?",
        "Give me advice on dealing with procrastination",
        "What exercises are best for building core strength?",
        "How do I negotiate a salary raise?",
        # explanations
        "Explain blockchain technology like I am five years old",
        "What is the difference between machine learning and AI?",
        "How does a jet engine work?",
        "Explain what DNA is and what it does",
        "What is the difference between weather and climate?",
        "How does the stock market work?",
        "Explain the concept of supply and demand",
        "What is quantum computing?",
        "How does the internet actually work?",
        "Explain neural networks in plain English",
        # task assistance
        "Summarise this text for me",
        "Help me write a cover letter for a software engineering job",
        "Translate hello how are you into French",
        "Proofread my essay introduction",
        "Help me come up with a business name",
        "Write a professional email declining a meeting",
        "Create a to-do list for planning a birthday party",
        "Help me outline a presentation on climate change",
        "Draft a LinkedIn bio for a graphic designer",
        "Write a product description for a smart water bottle",
        # conversation / small talk
        "Tell me a fun fact",
        "Do you think AI will replace human jobs?",
        "What is the meaning of life?",
        "Tell me a joke",
        "What do you think about space exploration?",
        "What is interesting about octopuses?",
        "Talk to me about philosophy",
        "What makes a good leader?",
        "What would you do if you were human for a day?",
        "What is your opinion on climate change?",
        # maths / logic
        "Solve this equation: 3x plus 5 equals 20",
        "What is 15 percent of 240?",
        "Help me understand compound interest",
        "Explain the Pythagorean theorem",
        "What is the factorial of 10?",
        "Solve this riddle: I have cities but no houses",
        "How do you calculate the area of a circle?",
        "Help me with this probability problem",
        "What is the square root of 144?",
        "Explain the Fibonacci sequence",
        # coding / technical
        "How do I reverse a string in Python?",
        "Explain what a REST API is",
        "What is the difference between SQL and NoSQL?",
        "Write a function that checks if a number is prime",
        "How does recursion work?",
        "What is version control and why do I need it?",
        "Help me debug this JavaScript code",
        "What is object-oriented programming?",
        "Explain what a Docker container is",
        "Write a regex pattern to match email addresses",
        # diverse real-user phrasings
        "Summarize the main points of this article for me",
        "Who won the FIFA World Cup in 2022?",
        "Help me plan a trip to Japan for two weeks",
        "What are good interview questions for a senior developer?",
        "Explain the difference between TCP and UDP protocols",
        "Give me a recipe for chocolate lava cake",
        "What does the word ephemeral mean?",
        "Help me respond to this angry customer email professionally",
        "What are the symptoms of dehydration?",
        "Why did the Roman Empire fall?",
        "Write a SQL query to find duplicate rows in a table",
        "What are best practices for password security?",
        "Summarize the plot of Hamlet for me",
        "Give me 5 synonyms for the word happy",
        "Help me name my new startup",
        "What is the best programming language for beginners?",
        "Explain how vaccines create immunity in the body",
        "Write a thank you letter for after a job interview",
        "What are the main differences between Python 2 and 3?",
        "Help me calculate compound interest on a loan",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 3 ─ HARD NEGATIVES
#
# These are confusable examples that explicitly teach class boundaries.
# The three failure modes they target:
#   • "Tell me HOW X is generated"  → sounds like X-class, is chat_with_llm
#   • "Explain X technology"        → sounds like X-class, is chat_with_llm
#   • "Give me IDEAS for X"         → sounds like X-class, is chat_with_llm
#
# All hard negatives here are chat_with_llm.  They force the model to learn
# that requesting INFORMATION about a medium ≠ requesting PRODUCTION of it.
# ══════════════════════════════════════════════════════════════════════════════

HARD_NEGATIVES: dict[str, list[str]] = {

    # Hard negatives that look like generate_music
    "chat_with_llm": [
        # ── confusable with generate_music ───────────────────────────────────
        "Tell me how AI generates music",
        "Explain how music synthesis works",
        "How do AI music generators actually work?",
        "What is the best AI model for generating music?",
        "Tell me about algorithmic music composition",
        "How does Suno AI create songs from text?",
        "What technology powers AI music generation tools?",
        "Explain the difference between MIDI and audio synthesis",
        "What makes AI-generated music sound realistic?",
        "Compare different AI music generation approaches",
        "Tell me about the history of electronic and generative music",
        "What is generative music theory?",
        "How does music theory apply to AI composition systems?",
        "What are the ethical concerns around AI-generated music?",
        "Which AI music tools are worth paying for?",
        "What is the difference between Suno and Udio?",
        "How do neural networks learn to compose music?",
        "Summarise the research on AI music generation",
        "Tell me what MIDI is and how it works",
        "What music genre should I use as background for my video?",
        "Give me ideas for what kind of music would suit my brand",
        "Help me think about what audio style fits a horror game",
        "Suggest some music genres I might enjoy based on my taste",
        "What mood does orchestral music typically create?",
        "What is the tempo of most lofi beats?",
        # ── confusable with read_aloud ────────────────────────────────────────
        "Explain how text-to-speech systems work",
        "Tell me about speech synthesis technology",
        "How does Amazon Polly generate voice audio?",
        "What is the best TTS model available right now?",
        "Compare different voice synthesis engines",
        "How do neural TTS systems differ from concatenative ones?",
        "What makes a text-to-speech voice sound natural?",
        "Tell me about WaveNet and how it changed speech synthesis",
        "Explain the history of text-to-speech research",
        "What is the difference between TTS and voice cloning?",
        "How do audiobooks get professionally narrated?",
        "What makes a good narrator for long-form content?",
        "Tell me about voice acting as a career",
        "What is the difference between a voiceover and narration?",
        "How does lip-sync animation use audio data?",
        "Explain how screen readers work for the visually impaired",
        "What is the best software for recording voiceovers?",
        "Tell me about the accessibility benefits of TTS technology",
        "How do podcasters add narration to their shows?",
        "What is phoneme synthesis in speech systems?",
        # ── confusable with generate_image ────────────────────────────────────
        "Tell me how AI generates images from text",
        "Explain how stable diffusion models work",
        "How does DALL-E 3 create images from a prompt?",
        "What is the best image generation model available?",
        "Compare Midjourney versus Stable Diffusion",
        "How do I write a good prompt for image generation?",
        "Explain what a GAN is and how it generates images",
        "What are the ethical concerns with AI image generation?",
        "Tell me about the history of computational art and AI",
        "How does inpainting work in image editing tools?",
        "What is ControlNet and how does it guide image generation?",
        "Explain the difference between diffusion and GAN models",
        "What resolution should AI-generated images be?",
        "Tell me about copyright issues with AI-generated art",
        "How do I evaluate the quality of an AI-generated image?",
        "Can you help me brainstorm ideas for what to put in an image?",
        "Give me some ideas for a visual concept I am working on",
        "Help me think about what style would suit my album cover art",
        "What visual elements make a good logo design?",
        "What color palette works well for a nature-themed brand?",
        "Describe the principles of good visual composition",
        "What makes a thumbnail eye-catching on YouTube?",
        "How should I compose a portrait photograph?",
        "What is the rule of thirds in photography?",
        "Tell me what vector graphics are and how they differ from raster",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 3b ─ DATA AUGMENTATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Typo augmentation ─────────────────────────────────────────────────────────
# QWERTY keyboard adjacency for realistic character substitutions
_KEYBOARD: dict[str, str] = {
    'q': 'wa',   'w': 'qase', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg',
    'y': 'tugh', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhvb', 'h': 'gyujbn', 'j': 'huikn', 'k': 'jilom', 'l': 'kop',
    'z': 'asx',  'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
    'n': 'bhjm', 'm': 'njk',
}


def _one_typo(text: str) -> str:
    """Introduce one realistic typo: keyboard sub | delete | transpose | dup."""
    if len(text) < 10:
        return text
    chars = list(text)
    alpha = [i for i, c in enumerate(chars) if c.isalpha() and i > 0]
    if not alpha:
        return text
    pos = random.choice(alpha)
    c   = chars[pos].lower()
    op  = random.choice(['sub', 'del', 'trans', 'dup'])
    if   op == 'sub'   and c in _KEYBOARD:
        chars[pos] = random.choice(_KEYBOARD[c])
    elif op == 'del':
        del chars[pos]
    elif op == 'trans'  and pos < len(chars) - 1:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    elif op == 'dup':
        chars.insert(pos, chars[pos])
    return ''.join(chars)


def augment_with_typos(rows: list[dict], rate: float = 0.20) -> list[dict]:
    """
    For `rate` fraction of rows, append a noisy copy with one typo.
    Original rows are always preserved; this only adds extras.
    Applied to TRAINING data only — never val/test.
    """
    extras = []
    for row in rows:
        if random.random() < rate:
            noisy = _one_typo(row["text"])
            if noisy != row["text"]:
                extras.append({"text": noisy, "label": row["label"]})
    return rows + extras


# ── Synthetic paraphrases for generate_image ─────────────────────────────────
# generate_image gets zero examples from any dataset, so we manufacture
# paraphrases by swapping the leading creation verb in each hardcoded example.
# This gives genuine lexical variety without semantic drift.

_IMAGE_VERB_SWAPS: dict[str, list[str]] = {
    "create":     ["generate", "make", "produce", "build"],
    "generate":   ["create",   "make", "produce", "design"],
    "draw":       ["sketch",   "illustrate", "paint", "render"],
    "make":       ["create",   "generate", "produce", "design"],
    "paint":      ["draw",     "illustrate", "render", "create"],
    "render":     ["generate", "create", "produce", "make"],
    "design":     ["create",   "make",   "generate", "produce"],
    "produce":    ["create",   "generate", "make",   "design"],
    "illustrate": ["draw",     "sketch",   "depict", "render"],
    "sketch":     ["draw",     "illustrate", "render", "create"],
}


def _image_paraphrases(examples: list[str], max_alts: int = 2) -> list[str]:
    """
    For each example, swap the leading creation verb with up to `max_alts`
    alternatives.  Returns only the NEW paraphrases (not the originals).
    """
    results: set[str] = set()
    for text in examples:
        lower = text.lower()
        for verb, alts in _IMAGE_VERB_SWAPS.items():
            if lower.startswith(verb + " "):
                suffix = text[len(verb):]          # keep original casing of rest
                for alt in alts[:max_alts]:
                    para = alt.capitalize() + suffix
                    if para != text:
                        results.add(para)
                break
    return list(results)


# ══════════════════════════════════════════════════════════════════════════════
# 4 ─ EXPLICIT INTENT MAPS FOR ALL DATASETS
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. CLINC_OOS (151 intents) ──────────────────────────────────────────────
CLINC_INTENT_MAP: dict[str, str | None] = {
    # generate_music
    "play_music":           "generate_music",
    "next_song":            "generate_music",
    "update_playlist":      "generate_music",
    "music_likability":     "generate_music",
    "music_settings":       "generate_music",
    "what_song":            "generate_music",
    # read_aloud (voice-output intents only)
    "repeat":               "read_aloud",
    "whisper_mode":         "read_aloud",
    "change_volume":        "read_aloud",
    "change_language":      "read_aloud",
    "change_accent":        "read_aloud",
    # chat_with_llm – utility / device
    "text":                 "chat_with_llm",
    "make_call":            "chat_with_llm",
    "alarm":                "chat_with_llm",
    "timer":                "chat_with_llm",
    "reminder":             "chat_with_llm",
    "reminder_update":      "chat_with_llm",
    "calculator":           "chat_with_llm",
    "measurement_conversion": "chat_with_llm",
    "flip_coin":            "chat_with_llm",
    "roll_dice":            "chat_with_llm",
    "spelling":             "chat_with_llm",
    "time":                 "chat_with_llm",
    "weather":              "chat_with_llm",
    "definition":           "chat_with_llm",
    "basic_math":           "chat_with_llm",
    "smart_home":           "chat_with_llm",
    "todo_list":            "chat_with_llm",
    "todo_list_update":     "chat_with_llm",
    "share_location":       "chat_with_llm",
    "directions":           "chat_with_llm",
    "traffic":              "chat_with_llm",
    "find_phone":           "chat_with_llm",
    # travel
    "book_flight":          "chat_with_llm",
    "book_hotel":           "chat_with_llm",
    "car_rental":           "chat_with_llm",
    "travel_alert":         "chat_with_llm",
    "travel_suggestion":    "chat_with_llm",
    "travel_notification":  "chat_with_llm",
    "carry_on":             "chat_with_llm",
    "timezone":             "chat_with_llm",
    "international_visa":   "chat_with_llm",
    "plug_type":            "chat_with_llm",
    "flight_status":        "chat_with_llm",
    "international_fees":   "chat_with_llm",
    "lost_luggage":         "chat_with_llm",
    "exchange_rate":        "chat_with_llm",
    "translate":            "chat_with_llm",
    # dining / food
    "restaurant_reviews":       "chat_with_llm",
    "restaurant_suggestion":    "chat_with_llm",
    "restaurant_reservation":   "chat_with_llm",
    "accept_reservations":      "chat_with_llm",
    "confirm_reservation":      "chat_with_llm",
    "recipe":                   "chat_with_llm",
    "nutrition_info":           "chat_with_llm",
    "food_last":                "chat_with_llm",
    "ingredient_substitution":  "chat_with_llm",
    "meal_suggestion":          "chat_with_llm",
    "calories":                 "chat_with_llm",
    # auto / commute
    "gas":                  "chat_with_llm",
    "gas_type":             "chat_with_llm",
    "distance":             "chat_with_llm",
    "mpg":                  "chat_with_llm",
    "current_location":     "chat_with_llm",
    "oil_change_when":      "chat_with_llm",
    "oil_change_how":       "chat_with_llm",
    "jump_start":           "chat_with_llm",
    "uber_lyft":            "chat_with_llm",
    "schedule_maintenance": "chat_with_llm",
    "last_maintenance":     "chat_with_llm",
    # banking / finance
    "freeze_account":       "chat_with_llm",
    "routing":              "chat_with_llm",
    "pin_change":           "chat_with_llm",
    "bill_balance":         "chat_with_llm",
    "bill_due":             "chat_with_llm",
    "pay_bill":             "chat_with_llm",
    "transfer":             "chat_with_llm",
    "transactions":         "chat_with_llm",
    "balance":              "chat_with_llm",
    "credit_limit":         "chat_with_llm",
    "credit_score":         "chat_with_llm",
    "account_blocked":      "chat_with_llm",
    "interest_rate":        "chat_with_llm",
    "min_payment":          "chat_with_llm",
    "report_fraud":         "chat_with_llm",
    "spending_history":     "chat_with_llm",
    "credit_limit_change":  "chat_with_llm",
    "report_lost_card":     "chat_with_llm",
    "apr":                  "chat_with_llm",
    "redeem_rewards":       "chat_with_llm",
    "rewards_balance":      "chat_with_llm",
    "direct_deposit":       "chat_with_llm",
    "improve_credit_score": "chat_with_llm",
    "taxes":                "chat_with_llm",
    "income":               "chat_with_llm",
    "w2":                   "chat_with_llm",
    "rollover_401k":        "chat_with_llm",
    # work / calendar
    "calendar":             "chat_with_llm",
    "calendar_update":      "chat_with_llm",
    "schedule_meeting":     "chat_with_llm",
    "meeting_schedule":     "chat_with_llm",
    "pto_request":          "chat_with_llm",
    "pto_balance":          "chat_with_llm",
    "pto_request_status":   "chat_with_llm",
    "next_holiday":         "chat_with_llm",
    "insurance":            "chat_with_llm",
    "insurance_change":     "chat_with_llm",
    # shopping / home
    "shopping_list":        "chat_with_llm",
    "shopping_list_update": "chat_with_llm",
    "order":                "chat_with_llm",
    "order_status":         "chat_with_llm",
    "order_checks_up":      "chat_with_llm",
    "return_item":          "chat_with_llm",
    # small talk / meta
    "meaning_of_life":      "chat_with_llm",
    "tell_joke":            "chat_with_llm",
    "do_you_have_pets":     "chat_with_llm",
    "are_you_a_bot":        "chat_with_llm",
    "what_is_your_name":    "chat_with_llm",
    "when_is_your_birthday":"chat_with_llm",
    "tell_me_a_story":      "chat_with_llm",
    "are_you_human":        "chat_with_llm",
    "thank_you":            "chat_with_llm",
    "goodbye":              "chat_with_llm",
    "good_morning":         "chat_with_llm",
    "good_afternoon":       "chat_with_llm",
    "good_evening":         "chat_with_llm",
    "good_night":           "chat_with_llm",
    "what_are_your_hobbies":"chat_with_llm",
    "what_are_your_interests": "chat_with_llm",
    "who_made_you":         "chat_with_llm",
    "change_user_name":     "chat_with_llm",
    "user_name":            "chat_with_llm",
    "where_are_you_from":   "chat_with_llm",
    "yes":                  "chat_with_llm",
    "no":                   "chat_with_llm",
    "maybe":                "chat_with_llm",
    "cancel":               "chat_with_llm",
    "sync_device":          "chat_with_llm",
    "reset_settings":       "chat_with_llm",
    "change_ai_name":       "chat_with_llm",
    "change_speed":         "chat_with_llm",
    # out-of-scope → discard
    "oos":                  None,
}
_CLINC_DEFAULT = "chat_with_llm"

# ── 3b. SNIPS (7 intents) ────────────────────────────────────────────────────
SNIPS_INTENT_MAP: dict[str, str] = {
    "AddToPlaylist":        "generate_music",
    "BookRestaurant":       "chat_with_llm",
    "GetWeather":           "chat_with_llm",
    "PlayMusic":            "generate_music",
    "RateBook":             "chat_with_llm",
    "SearchCreativeWork":   "chat_with_llm",
    "SearchScreeningEvent": "chat_with_llm",
}
_SNIPS_DEFAULT = "chat_with_llm"

# ── 3c. ATIS (all intents are flight/travel queries) ─────────────────────────
# Every atis_* intent is a question someone would ask about flights or airports.
# None map to generate_image, generate_music, or read_aloud.
_ATIS_DEFAULT = "chat_with_llm"

# ── 3d. MASSIVE (60 intents across 18 domains) ───────────────────────────────
MASSIVE_INTENT_MAP: dict[str, str] = {
    # generate_music
    "music_dislikeness":    "generate_music",
    "music_likeness":       "generate_music",
    "music_query":          "generate_music",
    "music_settings":       "generate_music",
    "play_music":           "generate_music",
    "play_radio":           "generate_music",
    # read_aloud
    "audio_volume_down":    "read_aloud",
    "audio_volume_mute":    "read_aloud",
    "audio_volume_other":   "read_aloud",
    "audio_volume_up":      "read_aloud",
    "play_audiobook":       "read_aloud",
    "general_repeat":       "read_aloud",
    # chat_with_llm (all remaining)
    "alarm_query":              "chat_with_llm",
    "alarm_remove":             "chat_with_llm",
    "alarm_set":                "chat_with_llm",
    "calendar_query":           "chat_with_llm",
    "calendar_remove":          "chat_with_llm",
    "calendar_set":             "chat_with_llm",
    "cooking_recipe":           "chat_with_llm",
    "cooking_query":            "chat_with_llm",
    "datetime_convert":         "chat_with_llm",
    "datetime_query":           "chat_with_llm",
    "email_addcontact":         "chat_with_llm",
    "email_query":              "chat_with_llm",
    "email_querycontact":       "chat_with_llm",
    "email_sendemail":          "chat_with_llm",
    "general_affirm":           "chat_with_llm",
    "general_commandstop":      "chat_with_llm",
    "general_confirm":          "chat_with_llm",
    "general_dontcare":         "chat_with_llm",
    "general_explain":          "chat_with_llm",
    "general_greet":            "chat_with_llm",
    "general_joke":             "chat_with_llm",
    "general_negate":           "chat_with_llm",
    "general_quirky":           "chat_with_llm",
    "iot_cleaning":             "chat_with_llm",
    "iot_coffee":               "chat_with_llm",
    "iot_hue_lightchange":      "chat_with_llm",
    "iot_hue_lightdim":         "chat_with_llm",
    "iot_hue_lightoff":         "chat_with_llm",
    "iot_hue_lighton":          "chat_with_llm",
    "iot_hue_lightup":          "chat_with_llm",
    "iot_wemo_off":             "chat_with_llm",
    "iot_wemo_on":              "chat_with_llm",
    "lists_createoradd":        "chat_with_llm",
    "lists_query":              "chat_with_llm",
    "lists_remove":             "chat_with_llm",
    "news_query":               "chat_with_llm",
    "play_game":                "chat_with_llm",
    "play_podcasts":            "chat_with_llm",
    "qa_currency":              "chat_with_llm",
    "qa_definition":            "chat_with_llm",
    "qa_factoid":               "chat_with_llm",
    "qa_maths":                 "chat_with_llm",
    "qa_stock":                 "chat_with_llm",
    "recommendation_events":    "chat_with_llm",
    "recommendation_locations": "chat_with_llm",
    "recommendation_movies":    "chat_with_llm",
    "social_post":              "chat_with_llm",
    "social_query":             "chat_with_llm",
    "takeaway_order":           "chat_with_llm",
    "takeaway_query":           "chat_with_llm",
    "transport_query":          "chat_with_llm",
    "transport_taxi":           "chat_with_llm",
    "transport_ticket":         "chat_with_llm",
    "transport_traffic":        "chat_with_llm",
    "weather_query":            "chat_with_llm",
}
_MASSIVE_DEFAULT = "chat_with_llm"


# ══════════════════════════════════════════════════════════════════════════════
# 4 ─ DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _flatten(ds) -> Dataset:
    if isinstance(ds, DatasetDict):
        return concatenate_datasets([ds[s] for s in ds])
    return ds


def _resolve_intent_str(raw, feature) -> str:
    """Convert a raw label value to its string intent name."""
    if hasattr(feature, "names"):          # ClassLabel
        return feature.names[int(raw)]
    return str(raw)


def load_hardcoded_data() -> list[dict]:
    """
    Returns curated examples + hard negatives + synthetic image paraphrases.

    Hard negatives teach the boundary between requesting production of media
    vs asking for information about that media type.

    Image paraphrases compensate for generate_image having zero dataset
    coverage by generating verb-swapped copies of the 100 curated examples.
    """
    rows: list[dict] = []

    # Original curated examples
    for lbl, texts in HARDCODED.items():
        for text in texts:
            rows.append({"text": text.strip(), "label": LABEL2ID[lbl]})

    # Hard negatives (confusable cross-boundary examples)
    for lbl, texts in HARD_NEGATIVES.items():
        for text in texts:
            rows.append({"text": text.strip(), "label": LABEL2ID[lbl]})

    # Synthetic paraphrases for generate_image (verb-swap augmentation)
    img_id    = LABEL2ID["generate_image"]
    img_paras = _image_paraphrases(HARDCODED["generate_image"])
    for text in img_paras:
        rows.append({"text": text.strip(), "label": img_id})

    print(f"[INFO] hardcoded: {len(rows):,} rows  "
          f"({len(img_paras)} image paraphrases, "
          f"{sum(len(v) for v in HARD_NEGATIVES.values())} hard negatives)")
    return rows


def load_clinc_data(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] clinc_oos not found at {p} – skipping.")
        return []
    try:
        ds = _flatten(load_from_disk(str(p)))
        if "intent" not in ds.features:
            print("[WARN] clinc_oos: no 'intent' column – skipping.")
            return []
        feat = ds.features["intent"]
        rows, skipped = [], 0
        for ex in ds:
            intent_str = _resolve_intent_str(ex["intent"], feat)
            mapped = CLINC_INTENT_MAP.get(intent_str, _CLINC_DEFAULT)
            if mapped is None:
                skipped += 1
                continue
            rows.append({"text": ex["text"].strip(), "label": LABEL2ID[mapped]})
        print(f"[INFO] clinc_oos: {len(rows):,} loaded, {skipped} oos skipped.")
        return rows
    except Exception as e:
        print(f"[WARN] clinc_oos failed: {e} – skipping.")
        return []


def load_banking_data(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] banking77 not found at {p} – skipping.")
        return []
    try:
        ds = _flatten(load_from_disk(str(p)))
        if "text" not in ds.features:
            print("[WARN] banking77: no 'text' column – skipping.")
            return []
        chat_id = LABEL2ID["chat_with_llm"]
        rows = [{"text": ex["text"].strip(), "label": chat_id} for ex in ds]
        print(f"[INFO] banking77: {len(rows):,} loaded.")
        return rows
    except Exception as e:
        print(f"[WARN] banking77 failed: {e} – skipping.")
        return []


def _generic_load(
    path: str,
    name: str,
    intent_map: dict[str, str],
    default_label: str,
    text_keys: list[str],
    label_keys: list[str],
) -> list[dict]:
    """
    Reusable loader for SNIPS / MASSIVE.
    Prints the actual column names when 0 rows load so the user can debug.
    """
    p = Path(path)
    if not p.exists():
        print(f"[INFO] {name} not found at {p} – skipping (optional).")
        return []
    try:
        ds = _flatten(load_from_disk(str(p)))
        cols = list(ds.features.keys())

        # Warn immediately if expected columns are absent
        found_text  = any(k in cols for k in text_keys)
        found_label = any(k in cols for k in label_keys)
        if not found_text or not found_label:
            print(f"[WARN] {name}: expected text={text_keys} label={label_keys} "
                  f"but dataset has columns {cols}  – skipping.")
            return []

        rows = []
        for ex in ds:
            text = next((ex.get(k) for k in text_keys if ex.get(k)), None)
            raw_label = next(
                (ex.get(k) for k in label_keys if ex.get(k) is not None), None)
            if text is None or raw_label is None:
                continue
            # Resolve ClassLabel → string
            feat = ds.features.get(
                next((k for k in label_keys if k in ds.features), None))
            if feat and hasattr(feat, "names"):
                intent_str = feat.names[int(raw_label)]
            else:
                intent_str = str(raw_label)
            mapped = intent_map.get(intent_str, default_label)
            rows.append({"text": str(text).strip(), "label": LABEL2ID[mapped]})
        print(f"[INFO] {name}: {len(rows):,} loaded.")
        return rows
    except Exception as e:
        print(f"[WARN] {name} failed: {e} – skipping.")
        return []


def load_snips_data(path: str) -> list[dict]:
    # snips_built_in_intents uses 'text' + 'label' (ClassLabel)
    # nlu_evaluation_data uses 'sentence' + 'label'
    # Try both; the diagnostic in _generic_load shows actual columns on failure.
    return _generic_load(
        path, "snips", SNIPS_INTENT_MAP, _SNIPS_DEFAULT,
        text_keys=["text", "sentence", "utt", "utterance"],
        label_keys=["label", "intent"],
    )


def load_atis_data(path: str) -> list[dict]:
    """
    ATIS (tuetschek/atis) stores text as a list under 'tokens', not a string.
    We join tokens with spaces. All ATIS intents → chat_with_llm.
    """
    p = Path(path)
    if not p.exists():
        print(f"[INFO] atis not found at {p} – skipping (optional).")
        return []
    try:
        ds = _flatten(load_from_disk(str(p)))
        cols = list(ds.features.keys())

        rows: list[dict] = []
        chat_id = LABEL2ID["chat_with_llm"]

        for ex in ds:
            # Handle 'tokens' (list) OR standard string text fields
            if "tokens" in cols and isinstance(ex.get("tokens"), list):
                text = " ".join(ex["tokens"]).strip()
            else:
                text = next(
                    (ex.get(k) for k in
                     ["text", "utterance", "sentence", "query"]
                     if ex.get(k)), None)
            if not text:
                continue
            rows.append({"text": text, "label": chat_id})

        if len(rows) == 0:
            print(f"[WARN] atis: 0 rows loaded. Dataset columns: {cols}")
        else:
            print(f"[INFO] atis: {len(rows):,} loaded.")
        return rows
    except Exception as e:
        print(f"[WARN] atis failed: {e} – skipping.")
        return []


def load_massive_data(path: str) -> list[dict]:
    return _generic_load(
        path, "massive", MASSIVE_INTENT_MAP, _MASSIVE_DEFAULT,
        text_keys=["utt", "text", "sentence"],
        label_keys=["intent", "label"],
    )


def load_diffusiondb_data(path: str) -> list[dict]:
    """
    Load the locally-saved DiffusionDB slice.

    The dataset was saved with string label "generate_image" (not an int),
    so we convert it explicitly.  Both raw prompts and template-wrapped
    prompts are included as-is; the variety is already baked in.
    """
    p = Path(path)
    if not p.exists():
        print(f"[INFO] diffusionDB not found at {p} – skipping (optional).")
        return []
    try:
        ds = _flatten(load_from_disk(str(p)))
        if "text" not in ds.features:
            print(f"[WARN] diffusionDB: no 'text' column. "
                  f"Columns present: {list(ds.features)} – skipping.")
            return []

        img_id = LABEL2ID["generate_image"]
        rows: list[dict] = []
        for ex in ds:
            text = ex.get("text", "").strip()
            if not text:
                continue
            rows.append({"text": text, "label": img_id})

        print(f"[INFO] diffusionDB: {len(rows):,} generate_image examples loaded.")
        return rows
    except Exception as e:
        print(f"[WARN] diffusionDB failed: {e} – skipping.")
        return []


def load_readaloud_data(path: str) -> list[dict]:
    """
    Load the locally-saved readAloud slice (CNN/DailyMail + Wikipedia).
    Label is the string "read_aloud"; we convert it to the integer id.
    """
    p = Path(path)
    if not p.exists():
        print(f"[INFO] readAloud not found at {p} – skipping (optional).")
        return []
    try:
        ds = _flatten(load_from_disk(str(p)))
        if "text" not in ds.features:
            print(f"[WARN] readAloud: no 'text' column. "
                  f"Columns present: {list(ds.features)} – skipping.")
            return []
        lbl_id = LABEL2ID["read_aloud"]
        rows   = [{"text": ex["text"].strip(), "label": lbl_id}
                  for ex in ds if ex.get("text", "").strip()]
        print(f"[INFO] readAloud: {len(rows):,} read_aloud examples loaded.")
        return rows
    except Exception as e:
        print(f"[WARN] readAloud failed: {e} – skipping.")
        return []


def load_musiccaps_data(path: str) -> list[dict]:
    """
    Load the locally-saved MusicCaps slice.
    Label is the string "generate_music"; we convert it to the integer id.
    """
    p = Path(path)
    if not p.exists():
        print(f"[INFO] musiccaps not found at {p} – skipping (optional).")
        return []
    try:
        ds = _flatten(load_from_disk(str(p)))
        if "text" not in ds.features:
            print(f"[WARN] musiccaps: no 'text' column. "
                  f"Columns present: {list(ds.features)} – skipping.")
            return []
        lbl_id = LABEL2ID["generate_music"]
        rows   = [{"text": ex["text"].strip(), "label": lbl_id}
                  for ex in ds if ex.get("text", "").strip()]
        print(f"[INFO] musiccaps: {len(rows):,} generate_music examples loaded.")
        return rows
    except Exception as e:
        print(f"[WARN] musiccaps failed: {e} – skipping.")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 5 ─ TOKEN LENGTH ANALYSIS  (determines MAX_LEN automatically)
# ══════════════════════════════════════════════════════════════════════════════

def compute_max_len(rows: list[dict], tokenizer, ceil: int = MAX_LEN_CEIL,
                    sample_n: int = 3000) -> int:
    sample = random.sample(rows, min(sample_n, len(rows)))
    lengths = [len(tokenizer.encode(r["text"], add_special_tokens=True))
               for r in sample]
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    chosen = min(p99 + 4, ceil)
    print(f"[INFO] Token lengths – P95={p95}  P99={p99}  "
          f"max_in_sample={max(lengths)}  → using MAX_LEN={chosen}")
    return chosen


# ══════════════════════════════════════════════════════════════════════════════
# 6 ─ BALANCING  (upsample minority, cap majority; weighted loss handles rest)
# ══════════════════════════════════════════════════════════════════════════════

def balance(
    rows: list[dict],
    min_floor: int = MIN_CLASS_FLOOR,
    chat_cap: int = MAX_CHAT_CAP,
) -> list[dict]:
    """
    • Cap chat_with_llm at chat_cap to prevent it dominating.
    • Upsample any class below min_floor by repeating (shuffled).
    • Leave generate_music and read_aloud untouched unless below min_floor.
    Weighted Focal Loss applied during training handles residual imbalance.
    """
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["label"]].append(r)

    out: list[dict] = []
    for lbl_id, lbl_rows in buckets.items():
        random.shuffle(lbl_rows)
        # Cap dominant class
        lbl_name = ID2LABEL.get(lbl_id, "")
        if lbl_name == "chat_with_llm" and len(lbl_rows) > chat_cap:
            lbl_rows = lbl_rows[:chat_cap]
        # Upsample minority classes
        if len(lbl_rows) < min_floor:
            factor = (min_floor // len(lbl_rows)) + 1
            lbl_rows = (lbl_rows * factor)[:min_floor]
            random.shuffle(lbl_rows)
        out.extend(lbl_rows)

    random.shuffle(out)
    return out


def compute_class_weights(rows: list[dict]) -> torch.Tensor:
    """Inverse-frequency weights, normalised so they average to 1."""
    counts = Counter(r["label"] for r in rows)
    total = sum(counts.values())
    weights = torch.zeros(len(LABELS))
    for idx in range(len(LABELS)):
        n = counts.get(idx, 1)
        weights[idx] = total / (len(LABELS) * n)
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# 7 ─ FOCAL LOSS + CUSTOM TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(torch.nn.Module):
    """
    Focal Loss with optional class weights and label smoothing.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Label smoothing is folded in before computing the focal weight so that
    the model is penalised less harshly on confident correct predictions
    while still receiving a gradient on hard examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight          # class weights tensor
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = logits.size(-1)
        # Build smoothed target distribution
        smooth = torch.full_like(logits, self.label_smoothing / (n - 1))
        smooth.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        # Per-example cross-entropy with smoothed targets
        ce = -(smooth * log_probs).sum(dim=-1)

        # Focal weight uses the true-class probability (not the smoothed one)
        pt = F.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        focal = ((1.0 - pt) ** self.gamma) * ce

        # Class weighting
        if self.weight is not None:
            focal = focal * self.weight.to(logits.device)[labels]

        return focal.mean()


class FocalTrainer(Trainer):
    """Trainer subclass that replaces the default CE loss with Focal Loss."""

    def __init__(
        self,
        focal_gamma: float,
        class_weights: torch.Tensor,
        label_smoothing: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loss_fn = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self._loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ══════════════════════════════════════════════════════════════════════════════
# 8 ─ METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_macro":    f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def full_evaluation(trainer: FocalTrainer, test_ds: Dataset) -> None:
    out   = trainer.predict(test_ds)
    preds = np.argmax(out.predictions, axis=-1)
    labels = out.label_ids
    print("\n" + "═" * 64)
    print("TEST-SET EVALUATION")
    print("═" * 64)
    print(classification_report(labels, preds, target_names=LABELS, digits=4))
    cm  = confusion_matrix(labels, preds)
    cw  = max(len(l) for l in LABELS) + 2
    hdr = " " * cw + "  ".join(f"{l:>{cw}}" for l in LABELS)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(hdr)
    for i, row in enumerate(cm):
        print(f"{LABELS[i]:<{cw}}" + "  ".join(f"{v:>{cw}}" for v in row))
    print("═" * 64 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9 ─ DATASET ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> tuple[Dataset, Dataset, Dataset, torch.Tensor]:
    """
    Returns: train_ds, val_ds, test_ds, class_weights

    Pipeline:
      load → balance → stratified split → typo-augment train only
    """
    # Load all sources
    all_rows: list[dict] = []
    all_rows.extend(load_hardcoded_data())
    all_rows.extend(load_clinc_data(CLINC_PATH))
    all_rows.extend(load_banking_data(BANKING_PATH))
    all_rows.extend(load_snips_data(SNIPS_PATH))
    all_rows.extend(load_atis_data(ATIS_PATH))
    all_rows.extend(load_massive_data(MASSIVE_PATH))
    all_rows.extend(load_diffusiondb_data(DIFFUSIONDB_PATH))
    all_rows.extend(load_readaloud_data(READALOUD_PATH))
    all_rows.extend(load_musiccaps_data(MUSICCAPS_PATH))

    # Raw distribution
    raw_counts = Counter(ID2LABEL[r["label"]] for r in all_rows)
    print("\n[INFO] Raw label distribution:")
    for lbl in LABELS:
        print(f"       {lbl:<22} {raw_counts.get(lbl, 0):>6}")
    print(f"       {'TOTAL':<22} {sum(raw_counts.values()):>6}\n")

    # Balance
    all_rows = balance(all_rows)
    bal_counts = Counter(ID2LABEL[r["label"]] for r in all_rows)
    print("[INFO] Balanced label distribution:")
    for lbl in LABELS:
        print(f"       {lbl:<22} {bal_counts.get(lbl, 0):>6}")
    print(f"       {'TOTAL':<22} {sum(bal_counts.values()):>6}\n")

    # Class weights for focal loss (computed AFTER balancing)
    class_weights = compute_class_weights(all_rows)
    print(f"[INFO] Class weights: " +
          " | ".join(f"{LABELS[i]}={class_weights[i]:.3f}" for i in range(len(LABELS))))

    # Stratified split BEFORE typo augmentation so val/test stay clean
    texts  = [r["text"]  for r in all_rows]
    labels = [r["label"] for r in all_rows]

    tr_t, te_t, tr_l, te_l = sk_split(
        texts, labels, test_size=test_frac, random_state=SEED, stratify=labels)
    val_size = val_frac / (1.0 - test_frac)
    tr_t, va_t, tr_l, va_l = sk_split(
        tr_t, tr_l, test_size=val_size, random_state=SEED, stratify=tr_l)

    # Apply typo augmentation to TRAINING ONLY
    train_rows = [{"text": t, "label": l} for t, l in zip(tr_t, tr_l)]
    train_rows = augment_with_typos(train_rows, rate=0.20)
    random.shuffle(train_rows)
    tr_t = [r["text"]  for r in train_rows]
    tr_l = [r["label"] for r in train_rows]
    print(f"[INFO] After typo augmentation: train={len(tr_t):,}")

    def to_hf(t, l):
        return Dataset.from_dict({"text": t, "label": l})

    train_ds = to_hf(tr_t, tr_l)
    val_ds   = to_hf(va_t, va_l)
    test_ds  = to_hf(te_t, te_l)

    print(f"[INFO] Splits – train: {len(train_ds):,}  "
          f"val: {len(val_ds):,}  test: {len(test_ds):,}")

    return train_ds, val_ds, test_ds, class_weights


# ══════════════════════════════════════════════════════════════════════════════
# 10 ─ INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    max_len: int = MAX_LEN_CEIL,
) -> list[str]:
    """Batch inference. Call model.to(device) once before using this."""
    model.eval()
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    return [ID2LABEL[i] for i in logits.argmax(dim=-1).tolist()]


# ══════════════════════════════════════════════════════════════════════════════
# 11 ─ MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── Precision and device ─────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16, use_fp16 = False, False
    if device == "cuda":
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 8:
            use_bf16 = True
            print(f"[INFO] Ampere+ GPU detected (cc={cc}) – using bf16")
        else:
            use_fp16 = True
            print(f"[INFO] Pre-Ampere GPU detected (cc={cc}) – using fp16")

    # ── Tokenizer and base model ─────────────────────────────────────────────
    print(f"[INFO] Loading: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # ── Build datasets ───────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, class_weights = build_dataset()

    # ── Auto MAX_LEN ─────────────────────────────────────────────────────────
    all_texts = [{"text": t} for t in train_ds["text"]]
    max_len = compute_max_len(all_texts, tokenizer)

    # ── Tokenize ─────────────────────────────────────────────────────────────
    def tok(batch):
        return tokenizer(batch["text"], padding=False,
                         truncation=True, max_length=max_len)

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tok,   batched=True, remove_columns=["text"])
    test_ds  = test_ds.map(tok,  batched=True, remove_columns=["text"])
    for ds in (train_ds, val_ds, test_ds):
        ds.set_format("torch")

    # ── torch.compile (PyTorch 2.x + CUDA) ───────────────────────────────────
    if hasattr(torch, "compile") and device == "cuda":
        print("[INFO] Applying torch.compile() …")
        model = torch.compile(model)

    # ── Training arguments ───────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_ds) // BATCH_SIZE)
    eval_steps      = steps_per_epoch
    logging_steps   = max(1, steps_per_epoch // 4)

    args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE * 2,
        learning_rate               = LR,
        weight_decay                = WEIGHT_DECAY,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = "cosine",
        eval_strategy               = "steps",   # transformers 5.x API
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = eval_steps,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",
        greater_is_better           = True,
        logging_dir                 = os.path.join(OUTPUT_DIR, "logs"),
        logging_steps               = logging_steps,
        report_to                   = "none",
        seed                        = SEED,
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        optim                       = "adamw_torch_fused" if device == "cuda" else "adamw_torch",
        dataloader_num_workers      = max(1, (os.cpu_count() or 2) // 2),
        label_smoothing_factor      = 0.0,  # smoothing is inside FocalLoss
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = FocalTrainer(
        focal_gamma     = FOCAL_GAMMA,
        class_weights   = class_weights,
        label_smoothing = LABEL_SMOOTHING,
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        data_collator   = DataCollatorWithPadding(tokenizer),
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[INFO] Starting fine-tuning …")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"[INFO] Saving to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as fh:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID,
                   "labels": LABELS, "max_len": max_len}, fh, indent=2)

    # ── Test-set evaluation ───────────────────────────────────────────────────
    full_evaluation(trainer, test_ds)

    # ── Smoke test ────────────────────────────────────────────────────────────
    samples = [
        "Draw a dragon sitting on a pile of gold coins",
        "Turn my idea into a piece of art",
        "Make a meme about programmers",
        "Compose a relaxing piano piece for studying",
        "Make a lofi beat with vinyl crackle",
        "Please read this paragraph aloud for me",
        "Play this text back to me using TTS",
        "What is the capital of Japan?",
        "Help me debug this Python function",
        "Generate an image of a futuristic city at night",
    ]

    best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    best_model.to(device)   # moved once, not per-call
    preds = predict(samples, best_model, tokenizer, device=device, max_len=max_len)

    print(f"\n{'Input':<52}  Predicted")
    print("─" * 72)
    for text, pred in zip(samples, preds):
        print(f"{text:<52}  {pred}")
    print()


if __name__ == "__main__":
    main()