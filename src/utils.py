from google.cloud import texttospeech, texttospeech_v1, speech
from langchain_core.prompts import (ChatPromptTemplate, 
                                    MessagesPlaceholder, 
                                    PromptTemplate)

from langchain_core.messages import SystemMessage, HumanMessage

import streamlit as st
from langchain_groq import ChatGroq
from typing import Iterator, List, Dict, Tuple
import os


lang_code_map = {
    "English": "en-US",
    "Hebrew": "he-IL",
    "French": "fr-FR"
}


tts_name_map = {
    "English": "en-US-Journey-F",
    "Hebrew": "he-IL-Standard-A",
    "French": "fr-FR-Standard-A"
}


def tuplify(history: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    return [(d['role'], d['content']) for d in history]


@st.cache_data
def ask_question(
    history: List[Tuple[str, str]],
    question: str,
    human_language: str,
    ai_language: str
) -> str:
    
    llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=f"""You are a helpful teacher having a conversation with a student that speaks {human_language}.
             Only reply back in {ai_language} even though the student only speaks in {human_language}."""),
            MessagesPlaceholder("history"),
            HumanMessage(content="{question}")
        ]
    )

    chain = prompt | llm 
    
    response = chain.invoke(
                    {
                        "history": history,
                        "question": question
                    }
    )
    
    answer = response.content

    return answer


@st.cache_data
def translate_text(language: str, text: str) -> str:
    if language not in ("English", "French", "Hebrew"):
        raise ValueError(f"Not valid language choice: {language}")
    
    template = "Translate the following into {language} and only return the translated text: {text}"

    llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2)

    prompt = PromptTemplate.from_template(template)

    translation_chain = prompt | llm 

    result = translation_chain.invoke(
            {
                    "language": language,
                    "text": text,
            }
    )

    return result.content


@st.cache_data
def text_to_speech(
    ai_language: str,
    google_api: str,
    translated_answer: str
) -> texttospeech_v1.types.cloud_tts.SynthesizeSpeechResponse:
    
    tts = texttospeech.TextToSpeechClient(client_options={"api_key": google_api})

    synthesis_input = texttospeech.SynthesisInput(text=translated_answer)

    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code_map[ai_language], name=tts_name_map[ai_language])

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response


@st.cache_data
def speech_to_text(
    human_language: str,
    google_api: str,
    question: st.runtime.uploaded_file_manager.UploadedFile
) -> str:
    if human_language not in ("English", "Hebrew", "French"):
        raise ValueError(f"Language choice not supported: {human_language}")
    
    stt = speech.SpeechClient(client_options={"api_key": google_api})

    stt_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=lang_code_map[human_language])

    audio = speech.RecognitionAudio(content=question.getvalue())

    response = stt.recognize(config=stt_config, audio=audio)

    text_question = response.results.pop().alternatives[0].transcript

    return text_question


def create_audio():
    return st.audio_input("Click the microphone to start and stop recording")


def clear_session():
    st.session_state.messages = []
    st.session_state.transcribe = None
