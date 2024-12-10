import streamlit as st
import requests
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os
from typing import Iterator, List, Dict, Tuple

groq_api = os.getenv("GROQ_API_KEY")
play_ht_api = os.getenv("PLAY_HT_API")
play_ht_user_id = os.getenv("PLAY_HT_USER_ID")



def tuplify(history: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    return [(d['role'], d['content']) for d in history]


def ask_question(llm: ChatGroq, history: List[Tuple[str, str]], question: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder("history"),
            ("human", "{question}")
        ]
    )

    new_prompt = prompt.invoke(
           {
           "history": history,
           "question": question
       }
    )

    response = llm.invoke(new_prompt)
    
    return response.content


def main():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("Ask A Question:")
    question = st.audio_input("Click the microphone to start and stop recording")

    client = Groq()

    llm = ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2)
    
    if question:
        transcription = client.audio.transcriptions.create(
                        file=("question.wav", question),
                        model="whisper-large-v3-turbo",
                        response_format="json")
        
        text_question = transcription.text 

        st.session_state.messages.append({"role": "human", "content": f"Human: {text_question}"})

        if text_question:
            
            history = tuplify(st.session_state.messages)
            answer = ask_question(llm=llm, history=history, question=text_question)

            st.session_state.messages.append({"role": "ai", "content": answer})
        
            url = "https://api.play.ht/api/v2/tts/stream"

            payload = {
                "voice": "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
                "output_format": "mp3",
                "text": answer,
                "voice_2": "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
                "voice_engine": "PlayDialog",
                "turn_prefix_2": "Town Mouse:",
                "turn_prefix": "Country Mouse:"
            }
            headers = {
                "accept": "audio/mpeg",
                "content-type": "application/json",
                "AUTHORIZATION": play_ht_api,
                "X-USER-ID": play_ht_user_id
            }
            
            response = requests.post(url, json=payload, headers=headers)

            if response:
                st.markdown("Play AI Response:")
                st.audio(response.content, "audio/wav")

if __name__ == "__main__":
    main()
