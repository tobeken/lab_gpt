import os
from langchain.prompts.prompt import PromptTemplate
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from playsound import playsound
import speech_recognition as sr
from io import BytesIO
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key= os.environ["OPENAI_API_KEY"]


llm = ChatOpenAI(model_name="gpt-4-1106-preview",temperature=0)
template = """次の会話は調べ物をしている人とエキスパートとの対話です.エキスパートは，物知りで，非常に短い言葉,100字程度でわかりやすく,応答することができます.

Current conversation:
{history}
Researcher: {input}
Expert:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(human_prefix="Researcher"),
)


EXIT_PHRASE = 'さようなら'
flag = True

while flag:
    #input audio
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        print("何か話してください")
        audio = r.listen(source)
        print("音声を取得しました")
    #speechtotext
    audio_data = BytesIO(audio.get_wav_data())
    audio_data.name = "from_mic.wav"
    transcript = openai.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_data, 
    response_format="text"
    )
    print(f"You:{transcript}")
    
    if EXIT_PHRASE in transcript.lower().strip():
        flag = False
    
    user_message = transcript
    ai_message = conversation.run(input=user_message)

    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=ai_message,
        speed=1.2
    )
    response.stream_to_file("output.mp3")
    playsound("output.mp3")

    print(f"AI:{ai_message}")