import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from fastapi import FastAPI, File, UploadFile
import requests
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv() 
print("ELEVEN_API_KEY loaded:", os.getenv("ELEVEN_API_KEY") is not None)

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

chroma_client = chromadb.PersistentClient(path="./fifa_db")
collection = chroma_client.get_or_create_collection("fifa_worldcup")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if collection.count() == 0:
    print("📥 Ingesting FIFA dataset into ChromaDB...")
    df = pd.read_csv("fifa_1930_2020.csv", encoding="latin1")
    df.columns = df.columns.str.strip().str.lower()  

    ids, docs, embs = [], [], []
    for idx, row in df.iterrows():
        match_date = row.get("match date", "")
        tournament = row.get("tournament name", "")
        stage = row.get("stage name", "")
        stadium = row.get("stadium name", "")
        city = row.get("city name", "")
        country = row.get("country name", "")
        home_team = row.get("home team name", "")
        away_team = row.get("away team name", "")
        score = row.get("score", "")
        result = row.get("result", "")

        text = (
            f"On {match_date}, during the {tournament} at the {stage} stage, "
            f"{home_team} played against {away_team} in {stadium}, {city}, {country}. "
            f"The final score was {score}. Result: {result}."
        )

        ids.append(str(idx))
        docs.append(text)
        embs.append(embedding_model.encode(text).tolist())

    collection.add(ids=ids, documents=docs, embeddings=embs)
    print("✅ Dataset successfully ingested into ChromaDB.")
else:
    print("⚡ FIFA dataset already in ChromaDB. Skipping ingestion.")


def retrieve_context(query: str, top_k=3):
    query_emb = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    return results["documents"][0]


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_fifa(question: str):
    context_docs = retrieve_context(question)
    context = "\n".join(context_docs)

    prompt = f"""
    You are a FIFA World Cup expert. Use the following historical data to answer:

    Context:
    {context}

    Question: {question}
    Answer:
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is running ✅"}


class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question
    try:
        answer = ask_fifa(question)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"⚠️ Error: {str(e)}"}


# --- ✅ FIXED SPEECH ENDPOINT ---
@app.post("/speech")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": ELEVEN_API_KEY}
        files = {"file": open(audio_path, "rb")}
        data = {"model_id": "scribe_v1"} 

        response = requests.post(url, headers=headers, files=files, data=data)
        result = response.json()

        print("🔥 ElevenLabs raw response:", result)

        question = result.get("text", "")

        answer = ask_fifa(question) if question else "⚠️ Couldn’t transcribe speech."

        return {"question": question, "answer": answer}

    except Exception as e:
        print("❌ Error in /speech:", str(e))
        return {"error": f"⚠️ {str(e)}"}


@app.get("/ping")
def ping():
    return {"msg": "pong"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
