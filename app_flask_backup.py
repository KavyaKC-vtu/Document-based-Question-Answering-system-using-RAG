"""
DocuMind RAG - Real RAG with FAISS vector DB + Groq LLM + SentenceTransformers
Run with global python: python app.py  (uses GROQ_API_KEY from .env)
"""
import os, re, json, uuid, pickle
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np

# --- Config ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("ROQ_API_KEY")  # support typo
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_DIR = Path("sample_docs")
FAISS_DIR = Path("faiss_store")
FAISS_DIR.mkdir(exist_ok=True)
ALLOWED = {".txt",".md",".pdf"}
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

print(f"[Config] GROQ_MODEL={GROQ_MODEL} EMBEDDING={EMBEDDING_MODEL_NAME} KEY={'set' if GROQ_API_KEY else 'MISSING'}")

# --- Imports that may need global env ---
try:
    from PyPDF2 import PdfReader
    HAS_PDF=True
except: HAS_PDF=False; print("PyPDF2 not found")

try:
    import faiss
    HAS_FAISS=True
    print(f"[FAISS] version {faiss.__version__}")
except Exception as e:
    HAS_FAISS=False; print(f"[FAISS] not available: {e}")

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST=True
except Exception as e:
    HAS_ST=False; print(f"[ST] not available: {e}")

try:
    from groq import Groq
    HAS_GROQ=True
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    print(f"[GROQ] client {'ready' if groq_client else 'no key'}")
except Exception as e:
    HAS_GROQ=False; groq_client=None; print(f"[GROQ] import failed: {e}")

# --- Embedding model ---
embedding_model = None
EMBED_DIM = 384
if HAS_ST:
    try:
        print(f"[Embed] Loading {EMBEDDING_MODEL_NAME} ...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        EMBED_DIM = embedding_model.get_sentence_embedding_dimension()
        print(f"[Embed] Loaded dim={EMBED_DIM}")
    except Exception as e:
        print(f"[Embed] Failed to load: {e}")
        HAS_ST=False

# Fallback TF-IDF if ST not available
if not HAS_ST:
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = None
    tfidf_matrix = None
    print("[Embed] Using TF-IDF fallback")

# --- Store ---
documents = []  # {id, filename, text, chars}
chunks = []     # {id, doc_id, filename, text, idx}
faiss_index = None
chunk_embeddings = None  # numpy array for FAISS (normalized)
INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "meta.json"

def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n==0]=1
    return v / n

def embed_texts(texts):
    if HAS_ST and embedding_model is not None:
        embs = embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        # already normalized if normalize_embeddings True, but double ensure
        return np.array(embs, dtype=np.float32)
    else:
        # TF-IDF fallback not used with FAISS really, but handle
        raise RuntimeError("ST not available")

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= chunk_size:
        return [text]
    out=[]
    start=0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            snap = text.rfind('. ', start+chunk_size-200, end)
            if snap != -1:
                end = snap+1
            else:
                snap = text.rfind(' ', end-80, end)
                if snap != -1: end = snap
        piece = text[start:end].strip()
        if len(piece) > 40:
            out.append(piece)
        if end >= len(text): break
        start = end - overlap
    return out

def extract_text_from_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf" and HAS_PDF:
        reader = PdfReader(str(path))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

def build_faiss_index():
    global faiss_index, chunk_embeddings
    if not chunks:
        faiss_index=None; chunk_embeddings=None; return
    texts = [c["text"] for c in chunks]
    if HAS_ST and embedding_model is not None:
        print(f"[FAISS] Embedding {len(texts)} chunks...")
        embs = embed_texts(texts)  # (n, dim) float32 normalized
        chunk_embeddings = embs
        dim = embs.shape[1]
        # IndexFlatIP for cosine (since normalized, IP = cosine)
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embs)
        print(f"[FAISS] Index built: {faiss_index.ntotal} vectors, dim {dim}")
        # persist
        try:
            faiss.write_index(faiss_index, str(INDEX_PATH))
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump({"chunks": chunks, "documents": documents}, f, indent=2)
            print(f"[FAISS] Persisted to {INDEX_PATH}")
        except Exception as e: print(f"[FAISS] Persist failed: {e}")
    else:
        # TF-IDF fallback build
        global tfidf_vectorizer, tfidf_matrix
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        print(f"[TF-IDF] Built matrix {tfidf_matrix.shape}")

def load_persisted():
    global documents, chunks, faiss_index, chunk_embeddings
    if META_PATH.exists() and INDEX_PATH.exists() and HAS_FAISS:
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                data=json.load(f)
                documents.extend(data.get("documents",[]))
                chunks.extend(data.get("chunks",[]))
            faiss_index = faiss.read_index(str(INDEX_PATH))
            # need embeddings for re-adding? We can re-embed or load? For now re-embed is fine, but we can keep faiss
            # Rebuild embeddings array by re-embedding chunks (to keep in sync)
            if chunks and HAS_ST:
                chunk_embeddings = embed_texts([c["text"] for c in chunks])
            print(f"[FAISS] Loaded persisted: {len(documents)} docs, {len(chunks)} chunks")
            return True
        except Exception as e:
            print(f"[FAISS] Load failed: {e}")
    return False

# Try load persisted, else load sample docs
if not load_persisted():
    for p in SAMPLE_DIR.glob("*"):
        if p.suffix.lower() in ALLOWED and p.is_file():
            try:
                txt = extract_text_from_file(p)
                if not txt.strip(): continue
                doc_id = str(uuid.uuid4())[:8]
                documents.append({"id":doc_id, "filename":p.name, "text":txt, "chars":len(txt)})
                for idx, ch in enumerate(chunk_text(txt)):
                    chunks.append({"id":f"{doc_id}_{idx}", "doc_id":doc_id, "filename":p.name, "text":ch, "idx":idx})
            except Exception as e: print("skip", p, e)
    if chunks:
        build_faiss_index()

def add_document(filename, text):
    doc_id = str(uuid.uuid4())[:8]
    documents.append({"id":doc_id, "filename":filename, "text":text, "chars":len(text)})
    new_chunks = []
    for idx, ch in enumerate(chunk_text(text)):
        c = {"id":f"{doc_id}_{idx}", "doc_id":doc_id, "filename":filename, "text":ch, "idx":idx}
        chunks.append(c); new_chunks.append(c)
    # incremental FAISS add or rebuild
    if HAS_ST and faiss_index is not None:
        embs = embed_texts([c["text"] for c in new_chunks])
        faiss_index.add(embs)
        # persist
        try:
            faiss.write_index(faiss_index, str(INDEX_PATH))
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump({"chunks": chunks, "documents": documents}, f, indent=2)
        except Exception as e: print("persist fail", e)
    else:
        build_faiss_index()
    return doc_id

def retrieve(query, top_k=4):
    if not chunks:
        return []
    if HAS_ST and faiss_index is not None and embedding_model is not None:
        q_emb = embed_texts([query])  # (1, dim)
        # faiss search
        scores, indices = faiss_index.search(q_emb, top_k*2)  # get extra for filtering
        results=[]
        for score, idx in zip(scores[0], indices[0]):
            if idx==-1: continue
            if score < 0.18:  # cosine threshold for ST (normalized IP)
                continue
            c = chunks[idx]
            results.append({**c, "score": round(float(score),4)})
            if len(results)>=top_k: break
        # if too strict and no results, return top 1 regardless
        if not results and len(scores[0])>0:
            idx = int(indices[0][0])
            if idx!=-1:
                results.append({**chunks[idx], "score": round(float(scores[0][0]),4)})
        return results
    else:
        # TF-IDF fallback cosine
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, tfidf_matrix)[0]
        idxs = np.argsort(sims)[::-1][:top_k]
        res=[]
        for i in idxs:
            score=float(sims[i])
            if score < 0.03: continue
            res.append({**chunks[i], "score": round(score,4)})
        return res

def groq_generate(query, contexts):
    if not contexts:
        return {"answer":"I couldn't find relevant info in your documents. Try rephrasing or upload more files.", "grounded": False, "via":"none"}
    if not groq_client:
        # fallback extractive
        return extractive_answer(query, contexts)
    ctx_text = "\n\n".join([f"[{c['filename']} chunk {c['idx']}]: {c['text']}" for c in contexts])
    system = "You are DocuMind, a grounded RAG assistant. Answer ONLY using the provided context. Cite sources as [filename]. Be concise, accurate, and helpful. If context insufficient, say you don't know."
    prompt = f"Context:\n{ctx_text}\n\nQuestion: {query}\n\nAnswer with citations:"
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role":"system","content": system},
                {"role":"user","content": prompt}
            ],
            temperature=0.2,
            max_tokens=800,
            top_p=0.9,
        )
        ans = completion.choices[0].message.content or ""
        # Groq's qwen & compound may include <think> tags - strip
        ans = re.sub(r'<think>.*?</think>', '', ans, flags=re.DOTALL).strip()
        # Also remove Reasoning / Answer headers from compound model
        # keep but clean
        if not ans:
            ans = extractive_answer(query, contexts)["answer"]
        return {"answer": ans, "grounded": True, "via": f"groq:{GROQ_MODEL}"}
    except Exception as e:
        print(f"[Groq] error: {e}")
        return extractive_answer(query, contexts, error=str(e))

def extractive_answer(query, contexts, error=None):
    query_words=set(re.findall(r'\w+', query.lower()))
    scored=[]
    for c in contexts:
        sents = re.split(r'(?<=[.!?])\s+', c["text"])
        for s in sents:
            overlap=len(query_words & set(re.findall(r'\w+', s.lower())))
            if overlap>0: scored.append((overlap, s.strip(), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        top=[]
        seen=set()
        for _, s, c in scored:
            if s not in seen:
                top.append(f"{s} [{c['filename']}]")
                seen.add(s)
            if len(top)>=3: break
        ans="Based on your documents:\n\n" + "\n\n".join([f"• {s}" for s in top])
        ans+=f"\n\n(Retrieved {len(contexts)} chunks via FAISS vector search)"
        if error: ans+=f"\n[Groq fallback: {error}]"
    else:
        ans="Relevant excerpts:\n\n" + "\n\n---\n\n".join([f"**{c['filename']}** (score {c['score']})\n{c['text'][:500]}" for c in contexts])
    return {"answer": ans, "grounded": True, "via":"extractive-fallback"}

# --- Flask ---
app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status":"ok",
        "docs": len(documents),
        "chunks": len(chunks),
        "vector_db": "FAISS" if HAS_FAISS and faiss_index is not None else "TF-IDF",
        "faiss_index_size": faiss_index.ntotal if faiss_index is not None else 0,
        "embedding_model": EMBEDDING_MODEL_NAME if HAS_ST else "tf-idf",
        "embedding_dim": EMBED_DIM if HAS_ST else 0,
        "groq_model": GROQ_MODEL if HAS_GROQ else "none",
        "groq_ready": bool(groq_client),
        "indexed": (faiss_index is not None and faiss_index.ntotal>0) or (not HAS_ST and tfidf_matrix is not None)
    })

@app.route("/api/documents")
def list_docs():
    return jsonify({"documents":[{"id":d["id"],"filename":d["filename"],"chars":d["chars"],"chunks":len([c for c in chunks if c["doc_id"]==d["id"]])} for d in documents], "chunks": len(chunks)})

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    f=request.files["file"]
    if not f.filename: return jsonify({"error":"Empty filename"}),400
    ext=Path(f.filename).suffix.lower()
    if ext not in ALLOWED: return jsonify({"error":f"Allowed: {ALLOWED}"}),400
    fname=secure_filename(f.filename)
    save_path=UPLOAD_DIR/fname
    i=1
    while save_path.exists():
        save_path=UPLOAD_DIR / f"{Path(fname).stem}_{i}{ext}"; i+=1
    f.save(save_path)
    try: text=extract_text_from_file(save_path)
    except Exception as e: return jsonify({"error":str(e)}),500
    if not text.strip(): return jsonify({"error":"Empty document"}),400
    doc_id=add_document(save_path.name, text)
    return jsonify({"ok":True,"doc_id":doc_id,"filename":save_path.name,"chunks":len([c for c in chunks if c["doc_id"]==doc_id])})

@app.route("/api/query", methods=["POST"])
def query():
    data=request.get_json(force=True)
    q=(data.get("query") or "").strip()
    top_k=int(data.get("top_k") or 4)
    if not q: return jsonify({"error":"Empty query"}),400
    if not chunks: return jsonify({"error":"No documents indexed. Upload files first."}),400
    contexts=retrieve(q, top_k=top_k)
    if not contexts:
        return jsonify({"query":q,"answer":"No relevant chunks found. Try a broader question or lower similarity threshold.","via":"none","contexts":[],"grounded":False})
    result=groq_generate(q, contexts)
    return jsonify({"query":q,"answer":result["answer"],"via":result.get("via","extractive"),"contexts":contexts,"grounded":result.get("grounded",True)})

@app.route("/api/clear", methods=["POST"])
def clear():
    global faiss_index, chunk_embeddings
    documents.clear(); chunks.clear()
    if HAS_FAISS:
        faiss_index=None
        try:
            if INDEX_PATH.exists(): INDEX_PATH.unlink()
            if META_PATH.exists(): META_PATH.unlink()
        except: pass
    else:
        global tfidf_vectorizer, tfidf_matrix
        tfidf_vectorizer=None; tfidf_matrix=None
    return jsonify({"ok":True})

@app.route("/api/reset_sample", methods=["POST"])
def reset_sample():
    global faiss_index
    documents.clear(); chunks.clear()
    faiss_index=None
    for p in SAMPLE_DIR.glob("*"):
        if p.suffix.lower() in ALLOWED:
            try:
                txt=extract_text_from_file(p)
                if txt.strip():
                    doc_id=str(uuid.uuid4())[:8]
                    documents.append({"id":doc_id,"filename":p.name,"text":txt,"chars":len(txt)})
                    for idx,ch in enumerate(chunk_text(txt)):
                        chunks.append({"id":f"{doc_id}_{idx}","doc_id":doc_id,"filename":p.name,"text":ch,"idx":idx})
            except: pass
    if chunks: build_faiss_index()
    return jsonify({"ok":True,"docs":len(documents),"chunks":len(chunks)})

if __name__=="__main__":
    print(f"[Ready] {len(documents)} docs, {len(chunks)} chunks | FAISS={'yes' if faiss_index is not None else 'no'} | Groq={GROQ_MODEL}")
    print("Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
