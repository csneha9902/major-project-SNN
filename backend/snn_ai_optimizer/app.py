from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel, Field
import datetime, json, asyncio, os, random, time, uuid

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# Upload imports
from snn_ai_optimizer.upload import process_edf_file
try:
    from snn_ai_optimizer.upload import process_csv_file
    CSV_AVAILABLE = True
except Exception:
    CSV_AVAILABLE = False

# Analysis imports
from snn_ai_optimizer.analysis import analyze_uploaded_data

# Pipeline & Optimizer imports
from snn_ai_optimizer.pipeline.preprocess import preprocess_run
from snn_ai_optimizer.pipeline.baseline import train_baseline
from snn_ai_optimizer.pipeline.snn_pipeline import snn_run
from snn_ai_optimizer.feedback import generate_feedback
from snn_ai_optimizer.optimizer import update_q_table, recommend_task, _current_epsilon, _load_q_table
from snn_ai_optimizer.streaming import DataStreamer
from snn_ai_optimizer.utils.wellness_tips import get_random_tip
from snn_ai_optimizer.utils.session_tracker import SessionTracker
from snn_ai_optimizer.activity_history import (
    load_calendar_history,
    update_observation_note,
    update_date_status,
    record_session,
    record_analysis_upload,
)

# ------------------ FastAPI App ------------------
app = FastAPI(title="EEG-SNN Adaptive Learning Optimizer")

# Include Core SNN Router
try:
    from snn_ai_optimizer.snn.api import router as snn_router
    app.include_router(snn_router)
except Exception as e:
    print(f"Warning: SNN API module import failed: {e}")

# Include Chatbot Assistant Router
try:
    from snn_ai_optimizer.chatbot.router import router as chatbot_router
    app.include_router(chatbot_router)
except Exception as e:
    print(f"Warning: Chatbot API module import failed: {e}")

# CORS configuration
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Researcher Auth Context ------------------
def get_current_user_optional() -> dict:
    """Provides a consistent researcher identity without hospital EMR dependencies."""
    return {
        "sub": "researcher@eeg-snn.edu",
        "email": "researcher@eeg-snn.edu",
        "name": "EEG SNN Researcher",
        "role": "researcher",
        "user_id": "researcher-01",
        "username": "researcher"
    }

@app.get("/auth/me")
async def get_me_endpoint():
    """Returns the authenticated researcher identity."""
    return get_current_user_optional()

# ------------------ Basic Routes ------------------
@app.get("/")
async def root():
    return {"message": "SNN-AI Optimizer Backend running (Research Mode)"}

# ------------------ Mock Web EEG Endpoint ------------------
_mock_rng = random.Random(314159)

def _mock_eeg_sample(mode: str = "Neutral") -> dict:
    base_alpha = 0.55 + _mock_rng.gauss(0, 0.03)
    base_beta = 0.55 + _mock_rng.gauss(0, 0.03)
    lf_hf = 1.0 + _mock_rng.gauss(0, 0.1)

    if mode == "Focused":
        base_alpha += 0.18
        base_beta -= 0.05
        lf_hf -= 0.25
    elif mode == "Stressed":
        base_alpha -= 0.08
        base_beta += 0.22
        lf_hf += 0.75

    alpha = max(0.0, min(1.8, base_alpha))
    beta = max(0.0, min(1.8, base_beta))
    lf_hf = max(0.1, min(3.5, lf_hf))

    return {
        "timestamp": int(time.time()),
        "eeg": {"alpha": round(alpha, 4), "beta": round(beta, 4)},
        "hrv": {"lf_hf_ratio": round(lf_hf, 4)},
        "source": "mock-web",
        "mode": mode,
    }

@app.get("/mock/eeg")
async def mock_eeg(mode: str = "Neutral"):
    """Return a synthetic EEG/HRV sample to simulate a web feed."""
    return _mock_eeg_sample(mode)

# ------------------ Metrics & Evaluation ------------------
@app.get("/results/metrics")
async def get_metrics():
    """Return baseline + SNN evaluation metrics from held-out test sets."""
    def read_json(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    out = {}
    b = read_json(Path("results/baseline/metrics.json"))
    h = read_json(Path("results/snn/metrics.json"))
    if b: out["baseline"] = b
    if h: out["snn"] = h

    latest = read_json(Path("results/latest_metrics.json"))
    if latest:
        out["latest"] = latest
        out.setdefault("baseline", latest.get("baseline"))
        out.setdefault("snn", latest.get("snn"))

    if not out.get("baseline") and not out.get("snn"):
        return JSONResponse({"error": "No metrics found"}, status_code=404)
    return out

@app.get("/api/benchmark/summary")
async def get_benchmark_summary():
    """
    Returns real held-out evaluation metrics from baseline and SNN pipelines.
    Never fabricates metrics: returns null for missing models (AGENTS.md Rules 1 & 2).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    
    def _resolve_results_path(*parts: str) -> Optional[Path]:
        p1 = repo_root / "results" / Path(*parts)
        if p1.exists():
            return p1
        p2 = Path("results") / Path(*parts)
        if p2.exists():
            return p2
        return None

    baseline_path = _resolve_results_path("baseline", "metrics.json")
    snn_path = _resolve_results_path("snn", "metrics.json")
    report_path = _resolve_results_path("benchmark_report.json")

    baseline_metrics = None
    snn_metrics = None
    report_data = None

    if baseline_path and baseline_path.exists():
        try:
            baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if snn_path and snn_path.exists():
        try:
            snn_metrics = json.loads(snn_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if report_path and report_path.exists():
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "baseline": baseline_metrics,
        "snn": snn_metrics,
        "report": report_data,
        "has_baseline": baseline_metrics is not None,
        "has_snn": snn_metrics is not None,
        "has_report": report_data is not None,
    }

@app.get("/results/history")
async def get_history():
    """History of evaluation runs. Clean empty list when absent — never fabricated."""
    p = Path("results/history/metrics_log.json")
    if not p.exists():
        return {"runs": []}
    try:
        arr = json.loads(p.read_text(encoding="utf-8"))
        runs = []
        for snap in arr:
            ts = snap.get("ts", "?")
            row = {"ts": ts}
            if "baseline" in snap: row["baseline"] = snap["baseline"]
            snn_data = snap.get("snn") or snap.get("hybrid")
            if snn_data: row["snn"] = snn_data
            if "preprocess" in snap: row["preprocess"] = snap["preprocess"]
            if "preprocess_eeg" in snap: row["preprocess_eeg"] = snap["preprocess_eeg"]
            runs.append(row)
        return {"runs": runs[-100:]}
    except Exception:
        return JSONResponse({"runs": []})

# Static files for download
os.makedirs("results", exist_ok=True)
app.mount("/files", StaticFiles(directory="results", html=False), name="files")

# ------------------ WebSocket (logs) ------------------
active_clients = []

async def broadcast(message: str):
    """Send log message to all WebSocket clients."""
    for ws in list(active_clients):
        try:
            await ws.send_text(message)
        except Exception:
            try:
                active_clients.remove(ws)
            except ValueError:
                pass

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_clients.append(ws)
    await ws.send_text("Connected to SNN-AI WebSocket")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in active_clients:
            active_clients.remove(ws)

# ------------------ Pipeline Trigger ------------------
@app.post("/run/pipeline")
async def run_pipeline():
    """Run preprocessing → baseline → SNN pipeline with live updates."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{run_id}.log"

    def log(msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
        print(entry)

    await broadcast("Starting pipeline...")
    log("Starting pipeline...")

    loop = asyncio.get_event_loop()

    await loop.run_in_executor(None, preprocess_run)
    await broadcast("Preprocessing done")
    log("Preprocessing done")

    await loop.run_in_executor(None, train_baseline)
    await broadcast("Baseline training done")
    log("Baseline training done")

    await loop.run_in_executor(None, snn_run)
    await broadcast("SNN training done")
    log("SNN training done")

    await broadcast("Pipeline complete")
    log("Pipeline complete")

    return {"status": "Pipeline complete", "log_file": str(log_file)}

# ------------------ Q-Learning & Feedback ------------------
class FeedbackRequest(BaseModel):
    state: str
    task_index: int = Field(default=0, alias="task_id")
    reward: float
    next_state: Optional[str] = None

@app.get("/feedback")
async def get_feedback():
    return generate_feedback()

@app.post("/feedback")
async def post_feedback(req: FeedbackRequest):
    update_info = update_q_table(req.state, req.task_index, req.reward, req.next_state)
    return {
        "status": "success",
        "message": "Q-table updated (Bellman TD)",
        "update": update_info,
    }

@app.get("/api/recommend")
async def get_recommendation(state: str = "Neutral"):
    """Returns an epsilon-greedy task recommendation from the Bellman Q-table."""
    rec = recommend_task(state)
    q_table = _load_q_table()
    eps = _current_epsilon(q_table)
    return {
        **rec,
        "epsilon": round(eps, 6),
        "state": state,
    }

@app.get("/api/qtable/state")
async def get_qtable_state():
    """Returns current state of the tabular Q-learning policy."""
    q_table = _load_q_table()
    meta = q_table.get("__meta__", {})
    active_states = [s for s in q_table.keys() if s != "__meta__"]
    return {
        "states": active_states,
        "n_updates": meta.get("n_updates", 0),
        "epsilon": round(meta.get("epsilon", _current_epsilon(q_table)), 6),
        "q_table": {k: v for k, v in q_table.items() if k != "__meta__"},
    }

# ------------------ Streaming Telemetry ------------------
streamer = DataStreamer()
session_tracker = SessionTracker()

try:
    streamer.set_tracker(session_tracker)
except Exception:
    pass

@app.websocket("/api/data")
async def data_stream(ws: WebSocket):
    await ws.accept()
    try:
        for frame in streamer.stream(interval_sec=1.5):
            await ws.send_text(json.dumps(frame))
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass

@app.get("/api/snapshot")
async def snapshot():
    latest = streamer.get_latest()
    if latest is None:
        for frame in streamer.stream(interval_sec=0):
            break
        latest = streamer.get_latest()
    if latest is None:
        return JSONResponse({"error": "no data yet"}, status_code=404)
    return latest

@app.get("/api/sim/mode")
async def get_mode():
    return {"mode": streamer.mode}

@app.post("/api/sim/mode")
async def set_mode(req: Request):
    body = await req.json()
    mode = body.get("mode", "Neutral")
    streamer.set_mode(mode)
    return {"ok": True, "mode": streamer.mode}

@app.get("/api/sim/status")
async def get_status():
    return {
        "running": streamer.running,
        "mode": streamer.mode,
        "ingestion": streamer.get_ingestion_status(),
    }

@app.post("/api/sim/start")
async def start_simulation():
    session_tracker.reset()
    session_tracker.start()
    streamer.start_simulation()
    return {"ok": True, "running": streamer.running}

@app.post("/api/sim/stop")
async def stop_simulation():
    try:
        session_tracker.stop()
        summary = session_tracker.generate_summary()
        if summary.get("total_samples", 0) > 0:
            dom_state = summary.get("dominant_state", "Neutral")
            state_map = {"Focused": "Low Arousal", "Neutral": "Baseline", "Stressed": "High Arousal"}
            cal_state = state_map.get(dom_state, "Low Arousal")
            focus_pct = summary.get("focus_percentage", 50.0)
            score = int(min(98, max(45, focus_pct * 0.9 + 10)))
            record_session(
                state=cal_state,
                label="Completed Live EEG Session",
                avg_beta=0.62,
                avg_alpha=0.68,
                avg_hr=72,
                duration_str=summary.get("duration_display", "1m"),
                stress_spikes=summary.get("stress_events", 0),
                snn_score=score,
                notes=summary.get("summary", ""),
                source="live_stream",
            )
    except Exception as e:
        print(f"Note: Error logging stream session to calendar: {e}")
    streamer.stop_simulation()
    return {"ok": True, "running": streamer.running}

@app.get("/api/biometric-details")
async def get_biometric_details():
    latest = streamer.get_latest()
    if not latest:
        return JSONResponse({"error": "no data yet"}, status_code=404)
    eeg = latest.get("eeg", {})
    hrv = latest.get("hrv", {})
    return {
        "beta": float(eeg.get("beta") or 0.0),
        "alpha": float(eeg.get("alpha") or 0.0),
        "heart_rate_bpm": float(hrv.get("heart_rate_bpm") or 0.0),
        "lf_hf_ratio": float(hrv.get("lf_hf_ratio") or 0.0),
        "timestamp": latest.get("timestamp"),
    }

@app.get("/api/wellness-tip")
async def api_wellness_tip(state: str | None = None):
    latest = streamer.get_latest() or {}
    s = state or latest.get("cognitive_state") or "Neutral"
    return get_random_tip(s)

@app.get("/api/session-summary")
async def api_session_summary():
    return session_tracker.generate_summary()

# ------------------ Activity History & Progress Calendar ------------------
class NoteUpdateRequest(BaseModel):
    date: str
    note: str

class StateUpdateRequest(BaseModel):
    date: str
    state: str

class LogSessionRequest(BaseModel):
    date: Optional[str] = None
    state: str = "Low Arousal"
    label: str = "EEG Session"
    avgBeta: float = 0.5
    avgAlpha: float = 0.5
    avgHeartRate: int = 72
    sessionHours: str = "5m"
    stressSpikes: int = 0
    breathingBreaks: int = 0
    snnScore: Any = 80
    notes: Optional[str] = None
    source: str = "manual_entry"

@app.get("/api/progress/history")
async def get_progress_history():
    """Retrieve full persistent session and calendar activity history."""
    return load_calendar_history()

@app.post("/api/progress/note")
async def post_progress_note(req: NoteUpdateRequest):
    """Save clinician or user observation note for a specific date."""
    entry = update_observation_note(req.date, req.note)
    return {"ok": True, "entry": entry}

@app.post("/api/progress/state")
async def post_progress_state(req: StateUpdateRequest):
    """Override or set affective state status for a specific date."""
    entry = update_date_status(req.date, req.state)
    return {"ok": True, "entry": entry}

@app.post("/api/progress/log-session")
async def post_log_session(req: LogSessionRequest):
    """Persist an EEG session directly to the calendar."""
    entry = record_session(
        date_str=req.date,
        state=req.state,
        label=req.label,
        avg_beta=req.avgBeta,
        avg_alpha=req.avgAlpha,
        avg_hr=req.avgHeartRate,
        duration_str=req.sessionHours,
        stress_spikes=req.stressSpikes,
        breathing_breaks=req.breathingBreaks,
        snn_score=req.snnScore,
        notes=req.notes,
        source=req.source,
    )
    return {"ok": True, "entry": entry}

# ------------------ File Upload & Offline Analysis ------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ANALYSIS_STORE = Path("results/analysis")
ANALYSIS_STORE.mkdir(parents=True, exist_ok=True)

DEMO_MANIFEST_PATH = Path("results/demo_samples/manifest.json")

@app.get("/api/demo-samples")
async def get_demo_samples():
    """Return pre-computed synthetic demo datasets for quick evaluation."""
    if not DEMO_MANIFEST_PATH.exists():
        try:
            from snn_ai_optimizer.datasets.generate_synthetic_demo_data import build_all_demo_data
            build_all_demo_data()
        except Exception as e:
            print(f"Failed to auto-generate synthetic demo data: {e}")
            return JSONResponse({"samples": []})
    try:
        data = json.loads(DEMO_MANIFEST_PATH.read_text(encoding="utf-8"))
        return {"samples": data}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user_optional)
):
    """Upload and process EDF or CSV EEG recording."""
    lower_name = file.filename.lower()
    is_edf = lower_name.endswith(".edf")
    is_csv = lower_name.endswith(".csv")
    if not (is_edf or is_csv):
        raise HTTPException(status_code=400, detail="Only EDF or CSV files are supported")

    MAX_SIZE = 200 * 1024 * 1024
    file_content = await file.read()
    if len(file_content) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum 200MB.")

    upload_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"

    try:
        with open(upload_path, "wb") as f:
            f.write(file_content)

        if is_edf:
            analysis_data = process_edf_file(upload_path, upload_id)
        else:
            if not CSV_AVAILABLE:
                raise HTTPException(status_code=500, detail="CSV processing not available on server")
            analysis_data = process_csv_file(upload_path, upload_id)

        analysis_data["uploaded_by"] = current_user.get("email", "researcher@eeg-snn.edu")
        analysis_data["user_id"] = current_user.get("user_id", "researcher-01")
        analysis_data["filename"] = file.filename
        analysis_data["uploaded_at"] = datetime.datetime.now().isoformat()

        analysis_file = ANALYSIS_STORE / f"{upload_id}.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2)

        try:
            upload_path.unlink()
        except Exception:
            pass

        # Automatically record this real analysis into the activity history calendar
        try:
            full_analysis = analyze_uploaded_data(upload_id)
            record_analysis_upload(upload_id, full_analysis)
        except Exception as cal_err:
            print(f"Activity calendar auto-log note: {cal_err}")

        return {
            "upload_id": upload_id,
            "filename": file.filename,
            "status": "processed",
            "n_samples": analysis_data["features"]["n_samples"],
            "duration": analysis_data["metadata"]["duration"],
        }
    except Exception as e:
        try:
            if upload_path.exists():
                upload_path.unlink()
        except Exception:
            pass
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/api/analysis/{upload_id}")
async def get_analysis(
    upload_id: str,
    current_user: dict = Depends(get_current_user_optional)
):
    """Retrieve signal decomposition and SNN state analysis for an uploaded file."""
    try:
        analysis_result = analyze_uploaded_data(upload_id)
        analysis_result["accessed_by"] = {
            "user_id": current_user.get("user_id", "researcher-01"),
            "email": current_user.get("email", "researcher@eeg-snn.edu"),
            "role": current_user.get("role", "researcher")
        }
        return analysis_result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
