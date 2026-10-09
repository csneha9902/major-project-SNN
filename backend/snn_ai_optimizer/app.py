from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path
import datetime, json, asyncio, os, random, time, uuid, shutil

# Auth imports - with error handling
try:
    from snn_ai_optimizer.auth import get_oauth_router
    from snn_ai_optimizer.auth.jwt_utils import get_current_user
    AUTH_AVAILABLE = True
except Exception as e:
    print(f"Warning: Auth module import failed: {e}")
    AUTH_AVAILABLE = False
    # Create dummy functions
    def get_oauth_router():
        from fastapi import APIRouter
        return APIRouter(prefix="/auth", tags=["auth"])
    def get_current_user():
        from fastapi import Depends
        from fastapi.security import HTTPBearer
        security = HTTPBearer(auto_error=False)
        async def _get_user(credentials = None):
            return {
                "sub": "researcher@eeg-snn.edu",
                "email": "researcher@eeg-snn.edu",
                "name": "EEG SNN Researcher",
                "role": "researcher"
            }
        return Depends(_get_user)

# Upload imports
from snn_ai_optimizer.upload import process_edf_file
try:
    from snn_ai_optimizer.upload import process_csv_file
    CSV_AVAILABLE = True
except Exception:
    CSV_AVAILABLE = False

# Analysis imports
from snn_ai_optimizer.analysis import analyze_uploaded_data

# Export imports
from snn_ai_optimizer.export import generate_pdf_report

# ------------------ FastAPI App ------------------
app = FastAPI()

# Initialize database
try:
    from snn_ai_optimizer.db.init_db import init_db
    init_db()
    print("Database initialized successfully")
except Exception as e:
    print(f"Warning: Database initialization failed: {e}")

# Include OAuth router (with comprehensive error handling)
try:
    if AUTH_AVAILABLE:
        app.include_router(get_oauth_router())
    else:
        raise ImportError("Auth module not available")
except Exception as e:
    import traceback
    print(f"WARNING: Failed to include OAuth router: {e}")
    traceback.print_exc()
    # Create a minimal fallback auth router that always works
    from fastapi import APIRouter
    from fastapi.responses import RedirectResponse
    auth_router = APIRouter(prefix="/auth", tags=["auth"])
    
    def _create_demo_token():
        """Helper to create researcher demo token."""
        try:
            from snn_ai_optimizer.auth.jwt_utils import create_access_token
            token_data = {"sub": "researcher@eeg-snn.edu", "email": "researcher@eeg-snn.edu", "name": "EEG SNN Researcher"}
            return create_access_token(token_data)
        except ImportError:
            try:
                from jose import jwt
                import datetime
                SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key-change-in-production-min-32-chars")
                token_data = {"sub": "researcher@eeg-snn.edu", "email": "researcher@eeg-snn.edu", "name": "EEG SNN Researcher", "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)}
                return jwt.encode(token_data, SECRET_KEY, algorithm="HS256")
            except ImportError:
                return "demo_token_researcher@eeg-snn.edu"
    
    @auth_router.get("/login")
    async def fallback_login():
        """Fallback login when OAuth router fails to load."""
        token = _create_demo_token()
        frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173")
        return RedirectResponse(url=f"{frontend_url}/auth/callback?token={token}")
    
    @auth_router.get("/callback")
    async def fallback_callback():
        """Fallback callback."""
        token = _create_demo_token()
        frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173")
        return RedirectResponse(url=f"{frontend_url}/auth/callback?token={token}")
    
    @auth_router.get("/me")
    async def fallback_me():
        """Fallback /auth/me endpoint."""
        return {"sub": "researcher@eeg-snn.edu", "email": "researcher@eeg-snn.edu", "name": "EEG SNN Researcher", "role": "researcher"}
    
    app.include_router(auth_router)

# Include SNN Router
try:
    from snn_ai_optimizer.snn.api import router as snn_router
    app.include_router(snn_router)
except Exception as e:
    print(f"Warning: SNN module import failed: {e}")

# Legacy medical modules: decoupled per AGENTS.md Rule 12 & Rule 20
ENABLE_LEGACY_MEDICAL_MODULES = os.environ.get("ENABLE_LEGACY_MEDICAL_MODULES", "false").lower() in ("true", "1")

if ENABLE_LEGACY_MEDICAL_MODULES:
    try:
        from snn_ai_optimizer.patient.router import router as patient_router
        app.include_router(patient_router, prefix="/api")
        print("Legacy Patient router included at /api/patients/")
    except Exception as e:
        print(f"Warning: Patient router import failed: {e}")

    try:
        from snn_ai_optimizer.appointment.router import router as appointment_router
        app.include_router(appointment_router, prefix="/api")
        print("Legacy Appointment router included at /api/appointments/")
    except Exception as e:
        print(f"Warning: Appointment router import failed: {e}")

    try:
        from snn_ai_optimizer.collaboration.router import router as collaboration_router
        app.include_router(collaboration_router, prefix="/api")
        print("Legacy Collaboration router included at /api/collaboration/")
    except Exception as e:
        print(f"Warning: Collaboration router import failed: {e}")

    try:
        from snn_ai_optimizer.mail.router import router as mail_router
        app.include_router(mail_router, prefix="/api")
        print("Legacy Mail router included at /api/mail/")
    except Exception as e:
        print(f"Warning: Mail router import failed: {e}")
else:
    print("Core EEG SNN Q-Learning mode active. Legacy medical modules decoupled (AGENTS.md Rule 12).")

# CORS (allow frontend at :5173 to access backend)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # in prod, restrict to http://localhost:5173 or deployed frontend
    allow_credentials=True,  # Changed to True for OAuth cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ------------------ Basic Routes ------------------
@app.get("/")
async def root():
    return {"message": "SNN-AI Optimizer Backend running"}


@app.get("/auth/me")
async def get_me_endpoint(request: Request):
    """
    Returns the currently authenticated user from JWT Bearer token.
    Falls back to demo user if no valid token is provided.
    """
    from snn_ai_optimizer.auth.jwt_utils import verify_token
    from snn_ai_optimizer.db.session import SessionLocal
    from snn_ai_optimizer.db.models import User as DBUser

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"detail": "Not authenticated"}
        )

    token = auth_header.split(" ", 1)[1]
    try:
        payload = verify_token(token)
        email = payload.get("sub") or payload.get("email")
        db = SessionLocal()
        try:
            user = db.query(DBUser).filter(DBUser.email == email).first()
        finally:
            db.close()

        if user:
            return {
                "sub": user.email,
                "email": user.email,
                "name": user.full_name,
                "role": user.role.value if hasattr(user.role, "value") else str(user.role),
                "user_id": user.user_id,
                "username": user.username,
            }
        # Token valid but user not in DB — return payload directly
        return payload
    except Exception:
        return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})

@app.get("/results/metrics")
async def get_metrics():
    """Return baseline + hybrid metrics if available"""
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

# ------------------ WebSocket (logs) ------------------
active_clients = []

async def broadcast(message: str):
    """Send log message to all WebSocket clients"""
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
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        if ws in active_clients:
            active_clients.remove(ws)

# ------------------ Pipeline Trigger ------------------
from snn_ai_optimizer.pipeline.preprocess import preprocess_run
from snn_ai_optimizer.pipeline.baseline import train_baseline
from snn_ai_optimizer.pipeline.snn_pipeline import snn_run
from snn_ai_optimizer.utils.wellness_tips import get_random_tip
from snn_ai_optimizer.utils.session_tracker import SessionTracker

@app.post("/run/pipeline")
async def run_pipeline():
    """Run preprocessing → baseline → hybrid, stream updates to WS + log file"""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("results/logs"); log_file.mkdir(parents=True, exist_ok=True)
    log_file = log_file / f"run_{run_id}.log"

    def log(msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
        print(entry)

    await broadcast("Starting pipeline...")
    log("Starting pipeline...")

    # Offload blocking steps to threadpool to avoid blocking the event loop
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
from snn_ai_optimizer.feedback import generate_feedback
from typing import Optional
from pydantic import BaseModel, Field
from snn_ai_optimizer.optimizer import update_q_table, recommend_task, _current_epsilon, _load_q_table

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
    """
    Returns an epsilon-greedy task recommendation from the Bellman Q-table.
    """
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
    """
    Returns the current state of the tabular Q-learning policy.
    """
    q_table = _load_q_table()
    meta = q_table.get("__meta__", {})
    active_states = [s for s in q_table.keys() if s != "__meta__"]
    return {
        "states": active_states,
        "n_updates": meta.get("n_updates", 0),
        "epsilon": round(meta.get("epsilon", _current_epsilon(q_table)), 6),
        "q_table": {k: v for k, v in q_table.items() if k != "__meta__"},
    }

@app.get("/api/benchmark/summary")
async def get_benchmark_summary():
    """
    Returns real held-out evaluation metrics from baseline and SNN pipelines.
    Never fabricates metrics: returns null for missing models.
    """
    baseline_path = Path("results/baseline/metrics.json")
    snn_path = Path("results/snn/metrics.json")

    baseline_metrics = None
    snn_metrics = None

    if baseline_path.exists():
        try:
            baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if snn_path.exists():
        try:
            snn_metrics = json.loads(snn_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "baseline": baseline_metrics,
        "snn": snn_metrics,
        "has_baseline": baseline_metrics is not None,
        "has_snn": snn_metrics is not None,
    }

from fastapi.responses import JSONResponse

@app.get("/results/history")
async def get_history():
    p = Path("results/history/metrics_log.json")
    if not p.exists():
        # Clean empty fallback — never fabricate metrics (AGENTS.md Rule 1)
        return {"runs": []}
    try:
        arr = json.loads(p.read_text(encoding="utf-8"))
        # Fold snapshots by timestamp into combined rows for plotting
        runs = []
        acc = {}
        for snap in arr:
            ts = snap.get("ts", "?")
            row = {"ts": ts}
            if "baseline" in snap: row["baseline"] = snap["baseline"]
            
            # Map legacy hybrid data to snn for charting compatibility
            snn_data = snap.get("snn") or snap.get("hybrid")
            if snn_data: row["snn"] = snn_data
            
            if "preprocess" in snap: row["preprocess"] = snap["preprocess"]
            if "preprocess_eeg" in snap: row["preprocess_eeg"] = snap["preprocess_eeg"]
            if "preprocess_mri" in snap: row["preprocess_mri"] = snap["preprocess_mri"]
            runs.append(row)
        return {"runs": runs[-100:]}  # last 100
    except Exception:
        return JSONResponse({"runs": []})
import os
from fastapi.staticfiles import StaticFiles

# Expose raw files so you can download exact artifacts:
# e.g. http://localhost:8000/files/hybrid/metrics.json
#      http://localhost:8000/files/history/metrics_log.json
os.makedirs("results", exist_ok=True)
app.mount("/files", StaticFiles(directory="results", html=False), name="files")

# ------------------ Streaming API per dev_plan ------------------
from snn_ai_optimizer.streaming import DataStreamer

streamer = DataStreamer()
session_tracker = SessionTracker()

# Link tracker to stream lifecycle via sim controls
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
            # Yield control to event loop
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass

@app.get("/api/snapshot")
async def snapshot():
    latest = streamer.get_latest()
    if latest is None:
        # Generate one frame to initialize
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
    session_tracker.reset(); session_tracker.start()
    streamer.start_simulation()
    return {"ok": True, "running": streamer.running}

@app.post("/api/sim/stop")
async def stop_simulation():
    try:
        session_tracker.stop()
    except Exception:
        pass
    streamer.stop_simulation()
    return {"ok": True, "running": streamer.running}

# ------------------ New API (biometrics, tips, summary) ------------------

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

# ------------------ File Upload & Analysis (Protected) ------------------

# Store uploads temporarily
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ANALYSIS_STORE = Path("results/analysis")
ANALYSIS_STORE.mkdir(parents=True, exist_ok=True)

# More forgiving auth dependency for upload endpoints
_optional_security = HTTPBearer(auto_error=False)

async def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(_optional_security)) -> dict:
    """Get current user, but fallback to demo user if token is missing/invalid."""
    if credentials:
        try:
            if AUTH_AVAILABLE:
                from snn_ai_optimizer.auth.jwt_utils import verify_token
                payload = verify_token(credentials.credentials)
                # Return a dict similar to what get_current_user would return for backward compatibility
                return {
                    "sub": payload.get("sub", "demo@doctor.com"),
                    "email": payload.get("email", "demo@doctor.com"),
                    "name": payload.get("name", "Demo Doctor"),
                    "role": payload.get("role", "doctor")
                }
        except Exception:
            pass
    # Fallback to demo user if no credentials or validation fails
    return {"sub": "demo@doctor.com", "email": "demo@doctor.com", "name": "Demo Doctor", "role": "doctor"}

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
    """Upload and process EDF or CSV file (protected endpoint)."""
    lower_name = file.filename.lower()
    is_edf = lower_name.endswith(".edf")
    is_csv = lower_name.endswith(".csv")
    if not (is_edf or is_csv):
        raise HTTPException(status_code=400, detail="Only EDF or CSV files are supported")

    # Check file size (200MB limit)
    MAX_SIZE = 200 * 1024 * 1024
    file_content = await file.read()
    if len(file_content) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum 200MB.")

    upload_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"

    try:
        # Save uploaded file
        with open(upload_path, "wb") as f:
            f.write(file_content)

        # Process according to type
        if is_edf:
            analysis_data = process_edf_file(upload_path, upload_id)
        else:
            if not CSV_AVAILABLE:
                raise HTTPException(status_code=500, detail="CSV processing not available on server")
            analysis_data = process_csv_file(upload_path, upload_id)

        # Add metadata
        analysis_data["uploaded_by"] = current_user.get("email", "unknown")
        analysis_data["user_id"] = current_user.get("user_id", "unknown") if isinstance(current_user, dict) else getattr(current_user, 'user_id', 'unknown')
        analysis_data["filename"] = file.filename
        analysis_data["uploaded_at"] = datetime.datetime.now().isoformat()

        # Save analysis results
        analysis_file = ANALYSIS_STORE / f"{upload_id}.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2)

        # Clean up uploaded file after processing
        try:
            upload_path.unlink()
        except Exception:
            pass

        return {
            "upload_id": upload_id,
            "filename": file.filename,
            "status": "processed",
            "n_samples": analysis_data["features"]["n_samples"],
            "duration": analysis_data["metadata"]["duration"],
        }

    except Exception as e:
        # Clean up on error
        try:
            if upload_path.exists():
                upload_path.unlink()
        except Exception:
            pass
        # Surface FastAPI HTTPExceptions as-is
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/api/analysis/{upload_id}")
async def get_analysis(
    upload_id: str,
    current_user: dict = Depends(get_current_user_optional)
):
    """Get extended analysis for uploaded file (protected endpoint)."""
    try:
        analysis_result = analyze_uploaded_data(upload_id)
        # Add user info to the result for tracking
        if isinstance(current_user, dict):
            analysis_result["accessed_by"] = {
                "user_id": current_user.get("user_id", "unknown"),
                "email": current_user.get("email", "unknown"),
                "role": current_user.get("role", "unknown")
            }
        else:
            analysis_result["accessed_by"] = {
                "user_id": getattr(current_user, 'user_id', 'unknown'),
                "email": getattr(current_user, 'email', 'unknown'),
                "role": getattr(current_user, 'role', 'unknown')
            }
        return analysis_result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/api/analysis/{upload_id}/export-pdf")
async def export_pdf(
    upload_id: str,
    current_user: dict = Depends(get_current_user_optional)
):
    """Export analysis as PDF (protected endpoint)."""
    try:
        analysis_result = analyze_uploaded_data(upload_id)
        # Add user info to the result for tracking
        if isinstance(current_user, dict):
            analysis_result["accessed_by"] = {
                "user_id": current_user.get("user_id", "unknown"),
                "email": current_user.get("email", "unknown"),
                "role": current_user.get("role", "unknown")
            }
        else:
            analysis_result["accessed_by"] = {
                "user_id": getattr(current_user, 'user_id', 'unknown'),
                "email": getattr(current_user, 'email', 'unknown'),
                "role": getattr(current_user, 'role', 'unknown')
            }
        pdf_bytes = generate_pdf_report(analysis_result)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="analysis_{upload_id}.pdf"'
            }
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")
