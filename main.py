from datetime import datetime, date, time, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import json
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, Column, Integer, String, DateTime, UniqueConstraint, select
from sqlalchemy.orm import declarative_base, sessionmaker


# ----------------------------
# Config DB (ruta fija)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "agenda.db"
DB_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# Reglas base (MVP)
WORK_START = time(9, 0)
WORK_END = time(18, 0)
SLOT_MINUTES = 30
BUFFER_MINUTES = 0  # 5 o 10 si querés buffer real

SERVICE_DURATION = {
    "Consulta": 30,
    "Control": 30,
    "Corte": 30,
    "Color": 90,
}

SESSION_CTX: Dict[str, Dict[str, str]] = {}

def save_session_ctx(sid: str, service: Optional[str], day_str: Optional[str], time_hint: Optional[str]):
    data: Dict[str, str] = {}
    if service:
        data["service"] = service
    if day_str:
        data["day"] = day_str
    if time_hint:
        data["time_hint"] = time_hint
    if data:
        SESSION_CTX[sid] = {**SESSION_CTX.get(sid, {}), **data}


# ----------------------------
# Ollama Cloud config
# ----------------------------
OLLAMA_API_BASE = "https://ollama.com/api"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")  # podés cambiarlo por env var


# ----------------------------
# DB Model
# ----------------------------
class Booking(Base):
    __tablename__ = "bookings"
    id = Column(Integer, primary_key=True, index=True)
    service = Column(String, nullable=False)
    start_dt = Column(DateTime, nullable=False, index=True)
    end_dt = Column(DateTime, nullable=False)
    name = Column(String, nullable=False)
    contact = Column(String, nullable=False)
    status = Column(String, nullable=False, default="booked")  # booked/canceled
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("start_dt", name="uq_booking_start_dt"),
    )


Base.metadata.create_all(bind=engine)


# ----------------------------
# API Schemas (Agenda)
# ----------------------------
class AvailabilityRequest(BaseModel):
    day: date
    service: str = Field(..., description="Nombre del servicio, ej: Consulta")


class Slot(BaseModel):
    start: datetime
    end: datetime


class AvailabilityResponse(BaseModel):
    slots: List[Slot]


class BookRequest(BaseModel):
    service: str
    start: datetime
    name: str
    contact: str


class BookResponse(BaseModel):
    booking_id: int
    service: str
    start: datetime
    end: datetime
    name: str
    contact: str
    status: str


class CancelRequest(BaseModel):
    booking_id: int
    contact: str


class CancelResponse(BaseModel):
    ok: bool


# ----------------------------
# API Schemas (LLM / Chat)
# ----------------------------
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, str]] = None
    session_id: Optional[str] = None # opcional por si luego guardás estado


class ChatResponse(BaseModel):
    intent: str  # book|cancel|reschedule|unknown
    service: Optional[str] = None
    day: Optional[str] = None         # YYYY-MM-DD si el usuario lo dio
    time_hint: Optional[str] = None   # "mañana", "tarde", "11:30"
    name: Optional[str] = None
    contact: Optional[str] = None
    missing: List[str] = []
    reply: str


# ----------------------------
# Helpers (Agenda)
# ----------------------------
def get_service_duration(service: str) -> int:
    # normalizamos capitalización mínima
    s = service.strip()
    # si el usuario manda "corte" lo convertimos a "Corte" para mapear
    s_norm = s[:1].upper() + s[1:].lower() if s else s
    return SERVICE_DURATION.get(s_norm, 30)


def iter_day_slots(day: date, duration_min: int) -> List[Slot]:
    slots = []
    start_dt = datetime.combine(day, WORK_START)
    end_limit = datetime.combine(day, WORK_END)
    last_start = end_limit - timedelta(minutes=duration_min)

    cur = start_dt
    while cur <= last_start:
        s = cur
        e = cur + timedelta(minutes=duration_min)
        slots.append(Slot(start=s, end=e))
        cur += timedelta(minutes=SLOT_MINUTES)
    return slots


def overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end

def normalize_service(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    return s[:1].upper() + s[1:].lower()

def parse_relative_day(text: str) -> Optional[date]:
    t = (text or "").lower()
    today = date.today()
    if "pasado mañana" in t or "pasadomañana" in t:
        return today + timedelta(days=2)
    if "mañana" in t:
        return today + timedelta(days=1)
    if "hoy" in t:
        return today
    return None

def free_slots_for_day(day_obj: date, service: str, time_hint: Optional[str] = None) -> List[Slot]:
    duration = get_service_duration(service)
    all_slots = iter_day_slots(day_obj, duration)

    with SessionLocal() as db:
        q = select(Booking).where(
            Booking.status == "booked",
            Booking.start_dt >= datetime.combine(day_obj, time(0, 0)),
            Booking.start_dt < datetime.combine(day_obj + timedelta(days=1), time(0, 0)),
        )
        bookings = db.execute(q).scalars().all()

    free: List[Slot] = []
    for sl in all_slots:
        blocked = False
        for b in bookings:
            b_start = b.start_dt - timedelta(minutes=BUFFER_MINUTES)
            b_end = b.end_dt + timedelta(minutes=BUFFER_MINUTES)
            if overlaps(sl.start, sl.end, b_start, b_end):
                blocked = True
                break
        if not blocked:
            free.append(sl)

    if time_hint:
        th = time_hint.lower()
        if "tarde" in th:
            free = [s for s in free if s.start.time() >= time(12, 0)]
        if "mañana" in th:
            free = [s for s in free if s.start.time() < time(12, 0)]

    return free




# ----------------------------
# Helpers (Ollama Cloud)
# ----------------------------
def ollama_cloud_json(user_message: str) -> dict:
    """
    Pide al modelo que devuelva SOLO JSON (sin texto extra).
    """
    if not OLLAMA_API_KEY:
        raise RuntimeError("Falta OLLAMA_API_KEY en variables de entorno.")

    system = """
Sos un asistente de turnos para un salón de belleza. Tu tarea es interpretar el mensaje del usuario
y devolver SIEMPRE un JSON estricto (sin texto extra, sin markdown) con esta forma exacta:

{
  "intent": "book" | "unknown",
  "service": string | null,
  "day": "YYYY-MM-DD" | null,
  "time_hint": "mañana" | "tarde" | null,
  "missing": string[],
  "reply": string
}

REGLAS CRITICAS (no romper):
- NO confirmes turnos. NO inventes horarios.
- NUNCA pidas nombre ni contacto en "reply". El sistema los pedirá después de que el usuario elija un horario.
- NUNCA pidas una hora exacta. El sistema muestra horarios disponibles (slots) y el usuario elige uno.
- Servicios permitidos: "Corte", "Color", "Peinado", "Uñas".
  - Si el usuario pide algo distinto, elegí el más cercano o poné service=null si no está claro.
- day:
  - SOLO si el usuario dio una fecha exacta en formato YYYY-MM-DD.
  - Si dice "hoy/mañana/pasado mañana" o un día de semana ("domingo", "viernes"), day=null.
- time_hint:
  - "tarde" si menciona tarde
  - "mañana" si menciona mañana
  - si no, null
- missing: SOLO puede contener lo necesario para mostrar horarios:
  - si falta service -> ["service"]
  - si falta day -> ["day"]
  - si falta ambos -> ["service","day"]
  - NO incluyas "name", "contact" ni "time" en missing.
- reply:
  - Si falta service: pedir el servicio (Corte/Color/Peinado/Uñas).
  - Si falta day: pedir la fecha en YYYY-MM-DD (dar ejemplo).
  - Si están service y day: responder algo corto tipo: "Perfecto, te muestro horarios disponibles 👇"
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    r = requests.post(
        f"{OLLAMA_API_BASE}/chat",
        headers={
            "Authorization": f"Bearer {OLLAMA_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if not r.ok:
        raise RuntimeError(f"Ollama Cloud HTTP {r.status_code}: {r.text}")

    data = r.json()
    content = data.get("message", {}).get("content", "")

    return json.loads(content)



# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Agenda Turnos API (SQLite + Ollama Cloud)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://salon-frontend-zujz.onrender.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "db": str(DB_PATH), "ollama_model": OLLAMA_MODEL}


# ----------------------------
# LLM endpoint
# ----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        j = ollama_cloud_json(req.message)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="El modelo no devolvió JSON válido.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error Ollama Cloud: {e}")

    return ChatResponse(
        intent=(j.get("intent") or "unknown"),
        service=j.get("service"),
        day=j.get("day"),
        time_hint=j.get("time_hint"),
        name=j.get("name"),
        contact=j.get("contact"),
        missing=j.get("missing") or [],
        reply=j.get("reply") or "¿Qué servicio y para qué día necesitás?"
    )


# ----------------------------
# Agenda endpoints
# ----------------------------
@app.post("/availability", response_model=AvailabilityResponse)
def availability(req: AvailabilityRequest):
    duration = get_service_duration(req.service)
    slots = iter_day_slots(req.day, duration)

    with SessionLocal() as db:
        q = select(Booking).where(
            Booking.status == "booked",
            Booking.start_dt >= datetime.combine(req.day, time(0, 0)),
            Booking.start_dt < datetime.combine(req.day + timedelta(days=1), time(0, 0)),
        )
        bookings = db.execute(q).scalars().all()

    free = []
    for sl in slots:
        blocked = False
        for b in bookings:
            b_start = b.start_dt - timedelta(minutes=BUFFER_MINUTES)
            b_end = b.end_dt + timedelta(minutes=BUFFER_MINUTES)
            if overlaps(sl.start, sl.end, b_start, b_end):
                blocked = True
                break
        if not blocked:
            free.append(sl)

    return AvailabilityResponse(slots=free)


@app.post("/book", response_model=BookResponse)
def book(req: BookRequest):
    duration = get_service_duration(req.service)
    start_dt = req.start.replace(second=0, microsecond=0)
    end_dt = start_dt + timedelta(minutes=duration)

    now = datetime.now()
    if start_dt < now:
        raise HTTPException(status_code=400, detail="No se pueden reservar turnos en fechas pasadas.")
    if start_dt > now + timedelta(days=90):
        raise HTTPException(status_code=400, detail="No se pueden reservar turnos con tanta anticipación.")

    if not (WORK_START <= start_dt.time() < WORK_END):
        raise HTTPException(status_code=400, detail="Horario fuera del rango laboral.")
    if end_dt.time() > WORK_END:
        raise HTTPException(status_code=400, detail="El servicio excede el horario laboral.")

    with SessionLocal() as db:
        q = select(Booking).where(Booking.status == "booked")
        bookings = db.execute(q).scalars().all()

        for b in bookings:
            b_start = b.start_dt - timedelta(minutes=BUFFER_MINUTES)
            b_end = b.end_dt + timedelta(minutes=BUFFER_MINUTES)
            if overlaps(start_dt, end_dt, b_start, b_end):
                raise HTTPException(status_code=409, detail="Ese horario ya está ocupado.")

        b = Booking(
            service=req.service.strip(),
            start_dt=start_dt,
            end_dt=end_dt,
            name=req.name.strip(),
            contact=req.contact.strip(),
            status="booked",
        )

        db.add(b)
        db.commit()
        db.refresh(b)

        return BookResponse(
            booking_id=b.id,
            service=b.service,
            start=b.start_dt,
            end=b.end_dt,
            name=b.name,
            contact=b.contact,
            status=b.status,
        )


@app.post("/cancel", response_model=CancelResponse)
def cancel(req: CancelRequest):
    with SessionLocal() as db:
        b = db.get(Booking, req.booking_id)
        if not b or b.status != "booked":
            raise HTTPException(status_code=404, detail="Turno no encontrado o ya cancelado.")

        if b.contact.strip().lower() != req.contact.strip().lower():
            raise HTTPException(status_code=403, detail="Contacto no coincide.")

        b.status = "canceled"
        db.commit()

    return CancelResponse(ok=True)


@app.get("/bookings")
def list_bookings(day: Optional[date] = None, status: Optional[str] = None):
    with SessionLocal() as db:
        q = select(Booking)

        if day:
            start = datetime.combine(day, time(0, 0))
            end = datetime.combine(day + timedelta(days=1), time(0, 0))
            q = q.where(Booking.start_dt >= start, Booking.start_dt < end)

        if status:
            q = q.where(Booking.status == status)

        q = q.order_by(Booking.start_dt.asc())
        rows = db.execute(q).scalars().all()

        return [
            {
                "id": b.id,
                "service": b.service,
                "start": b.start_dt.isoformat(),
                "end": b.end_dt.isoformat(),
                "name": b.name,
                "contact": b.contact,
                "status": b.status,
                "created_at": b.created_at.isoformat(),
            }
            for b in rows
        ]


@app.post("/admin/cancel")
def admin_cancel(booking_id: int):
    with SessionLocal() as db:
        b = db.get(Booking, booking_id)
        if not b or b.status != "booked":
            raise HTTPException(status_code=404, detail="Turno no encontrado o ya cancelado.")
        b.status = "canceled"
        db.commit()
    return {"ok": True}


class AgentResponse(BaseModel):
    intent: str
    reply: str
    missing: List[str] = []
    service: Optional[str] = None
    day: Optional[str] = None
    time_hint: Optional[str] = None
    slots: Optional[List[Slot]] = None
    suggestions: Optional[List[Dict[str, str]]] = None  # [{day,label}]



@app.post("/agent", response_model=AgentResponse)
def agent(req: ChatRequest):
    # 1) Interpretar con LLM
    try:
        j = ollama_cloud_json(req.message)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="El modelo no devolvió JSON válido.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error Ollama Cloud: {e}")

    # ✅ PASO 5: sesión + contexto (memoria + lo que viene del front)
    sid = req.session_id or "default"
    ctx_mem = SESSION_CTX.get(sid, {})
    ctx_in = req.context or {}
    ctx = {**ctx_mem, **ctx_in}

    intent = (j.get("intent") or "unknown").strip()

    service = normalize_service(j.get("service"))
    day_str = j.get("day")  # YYYY-MM-DD o null
    time_hint = j.get("time_hint")
    missing = j.get("missing") or []
    reply = j.get("reply") or "¿Qué necesitás?"

    # ✅ Fallbacks desde sesión/contexto
    if intent == "book" and (not service) and ctx.get("service"):
        service = normalize_service(ctx.get("service"))

    if intent == "book" and (not day_str) and ctx.get("day"):
        day_str = ctx.get("day")

    if intent == "book" and (not time_hint) and ctx.get("time_hint"):
        time_hint = ctx.get("time_hint")

    # 2) Resolver "hoy/mañana/pasado mañana"
    if intent == "book" and not day_str:
        rel = parse_relative_day(req.message)
        if rel:
            day_str = rel.isoformat()
            missing = [m for m in missing if m != "day"]

    # 2b) Fecha sola YYYY-MM-DD
    import re
    if intent == "book" and not day_str:
        m = re.fullmatch(r"\s*(\d{4}-\d{2}-\d{2})\s*", req.message)
        if m:
            day_str = m.group(1)
            missing = [x for x in missing if x != "day"]

    # 3) Si tengo service + day => slots reales
    if intent == "book" and service and day_str:
        try:
            day_obj = date.fromisoformat(day_str)
        except Exception:
            save_session_ctx(sid, service, None, time_hint)
            return AgentResponse(
                intent=intent,
                reply="La fecha debe ser YYYY-MM-DD. Ej: 2026-01-20",
                missing=["day"],
                service=service,
                day=None,
                time_hint=time_hint,
                slots=None,
                suggestions=None,
            )

        free = free_slots_for_day(day_obj, service, time_hint=time_hint)

        # 4) Si no hay, sugerimos próximos 3 días
        if not free:
            suggestions: List[Dict[str, str]] = []
            for i in range(1, 8):
                d = day_obj + timedelta(days=i)
                free_next = free_slots_for_day(d, service, time_hint=time_hint)
                if free_next:
                    suggestions.append({"day": d.isoformat(), "label": d.strftime("%A %d/%m")})
                if len(suggestions) >= 3:
                    break

            save_session_ctx(sid, service, day_str, time_hint)

            if suggestions:
                return AgentResponse(
                    intent=intent,
                    reply="No hay horarios ese día. Te propongo estas fechas con disponibilidad:",
                    missing=[],
                    service=service,
                    day=day_str,
                    time_hint=time_hint,
                    slots=[],
                    suggestions=suggestions,
                )

            return AgentResponse(
                intent=intent,
                reply="No hay horarios disponibles. ¿Querés probar otro día?",
                missing=["day"],
                service=service,
                day=day_str,
                time_hint=time_hint,
                slots=[],
                suggestions=None,
            )

        save_session_ctx(sid, service, day_str, time_hint)
        return AgentResponse(
            intent=intent,
            reply="Estos son los horarios disponibles. Elegí uno:",
            missing=[],
            service=service,
            day=day_str,
            time_hint=time_hint,
            slots=free[:8],
            suggestions=None,
        )

    # Si quiere reservar pero falta la fecha
    if intent == "book" and service and not day_str:
        save_session_ctx(sid, service, None, time_hint)  # ✅ CLAVE
        return AgentResponse(
            intent="book",
            reply="Perfecto 💇‍♀️ ¿Para qué fecha lo querés? (YYYY-MM-DD)\nEj: 2026-01-20",
            missing=["day"],
            service=service,
            day=None,
            time_hint=time_hint,
            slots=None,
            suggestions=None,
        )

    # Caso general
    save_session_ctx(sid, service, day_str, time_hint)
    return AgentResponse(
        intent=intent,
        reply=reply,
        missing=missing,
        service=service,
        day=day_str,
        time_hint=time_hint,
        slots=None,
        suggestions=None,
    )

