from datetime import datetime, date, time, timedelta
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, Column, Integer, String, DateTime, UniqueConstraint, select
from sqlalchemy.orm import declarative_base, sessionmaker

from sqlalchemy import desc

# ----------------------------
# Config
# ----------------------------
DB_URL = "sqlite:///./agenda.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# Reglas base (MVP)
WORK_START = time(9, 0)
WORK_END = time(18, 0)
SLOT_MINUTES = 30
BUFFER_MINUTES = 0  # si querés, poné 5 o 10
# Duración por servicio (min). Podés extenderlo.
SERVICE_DURATION = {
    "Consulta": 30,
    "Control": 30,
    "Corte": 30,
    "Color": 90,
}

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
# API Schemas
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
# Helpers
# ----------------------------
def round_to_slot(dt: datetime) -> datetime:
    # redondea hacia arriba a múltiplo de SLOT_MINUTES
    minutes = (dt.minute // SLOT_MINUTES) * SLOT_MINUTES
    base = dt.replace(minute=minutes, second=0, microsecond=0)
    if base < dt.replace(second=0, microsecond=0):
        base += timedelta(minutes=SLOT_MINUTES)
    return base

def get_service_duration(service: str) -> int:
    return SERVICE_DURATION.get(service.strip(), 30)

def iter_day_slots(day: date, duration_min: int) -> List[Slot]:
    slots = []
    start_dt = datetime.combine(day, WORK_START)
    end_limit = datetime.combine(day, WORK_END)

    # el último inicio posible es end_limit - duration
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

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Agenda Turnos API (SQLite)")

# CORS para tu Next.js local
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

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
            # considerar buffer
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

    # ✅ NUEVO: no permitir turnos en el pasado
    now = datetime.now()
    if start_dt < now:
        raise HTTPException(
            status_code=400,
            detail="No se pueden reservar turnos en fechas pasadas."
        )

    # ✅ (Opcional) límite a futuro: 90 días
    if start_dt > now + timedelta(days=90):
        raise HTTPException(
            status_code=400,
            detail="No se pueden reservar turnos con tanta anticipación."
        )

    # Validar horario laboral
    if not (WORK_START <= start_dt.time() < WORK_END):
        raise HTTPException(status_code=400, detail="Horario fuera del rango laboral.")
    if end_dt.time() > WORK_END:
        raise HTTPException(status_code=400, detail="El servicio excede el horario laboral.")

    with SessionLocal() as db:
        # Chequear solapamiento (solo contra turnos confirmados)
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

        # verificación simple por contacto
        if b.contact.strip().lower() != req.contact.strip().lower():
            raise HTTPException(status_code=403, detail="Contacto no coincide.")

        b.status = "canceled"
        db.commit()

    return CancelResponse(ok=True)


@app.get("/bookings")
def list_bookings(day: Optional[date] = None, status: Optional[str] = None):
    """
    Lista turnos. Filtros opcionales:
    - day=YYYY-MM-DD
    - status=booked|canceled
    """
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
    """
    Cancelación admin (sin validar contacto).
    """
    with SessionLocal() as db:
        b = db.get(Booking, booking_id)
        if not b or b.status != "booked":
            raise HTTPException(status_code=404, detail="Turno no encontrado o ya cancelado.")
        b.status = "canceled"
        db.commit()
    return {"ok": True}

