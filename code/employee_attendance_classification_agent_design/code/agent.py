try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import logging
import asyncio
import time as _time
from typing import Any, Dict, Optional, List, Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator, ConfigDict
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from dateutil import parser as date_parser
from datetime import datetime, timedelta
from functools import wraps

# Observability wrappers (trace_step, trace_step_sync, etc.) are injected by runtime

# Logging configuration
logger = logging.getLogger("attendance_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# FastAPI app
app = FastAPI(
    title="Employee Attendance Classification Agent",
    description="Classifies employee attendance using check-in logs, leave data, shift rules, and holiday calendars.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration Management ---

class Config:
    """Configuration loader for environment variables."""
    @staticmethod
    def get_azure_openai_key() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_API_KEY")

    @staticmethod
    def get_azure_openai_endpoint() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_ENDPOINT")

    @staticmethod
    def get_azure_openai_deployment() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_DEPLOYMENT")

    @staticmethod
    def get_db_url() -> str:
        return os.getenv("ATTENDANCE_AGENT_DB_URL", "sqlite:///./attendance_agent.db")

    @staticmethod
    @trace_agent(agent_name='Employee Attendance Classification Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_llm_config() -> None:
        # Only validate when LLM is actually called
        missing = []
        if not Config.get_azure_openai_key():
            missing.append("AZURE_OPENAI_API_KEY")
        if not Config.get_azure_openai_endpoint():
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not Config.get_azure_openai_deployment():
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if missing:
            raise ValueError(f"Missing Azure OpenAI config: {', '.join(missing)}")

# --- Persistence Layer (SQLAlchemy ORM) ---

Base = declarative_base()

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(64), nullable=False)
    date = Column(String(16), nullable=False)
    status = Column(String(32), nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(128), nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database session factory
engine = create_engine(Config.get_db_url(), connect_args={"check_same_thread": False} if "sqlite" in Config.get_db_url() else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# --- Pydantic Models (Presentation Layer) ---

class AttendanceInput(BaseModel):
    employee_id: str = Field(..., min_length=1, max_length=64)
    checkin_time: Optional[str] = Field(None, description="Check-in time in ISO8601 or HH:MM format")
    shift_start: str = Field(..., description="Shift start time in ISO8601 or HH:MM format")
    grace_period: Optional[int] = Field(10, description="Grace period in minutes")
    half_day_cutoff: Optional[str] = Field(None, description="Half day cutoff time in ISO8601 or HH:MM format")
    leave_status: Optional[str] = Field("None", description="Leave status: None, Approved, Pending")
    holiday_status: Optional[str] = Field("None", description="Holiday status: None, True")
    date: Optional[str] = Field(None, description="Date of attendance in YYYY-MM-DD")
    additional_info: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("employee_id cannot be empty")
        return v

    @field_validator("checkin_time", "shift_start", "half_day_cutoff")
    @classmethod
    def validate_time_fields(cls, v):
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        # Accept both HH:MM and ISO8601
        try:
            _obs_t0 = _time.time()
            _ = date_parser.parse(v)
            try:
                trace_tool_call(
                    tool_name='date_parser.parse',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(_)[:200] if _ is not None else None,
                    status="success",
                )
            except Exception:
                pass
        except Exception:
            raise ValueError(f"Invalid time format: {v}")
        return v

    @field_validator("leave_status", "holiday_status")
    @classmethod
    def validate_status_fields(cls, v):
        if v is None:
            return "None"
        return v.strip().capitalize()

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        if v is None:
            return datetime.utcnow().strftime("%Y-%m-%d")
        try:
            _obs_t0 = _time.time()
            dt = date_parser.parse(v)
            try:
                trace_tool_call(
                    tool_name='date_parser.parse',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(dt)[:200] if dt is not None else None,
                    status="success",
                )
            except Exception:
                pass
            return dt.strftime("%Y-%m-%d")
        except Exception:
            raise ValueError("Invalid date format")
    
    @model_validator(mode="after")
    def check_content_length(self):
        # Limit input size
        total = sum(len(str(getattr(self, f))) for f in self.model_fields)
        if total > 50000:
            raise ValueError("Input too large (max 50,000 characters)")
        return self

class AttendanceOutput(BaseModel):
    success: bool
    attendance_status: Optional[str] = None
    message: Optional[str] = None
    error_type: Optional[str] = None
    error_description: Optional[str] = None
    fixing_tips: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# --- Persistence Manager ---

class PersistenceManager:
    """Handles persistence of attendance records and audit logs."""
    def __init__(self):
        self.Session = SessionLocal

    def save_attendance_record(self, record: Dict[str, Any]) -> None:
        """Persist attendance record."""
        with trace_step_sync(
            "save_attendance_record", step_type="process",
            decision_summary="Persist attendance record to DB",
            output_fn=lambda r: f"employee_id={record.get('employee_id')}, status={record.get('status')}"
        ) as step:
            session = self.Session()
            try:
                rec = AttendanceRecord(
                    employee_id=record["employee_id"],
                    date=record["date"],
                    status=record["status"],
                    details=record.get("details")
                )
                session.add(rec)
                session.commit()
                step.capture({"success": True})
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to save attendance record: {e}")
                raise
            finally:
                session.close()

    def save_audit_log(self, log: Dict[str, Any]) -> None:
        """Persist audit log."""
        with trace_step_sync(
            "save_audit_log", step_type="process",
            decision_summary="Persist audit log to DB",
            output_fn=lambda r: f"action={log.get('action')}"
        ) as step:
            session = self.Session()
            try:
                entry = AuditLog(
                    action=log["action"],
                    details=log.get("details")
                )
                session.add(entry)
                session.commit()
                step.capture({"success": True})
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to save audit log: {e}")
                raise
            finally:
                session.close()

# --- Audit Logger ---

class AuditLogger:
    """Logs all classification actions for compliance and transparency."""
    def __init__(self, persistence_manager: PersistenceManager):
        self.persistence_manager = persistence_manager

    def log_action(self, action_details: Dict[str, Any]) -> None:
        """Log attendance classification actions for audit."""
        with trace_step_sync(
            "log_action", step_type="process",
            decision_summary="Log action for audit",
            output_fn=lambda r: f"action={action_details.get('action')}"
        ) as step:
            try:
                self.persistence_manager.save_audit_log(action_details)
                step.capture({"success": True})
            except Exception as e:
                logger.error(f"Failed to log action: {e}")
                # Escalate if audit log cannot be persisted
                raise

# --- Notification Manager ---

class NotificationManager:
    """Notifies HR of exceptions and errors."""
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def notify_hr(self, exception_details: Dict[str, Any]) -> None:
        """Notify HR of exceptions or errors."""
        with trace_step_sync(
            "notify_hr", step_type="process",
            decision_summary="Notify HR of exception",
            output_fn=lambda r: f"exception={exception_details.get('error_type')}"
        ) as step:
            try:
                # In production, send email or dashboard notification
                logger.warning(f"HR Notification: {exception_details}")
                self.audit_logger.log_action({
                    "action": "HR_NOTIFICATION",
                    "details": str(exception_details)
                })
                step.capture({"success": True})
            except Exception as e:
                logger.error(f"Failed to notify HR: {e}")
                # Escalate if notification undelivered
                raise

# --- Tool Integration Manager ---

class ToolIntegrationManager:
    """Manages calls to tool integrations (policy validator, leave processor, holiday checker, check-in analyzer)."""
    def __init__(self):
        # In real implementation, inject tool clients here
        pass

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call tool integrations for policy validation, leave processing, holiday checking, check-in analysis."""
        async with trace_step(
            f"invoke_tool_{tool_name}", step_type="tool_call",
            decision_summary=f"Invoke tool: {tool_name}",
            output_fn=lambda r: f"result={str(r)[:100]}"
        ) as step:
            # Simulate tool integration
            await asyncio.sleep(0.05)
            result = {"tool": tool_name, "status": "success", "params": params}
            step.capture(result)
            return result

# --- Attendance Input Processor ---

class AttendanceInputProcessor:
    """Validates and preprocesses incoming attendance data."""
    def __init__(self, tool_manager: ToolIntegrationManager):
        self.tool_manager = tool_manager

    async def validate_input(self, data: AttendanceInput) -> AttendanceInput:
        """Validate and preprocess attendance input data."""
        async with trace_step(
            "validate_input", step_type="parse",
            decision_summary="Validate and preprocess input",
            output_fn=lambda r: f"employee_id={r.employee_id}"
        ) as step:
            # Check for required fields
            if not data.employee_id or not data.shift_start:
                raise ValueError("Missing required fields: employee_id, shift_start")
            # Validate time fields
            try:
                _obs_t0 = _time.time()
                _ = date_parser.parse(data.shift_start)
                try:
                    trace_tool_call(
                        tool_name='date_parser.parse',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(_)[:200] if _ is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                if data.checkin_time:
                    _obs_t0 = _time.time()
                    _ = date_parser.parse(data.checkin_time)
                    try:
                        trace_tool_call(
                            tool_name='date_parser.parse',
                            latency_ms=int((_time.time() - _obs_t0) * 1000),
                            output=str(_)[:200] if _ is not None else None,
                            status="success",
                        )
                    except Exception:
                        pass
                if data.half_day_cutoff:
                    _obs_t0 = _time.time()
                    _ = date_parser.parse(data.half_day_cutoff)
                    try:
                        trace_tool_call(
                            tool_name='date_parser.parse',
                            latency_ms=int((_time.time() - _obs_t0) * 1000),
                            output=str(_)[:200] if _ is not None else None,
                            status="success",
                        )
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Invalid time format: {e}")
                raise ValueError("Invalid time format in input")
            # Optionally, check holiday calendar or policy via tool integration
            # (simulate for now)
            step.capture(data)
            return data

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_input(self, data: AttendanceInput) -> Dict[str, Any]:
        """Validate and preprocess attendance input data."""
        async with trace_step(
            "process_input", step_type="process",
            decision_summary="Process and normalize input",
            output_fn=lambda r: f"employee_id={r.get('employee_id')}"
        ) as step:
            validated = await self.validate_input(data)
            # Normalize fields
            result = validated.model_dump()
            # Add derived fields if needed
            step.capture(result)
            return result

# --- Attendance Classifier (Domain Layer) ---

class AttendanceClassifier:
    """Applies strict policy order and business rules to classify attendance status."""
    def __init__(self):
        # Decision tables and rule sets can be loaded from config or code
        self.rules = [
            # Strict policy order
            self._rule_holiday,
            self._rule_leave,
            self._rule_present,
            self._rule_late_present,
            self._rule_half_day,
            self._rule_absent
        ]

    async def classify_attendance(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply business rules to classify attendance status."""
        async with trace_step(
            "classify_attendance", step_type="process",
            decision_summary="Apply business rules for attendance classification",
            output_fn=lambda r: f"status={r.get('attendance_status')}"
        ) as step:
            for rule in self.rules:
                result = rule(employee_data)
                if result is not None:
                    step.capture(result)
                    return result
            # If no rule matches, return error
            logger.error("No classification rule matched")
            raise ValueError("No classification rule matched")

    def _rule_present(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Present: check-in on time, no leave, not holiday
        if (
            d.get("leave_status") == "None"
            and d.get("holiday_status") == "None"
            and self._checkin_status(d) == "OnTime"
        ):
            return {"attendance_status": "Present"}
        return None

    def _rule_late_present(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Late Present: check-in after shift start but within grace period
        if (
            d.get("leave_status") == "None"
            and d.get("holiday_status") == "None"
            and self._checkin_status(d) == "LateWithinGrace"
        ):
            return {"attendance_status": "Late Present"}
        return None

    def _rule_half_day(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Half Day: check-in after grace period but before half-day cutoff
        if (
            d.get("leave_status") == "None"
            and d.get("holiday_status") == "None"
            and self._checkin_status(d) == "LateBeyondGraceBeforeHalfDay"
        ):
            return {"attendance_status": "Half Day"}
        return None

    def _rule_leave(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Leave: leave approved for the day
        if d.get("leave_status") == "Approved" and d.get("holiday_status") == "None":
            return {"attendance_status": "Leave"}
        return None

    def _rule_absent(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Absent: no check-in, no leave, not holiday
        if (
            not d.get("checkin_time")
            and d.get("leave_status") == "None"
            and d.get("holiday_status") == "None"
        ):
            return {"attendance_status": "Absent"}
        return None

    def _rule_holiday(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Holiday: day is marked as holiday
        if d.get("holiday_status") == "True":
            return {"attendance_status": "Holiday"}
        return None

    def _checkin_status(self, d: Dict[str, Any]) -> str:
        """Derive check-in status: OnTime, LateWithinGrace, LateBeyondGraceBeforeHalfDay, None"""
        try:
            if not d.get("checkin_time"):
                return "None"
            _obs_t0 = _time.time()
            checkin = date_parser.parse(d["checkin_time"])
            try:
                trace_tool_call(
                    tool_name='date_parser.parse',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(checkin)[:200] if checkin is not None else None,
                    status="success",
                )
            except Exception:
                pass
            _obs_t0 = _time.time()
            shift_start = date_parser.parse(d["shift_start"])
            try:
                trace_tool_call(
                    tool_name='date_parser.parse',
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                    output=str(shift_start)[:200] if shift_start is not None else None,
                    status="success",
                )
            except Exception:
                pass
            grace_period = int(d.get("grace_period", 10))
            half_day_cutoff = (
                date_parser.parse(d["half_day_cutoff"])
                if d.get("half_day_cutoff")
                else shift_start + timedelta(hours=4)
            )
            if checkin <= shift_start:
                return "OnTime"
            elif shift_start < checkin <= shift_start + timedelta(minutes=grace_period):
                return "LateWithinGrace"
            elif shift_start + timedelta(minutes=grace_period) < checkin <= half_day_cutoff:
                return "LateBeyondGraceBeforeHalfDay"
            else:
                return "None"
        except Exception as e:
            logger.error(f"Error in checkin status calculation: {e}")
            return "None"

# --- LLM Interaction Handler ---

class LLMInteractionHandler:
    """Handles prompt construction, LLM calls, and response parsing."""
    def __init__(self, model: str, temperature: float, max_tokens: int, system_prompt: str, deployment: str):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.deployment = deployment
        self._client = None

    def _get_client(self):
        # Lazy import and client creation
        import openai
        Config.validate_llm_config()
        if self._client is None:
            self._client = openai.AsyncAzureOpenAI(
                api_key=Config.get_azure_openai_key(),
                azure_endpoint=Config.get_azure_openai_endpoint(),
                azure_deployment=self.deployment,
                api_version="2024-02-15-preview"
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def generate_prompt(self, data: Dict[str, Any]) -> str:
        """Construct LLM prompt for attendance classification."""
        with trace_step_sync(
            "generate_prompt", step_type="plan",
            decision_summary="Construct LLM prompt",
            output_fn=lambda r: f"prompt_length={len(r)}"
        ) as step:
            # Use few-shot examples and user prompt template
            prompt = (
                f"{self.system_prompt}\n"
                "Examples:\n"
                "Employee A checked in at 8:55 AM, shift starts at 9:00 AM, no leave, not a holiday. Employee A is classified as Present.\n"
                "Employee B checked in at 9:15 AM, shift starts at 9:00 AM, grace period is 10 minutes, no leave, not a holiday. Employee B is classified as Late Present.\n"
                "Employee C has approved leave for today. Employee C is classified as Leave.\n"
                "\n"
                f"Today's data:\n"
                f"Employee ID: {data.get('employee_id')}\n"
                f"Check-in Time: {data.get('checkin_time')}\n"
                f"Shift Start: {data.get('shift_start')}\n"
                f"Grace Period: {data.get('grace_period')}\n"
                f"Half Day Cutoff: {data.get('half_day_cutoff')}\n"
                f"Leave Status: {data.get('leave_status')}\n"
                f"Holiday Status: {data.get('holiday_status')}\n"
                f"Date: {data.get('date')}\n"
                "Classify the employee's attendance status for today."
            )
            step.capture(prompt)
            return prompt

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def call_llm(self, prompt: str) -> str:
        """Invoke LLM for classification."""
        async with trace_step(
            "call_llm", step_type="llm_call",
            decision_summary="Call Azure OpenAI GPT-4.1 for classification",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            client = self._get_client()
            _t0 = _time.time()
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=self.model,
                        prompt_tokens=response.usage.prompt_tokens if hasattr(response, "usage") else None,
                        completion_tokens=response.usage.completion_tokens if hasattr(response, "usage") else None,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else ""
                    )
                except Exception:
                    pass
                step.capture(content)
                return content
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def parse_llm_response(self, response: str) -> str:
        """Parse LLM response for attendance status."""
        with trace_step_sync(
            "parse_llm_response", step_type="parse",
            decision_summary="Parse LLM response for attendance status",
            output_fn=lambda r: f"attendance_status={r}"
        ) as step:
            # Try to extract attendance status from response
            try:
                # Look for "classified as X" in response
                import re
                match = re.search(r"classified as ([A-Za-z ]+)[\.\n]", response)
                if match:
                    status = match.group(1).strip()
                else:
                    # Fallback: look for status keywords
                    for s in ["Present", "Late Present", "Half Day", "Leave", "Absent", "Holiday"]:
                        if s.lower() in response.lower():
                            status = s
                            break
                    else:
                        status = "Unknown"
                step.capture(status)
                return status
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")
                raise

# --- Main Agent Class ---

class AttendanceClassificationAgent:
    """Main agent for employee attendance classification."""
    def __init__(self):
        self.persistence_manager = PersistenceManager()
        self.audit_logger = AuditLogger(self.persistence_manager)
        self.notification_manager = NotificationManager(self.audit_logger)
        self.tool_manager = ToolIntegrationManager()
        self.input_processor = AttendanceInputProcessor(self.tool_manager)
        self.classifier = AttendanceClassifier()
        self.llm_handler = LLMInteractionHandler(
            model="gpt-4.1",
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are a professional attendance classification agent. Your task is to record and classify daily employee attendance using check-in logs, leave data, shift rules, and holiday calendars. Apply strict policy order and communicate results clearly, ensuring compliance and data privacy.",
            deployment=Config.get_azure_openai_deployment() or ""
        )

    async def classify(self, data: AttendanceInput) -> AttendanceOutput:
        """Classify attendance for a given input."""
        async with trace_step(
            "agent_classify", step_type="final",
            decision_summary="End-to-end attendance classification",
            output_fn=lambda r: f"success={r.success}, status={r.attendance_status}"
        ) as step:
            try:
                # Step 1: Validate and preprocess input
                processed = await self.input_processor.process_input(data)
                # Step 2: Try strict business rules first
                try:
                    rule_result = await self.classifier.classify_attendance(processed)
                    attendance_status = rule_result["attendance_status"]
                    message = f"Attendance classified as {attendance_status} by business rules."
                except Exception as e:
                    logger.warning(f"Business rule classification failed: {e}")
                    # Step 3: Fallback to LLM if rules fail
                    prompt = self.llm_handler.generate_prompt(processed)
                    llm_response = await self.llm_handler.call_llm(prompt)
                    attendance_status = self.llm_handler.parse_llm_response(llm_response)
                    message = f"Attendance classified as {attendance_status} by LLM."
                # Step 4: Persist attendance record
                record = {
                    "employee_id": processed["employee_id"],
                    "date": processed["date"],
                    "status": attendance_status,
                    "details": message
                }
                self.persistence_manager.save_attendance_record(record)
                # Step 5: Audit log
                self.audit_logger.log_action({
                    "action": "ATTENDANCE_CLASSIFICATION",
                    "details": f"{record}"
                })
                result = AttendanceOutput(
                    success=True,
                    attendance_status=attendance_status,
                    message=message,
                    details=record
                )
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                # Notify HR of exception
                self.notification_manager.notify_hr({
                    "error_type": type(e).__name__,
                    "error_description": str(e)
                })
                result = AttendanceOutput(
                    success=False,
                    error_type=type(e).__name__,
                    error_description=str(e),
                    fixing_tips="Check input data for required fields and correct formats. If the error persists, contact system administrator."
                )
                step.capture(result)
                return result

# --- FastAPI Endpoints (Presentation Layer) ---

agent = AttendanceClassificationAgent()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "ValidationError",
            "error_description": str(exc),
            "fixing_tips": "Check your JSON formatting, required fields, and value types."
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "HTTPException",
            "error_description": exc.detail,
            "fixing_tips": "Check your request and try again."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_type": type(exc).__name__,
            "error_description": str(exc),
            "fixing_tips": "Check your input and try again. If the error persists, contact support."
        }
    )

@app.post("/classify", response_model=AttendanceOutput)
async def classify_attendance_endpoint(input_data: AttendanceInput):
    """
    Classify employee attendance for a given day.
    """
    try:
        _obs_t0 = _time.time()
        result = await agent.classify(input_data)
        try:
            trace_tool_call(
                tool_name='agent.classify',
                latency_ms=int((_time.time() - _obs_t0) * 1000),
                output=str(result)[:200] if result is not None else None,
                status="success",
            )
        except Exception:
            pass
        return result
    except ValidationError as ve:
        logger.warning(f"Input validation failed: {ve}")
        raise HTTPException(
            status_code=422,
            detail=f"Input validation failed: {ve}"
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {e}"
        )

@app.post("/health")
async def health_check():
    return {"success": True, "status": "ok"}

# --- Main block ---



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting Employee Attendance Classification Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())