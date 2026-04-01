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
from typing import List, Dict, Any, Optional, Tuple, Union
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
from dotenv import load_dotenv
import requests
import httpx
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet

# Observability wrappers (trace_step, trace_step_sync, etc.) are injected by the runtime.

# Load environment variables from .env if present
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("employee_assignment_agent")

# Constants for KB API endpoints
BASE_URL = "https://workforce.example.com"
ENDPOINTS = {
    "attendance_status": "/api/attendance/status",
    "employee_skills": "/api/employees/skills",
    "employee_capacity": "/api/employees/capacity",
    "task_priority": "/api/tasks/priority",
    "task_due_date": "/api/tasks/due-date",
    "task_dependencies": "/api/tasks/dependencies",
    "create_assignment": "/api/assignments",
    "utilization_summary": "/api/assignments/utilization-summary",
    "unassigned_tasks": "/api/assignments/unassigned-tasks"
}

# AES-256 encryption for PII masking (key must be 32 url-safe base64-encoded bytes)
def get_fernet():
    key = os.getenv("PII_ENCRYPTION_KEY")
    if not key:
        raise ValueError("PII_ENCRYPTION_KEY not configured")
    _obs_t0 = _time.time()
    _obs_resp = Fernet(key.encode())
    try:
        trace_tool_call(
            tool_name='key.encode',
            latency_ms=int((_time.time() - _obs_t0) * 1000),
            output=str(_obs_resp)[:200] if _obs_resp is not None else None,
            status="success",
        )
    except Exception:
        pass
    return _obs_resp

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_pii(text: str) -> str:
    try:
        f = get_fernet()
        _obs_t0 = _time.time()
        _obs_resp = f.encrypt(text.encode()).decode()
        try:
            trace_tool_call(
                tool_name='f.encrypt',
                latency_ms=int((_time.time() - _obs_t0) * 1000),
                output=str(_obs_resp)[:200] if _obs_resp is not None else None,
                status="success",
            )
        except Exception:
            pass
        return _obs_resp
    except Exception:
        return "***"

# Configuration management
class Config:
    @staticmethod
    def get_oauth_token() -> Optional[str]:
        return os.getenv("WORKFORCE_API_TOKEN")

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
    def validate():
        missing = []
        if not Config.get_oauth_token():
            missing.append("WORKFORCE_API_TOKEN")
        if not Config.get_azure_openai_key():
            missing.append("AZURE_OPENAI_API_KEY")
        if not Config.get_azure_openai_endpoint():
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not Config.get_azure_openai_deployment():
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if not os.getenv("PII_ENCRYPTION_KEY"):
            missing.append("PII_ENCRYPTION_KEY")
        if missing:
            raise RuntimeError(f"Missing required configuration keys: {', '.join(missing)}")

# Pydantic models for input/output
class Employee(BaseModel):
    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    # Optionally: other fields

    @field_validator("id", "name")
    @classmethod
    def not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Must not be empty")
        return v

class Task(BaseModel):
    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    required_skills: List[str] = Field(default_factory=list)
    estimated_effort: float = Field(..., gt=0)
    # Optionally: other fields

    @field_validator("id", "name")
    @classmethod
    def not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Must not be empty")
        return v

    @field_validator("required_skills", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

class AssignmentResult(BaseModel):
    assignments: List[Dict[str, Any]]
    utilization_summary: Dict[str, Any]
    unassigned_tasks: List[Dict[str, Any]]

class UtilizationSummary(BaseModel):
    summary: Dict[str, Any]

class UnassignedTaskReport(BaseModel):
    unassigned_tasks: List[Dict[str, Any]]

class ValidationResult(BaseModel):
    valid: bool
    errors: Optional[List[str]] = None

class AuditLogResult(BaseModel):
    success: bool
    details: Optional[str] = None

class LLMResponse(BaseModel):
    content: str

class AssignmentRequest(BaseModel):
    employees: List[Employee]
    tasks: List[Task]

    @model_validator(mode="after")
    def validate_lists(self):
        if not self.employees:
            raise ValueError("Employee list must not be empty")
        if not self.tasks:
            raise ValueError("Task list must not be empty")
        return self

# Input Processor
class InputProcessor:
    """
    Validates and parses incoming employee and task lists; ensures data completeness.
    """
    @staticmethod
    async def parse_input(data: Dict[str, Any]) -> Tuple[List[Employee], List[Task]]:
        with_trace = "parse_input"
        async with trace_step(
            with_trace, step_type="parse",
            decision_summary="Parse and validate employee/task input",
            output_fn=lambda r: f"employees={len(r[0])}, tasks={len(r[1])}"
        ) as step:
            try:
                req = AssignmentRequest(**data)
                employees = req.employees
                tasks = req.tasks
                step.capture((employees, tasks))
                return employees, tasks
            except ValidationError as ve:
                logger.error(f"Input validation error: {ve}")
                step.capture({"error": str(ve)})
                raise

    @staticmethod
    async def validate_input(data: Dict[str, Any]) -> ValidationResult:
        with_trace = "validate_input"
        async with trace_step(
            with_trace, step_type="parse",
            decision_summary="Validate employee/task input",
            output_fn=lambda r: f"valid={r.valid}"
        ) as step:
            errors = []
            try:
                req = AssignmentRequest(**data)
                for emp in req.employees:
                    if not emp.id or not emp.name:
                        errors.append(f"Employee missing id or name: {emp}")
                for task in req.tasks:
                    if not task.id or not task.name:
                        errors.append(f"Task missing id or name: {task}")
                    if not task.required_skills:
                        errors.append(f"Task {task.id} missing required_skills")
                    if task.estimated_effort <= 0:
                        errors.append(f"Task {task.id} has non-positive estimated_effort")
                valid = len(errors) == 0
                result = ValidationResult(valid=valid, errors=errors if not valid else None)
                step.capture(result)
                return result
            except ValidationError as ve:
                logger.error(f"Validation error: {ve}")
                step.capture({"error": str(ve)})
                return ValidationResult(valid=False, errors=[str(ve)])

# Data Integration Layer
class DataIntegrationClient:
    """
    Fetches real-time attendance, skills, capacity, task info from KB APIs.
    """
    def __init__(self):
        self.session = httpx.AsyncClient(timeout=10.0)
        self.max_retries = 3

    def _get_headers(self) -> Dict[str, str]:
        token = Config.get_oauth_token()
        if not token:
            raise RuntimeError("WORKFORCE_API_TOKEN not configured")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = BASE_URL + endpoint
        headers = self._get_headers()
        retries = 0
        while retries < self.max_retries:
            _t0 = _time.time()
            try:
                resp = await self.session.get(url, headers=headers, params=params)
                try:
                    trace_tool_call(
                        tool_name=f"GET {endpoint}",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(resp.text)[:200],
                        status="success" if resp.status_code == 200 else "error"
                    )
                except Exception:
                    pass
                if resp.status_code == 200:
                    return resp.json()
                else:
                    logger.error(f"API GET {endpoint} failed: {resp.status_code} {resp.text}")
            except Exception as e:
                logger.error(f"API GET {endpoint} exception: {e}")
            retries += 1
            await asyncio.sleep(2 ** retries)
        raise RuntimeError(f"API GET {endpoint} failed after {self.max_retries} retries")

    async def _post(self, endpoint: str, json_data: Dict[str, Any]) -> Any:
        url = BASE_URL + endpoint
        headers = self._get_headers()
        retries = 0
        while retries < self.max_retries:
            _t0 = _time.time()
            try:
                resp = await self.session.post(url, headers=headers, json=json_data)
                try:
                    trace_tool_call(
                        tool_name=f"POST {endpoint}",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(resp.text)[:200],
                        status="success" if resp.status_code in (200, 201) else "error"
                    )
                except Exception:
                    pass
                if resp.status_code in (200, 201):
                    return resp.json()
                else:
                    logger.error(f"API POST {endpoint} failed: {resp.status_code} {resp.text}")
            except Exception as e:
                logger.error(f"API POST {endpoint} exception: {e}")
            retries += 1
            await asyncio.sleep(2 ** retries)
        raise RuntimeError(f"API POST {endpoint} failed after {self.max_retries} retries")

    async def get_attendance(self, employee_ids: List[str], date: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if employee_ids:
            params["employee_id"] = ",".join(employee_ids)
        if date:
            params["date"] = date
        return await self._get(ENDPOINTS["attendance_status"], params)

    async def get_skills(self, employee_ids: List[str]) -> Dict[str, Any]:
        params = {}
        if employee_ids:
            params["employee_id"] = ",".join(employee_ids)
        return await self._get(ENDPOINTS["employee_skills"], params)

    async def get_capacity(self, employee_ids: List[str], date: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if employee_ids:
            params["employee_id"] = ",".join(employee_ids)
        if date:
            params["date"] = date
        return await self._get(ENDPOINTS["employee_capacity"], params)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_task_priority(self, task_ids: List[str]) -> Dict[str, Any]:
        params = {}
        if task_ids:
            params["task_id"] = ",".join(task_ids)
        return await self._get(ENDPOINTS["task_priority"], params)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_task_due_date(self, task_ids: List[str]) -> Dict[str, Any]:
        params = {}
        if task_ids:
            params["task_id"] = ",".join(task_ids)
        return await self._get(ENDPOINTS["task_due_date"], params)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_task_dependencies(self, task_ids: List[str]) -> Dict[str, Any]:
        params = {}
        if task_ids:
            params["task_id"] = ",".join(task_ids)
        return await self._get(ENDPOINTS["task_dependencies"], params)

    async def create_assignment(self, employee_id: str, task_id: str, assignment_details: Dict[str, Any]) -> Any:
        data = {
            "employee_id": employee_id,
            "task_id": task_id,
            "assignment_details": assignment_details
        }
        return await self._post(ENDPOINTS["create_assignment"], data)

    async def get_utilization_summary(self, date: Optional[str] = None) -> Any:
        params = {}
        if date:
            params["date"] = date
        return await self._get(ENDPOINTS["utilization_summary"], params)

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_unassigned_tasks(self, date: Optional[str] = None) -> Any:
        params = {}
        if date:
            params["date"] = date
        return await self._get(ENDPOINTS["unassigned_tasks"], params)

    async def close(self):
        await self.session.aclose()

# Assignment Optimizer
class AssignmentOptimizer:
    """
    Implements assignment logic, applies business rules, balances workload, respects constraints.
    """
    def __init__(self):
        pass

    async def optimize_assignments(
        self,
        employees: List[Employee],
        tasks: List[Task],
        attendance_data: Dict[str, Any],
        skills_data: Dict[str, Any],
        capacity_data: Dict[str, Any],
        priority_data: Dict[str, Any],
        due_date_data: Dict[str, Any],
        dependencies_data: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns (assignments, unassigned_tasks)
        """
        async with trace_step(
            "optimize_assignments", step_type="process",
            decision_summary="Apply business rules to assign tasks",
            output_fn=lambda r: f"assigned={len(r[0])}, unassigned={len(r[1])}"
        ) as step:
            # Build lookup tables
            emp_map = {e.id: e for e in employees}
            task_map = {t.id: t for t in tasks}
            attendance = {str(eid): status for eid, status in attendance_data.items()}
            skills = {str(eid): set(skills_data.get(str(eid), [])) for eid in emp_map}
            capacity = {str(eid): float(capacity_data.get(str(eid), 0)) for eid in emp_map}
            priority = {str(tid): priority_data.get(str(tid), "medium") for tid in task_map}
            due_date = {str(tid): due_date_data.get(str(tid), None) for tid in task_map}
            dependencies = {str(tid): dependencies_data.get(str(tid), []) for tid in task_map}

            assignments = []
            unassigned = []

            # Attendance-based filtering and capacity adjustment
            eligible_employees = []
            for eid in emp_map:
                att = attendance.get(eid, "absent")
                if att not in ["present", "half-day"]:
                    continue
                if att == "half-day":
                    capacity[eid] = capacity.get(eid, 0) * 0.5
                eligible_employees.append(eid)

            # Build skill/capacity matrix
            emp_skill_matrix = {
                eid: skills.get(eid, set()) for eid in eligible_employees
            }
            emp_capacity = {
                eid: capacity.get(eid, 0) for eid in eligible_employees
            }

            # Task assignment
            assigned_tasks = set()
            for tid, task in task_map.items():
                # Check dependencies
                deps = dependencies.get(tid, [])
                if deps:
                    incomplete = [d for d in deps if d not in assigned_tasks]
                    if incomplete:
                        unassigned.append({
                            "task_id": tid,
                            "reason": "Dependency not completed",
                            "dependencies": incomplete
                        })
                        continue
                # Find eligible employee
                assigned = False
                for eid in eligible_employees:
                    if set(task.required_skills).issubset(emp_skill_matrix.get(eid, set())):
                        if emp_capacity.get(eid, 0) >= task.estimated_effort:
                            # Assign
                            assignments.append({
                                "employee_id": eid,
                                "employee_name": emp_map[eid].name,
                                "task_id": tid,
                                "task_name": task.name,
                                "priority": priority.get(tid, "medium"),
                                "due_date": due_date.get(tid, None),
                                "effort": task.estimated_effort
                            })
                            emp_capacity[eid] -= task.estimated_effort
                            assigned_tasks.add(tid)
                            assigned = True
                            break
                if not assigned:
                    reason = "No available employee with required skills/capacity"
                    unassigned.append({
                        "task_id": tid,
                        "reason": reason
                    })
            step.capture((assignments, unassigned))
            return assignments, unassigned

# Reporting Tool
class ReportingTool:
    """
    Generates utilization summaries and unassigned task reports.
    """
    def __init__(self, data_client: DataIntegrationClient):
        self.data_client = data_client

    async def generate_utilization_summary(self, assignments: List[Dict[str, Any]]) -> UtilizationSummary:
        async with trace_step(
            "generate_utilization_summary", step_type="process",
            decision_summary="Generate utilization summary",
            output_fn=lambda r: f"summary_keys={list(r.summary.keys())}"
        ) as step:
            # Calculate utilization per employee
            df = pd.DataFrame(assignments)
            summary = {}
            if not df.empty:
                grouped = df.groupby("employee_id")["effort"].sum().to_dict()
                for eid, total_effort in grouped.items():
                    summary[eid] = {
                        "employee_name": assignments[0]["employee_name"] if assignments else "",
                        "total_effort": total_effort
                    }
            else:
                summary = {}
            step.capture(UtilizationSummary(summary=summary))
            return UtilizationSummary(summary=summary)

    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_unassigned_tasks_report(self, unassigned_tasks: List[Dict[str, Any]]) -> UnassignedTaskReport:
        async with trace_step(
            "generate_unassigned_tasks_report", step_type="process",
            decision_summary="Generate unassigned task report",
            output_fn=lambda r: f"unassigned_count={len(r.unassigned_tasks)}"
        ) as step:
            report = UnassignedTaskReport(unassigned_tasks=unassigned_tasks)
            step.capture(report)
            return report

# Audit Logger
class AuditLogger:
    """
    Logs assignment decisions, rationales, and errors for compliance and transparency.
    """
    def __init__(self):
        self.audit_log = []

    async def log_assignment_decision(self, decision: Dict[str, Any]) -> AuditLogResult:
        async with trace_step(
            "log_assignment_decision", step_type="process",
            decision_summary="Log assignment decision",
            output_fn=lambda r: f"success={r.success}"
        ) as step:
            try:
                # Mask PII in logs
                masked_decision = {
                    k: (mask_pii(str(v)) if k in ("employee_name",) else v)
                    for k, v in decision.items()
                }
                self.audit_log.append(masked_decision)
                step.capture(AuditLogResult(success=True, details="Logged"))
                return AuditLogResult(success=True, details="Logged")
            except Exception as e:
                logger.error(f"Audit log error: {e}")
                step.capture(AuditLogResult(success=False, details=str(e)))
                return AuditLogResult(success=False, details=str(e))

    async def log_error(self, error_detail: str) -> AuditLogResult:
        async with trace_step(
            "log_error", step_type="process",
            decision_summary="Log error",
            output_fn=lambda r: f"success={r.success}"
        ) as step:
            try:
                self.audit_log.append({"error": error_detail})
                step.capture(AuditLogResult(success=True, details="Error logged"))
                return AuditLogResult(success=True, details="Error logged")
            except Exception as e:
                logger.error(f"Audit log error: {e}")
                step.capture(AuditLogResult(success=False, details=str(e)))
                return AuditLogResult(success=False, details=str(e))

# LLM Interaction Handler
class LLMInteractionHandler:
    """
    Manages prompt construction, LLM calls, and response parsing.
    """
    def __init__(self):
        self.model = "gpt-4.1"
        self.temperature = 0.7
        self.max_tokens = 2000
        self.system_prompt = (
            "You are a professional Employee Work Assignment Agent. Assign daily work only to employees marked as available in attendance. "
            "Use task priority, due date, required skills, dependencies, and individual capacity to create balanced allocations. "
            "Exclude absent or leave employees and reduce capacity for half-day availability. Output assignments, a utilization summary, "
            "and clearly labeled unassigned tasks with reasons. Communicate decisions clearly and transparently."
        )
        self.few_shot_examples = [
            "Assign the following tasks: Task A (priority high, due today, skill X), Task B (priority medium, due tomorrow, skill Y). Employees: John (present, skill X), Jane (half-day, skill Y).",
            "Tasks: Task C (priority high, skill Z). Employees: Mike (absent, skill Z)."
        ]

    def _get_llm_client(self):
        try:
            from openai import AsyncAzureOpenAI
        except ImportError:
            raise RuntimeError("azure-ai (openai) package not installed")
        api_key = Config.get_azure_openai_key()
        endpoint = Config.get_azure_openai_endpoint()
        deployment = Config.get_azure_openai_deployment()
        if not api_key or not endpoint or not deployment:
            raise RuntimeError("Azure OpenAI configuration missing")
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment
        )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def call_llm(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> LLMResponse:
        async with trace_step(
            "call_llm", step_type="llm_call",
            decision_summary="Call Azure GPT-4.1 for assignment reasoning",
            output_fn=lambda r: f"length={len(r.content) if r else 0}"
        ) as step:
            client = self._get_llm_client()
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            for ex in self.few_shot_examples:
                messages.append({"role": "user", "content": ex})
            messages.append({"role": "user", "content": prompt})
            _t0 = _time.time()
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
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
                result = LLMResponse(content=content)
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"LLM call error: {e}")
                step.capture(LLMResponse(content=f"LLM error: {e}"))
                raise

# Main Agent
class BaseAgent:
    pass

class EmployeeWorkAssignmentAgent(BaseAgent):
    """
    Main agent class for employee work assignment.
    """
    def __init__(self):
        self.input_processor = InputProcessor()
        self.data_client = DataIntegrationClient()
        self.assignment_optimizer = AssignmentOptimizer()
        self.reporting_tool = ReportingTool(self.data_client)
        self.audit_logger = AuditLogger()
        self.llm_handler = LLMInteractionHandler()

    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def validate_employee_and_task_data(self, employee_data, task_data) -> ValidationResult:
        async with trace_step(
            "validate_employee_and_task_data", step_type="parse",
            decision_summary="Validate employee and task data",
            output_fn=lambda r: f"valid={r.valid}"
        ) as step:
            try:
                data = {"employees": employee_data, "tasks": task_data}
                result = await self.input_processor.validate_input(data)
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Validation error: {e}")
                await self.audit_logger.log_error(str(e))
                step.capture(ValidationResult(valid=False, errors=[str(e)]))
                return ValidationResult(valid=False, errors=[str(e)])

    async def assign_daily_work(self, employee_list: List[Employee], task_list: List[Task]) -> AssignmentResult:
        async with trace_step(
            "assign_daily_work", step_type="plan",
            decision_summary="Assign daily work tasks to employees",
            output_fn=lambda r: f"assignments={len(r.assignments)}, unassigned={len(r.unassigned_tasks)}"
        ) as step:
            try:
                employee_ids = [e.id for e in employee_list]
                task_ids = [t.id for t in task_list]
                # Fetch all data in parallel
                fetchers = [
                    self.data_client.get_attendance(employee_ids),
                    self.data_client.get_skills(employee_ids),
                    self.data_client.get_capacity(employee_ids),
                    self.data_client.get_task_priority(task_ids),
                    self.data_client.get_task_due_date(task_ids),
                    self.data_client.get_task_dependencies(task_ids)
                ]
                attendance_data, skills_data, capacity_data, priority_data, due_date_data, dependencies_data = await asyncio.gather(*fetchers)
                # Assignment logic
                assignments, unassigned = await self.assignment_optimizer.optimize_assignments(
                    employee_list, task_list,
                    attendance_data, skills_data, capacity_data,
                    priority_data, due_date_data, dependencies_data
                )
                # Create assignments via API
                for a in assignments:
                    try:
                        await self.data_client.create_assignment(
                            a["employee_id"], a["task_id"], assignment_details=a
                        )
                        await self.audit_logger.log_assignment_decision(a)
                    except Exception as e:
                        logger.error(f"Assignment API error: {e}")
                        await self.audit_logger.log_error(str(e))
                # Reporting
                utilization_summary = await self.reporting_tool.generate_utilization_summary(assignments)
                unassigned_report = await self.reporting_tool.generate_unassigned_tasks_report(unassigned)
                result = AssignmentResult(
                    assignments=assignments,
                    utilization_summary=utilization_summary.summary,
                    unassigned_tasks=unassigned_report.unassigned_tasks
                )
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Assignment error: {e}")
                await self.audit_logger.log_error(str(e))
                raise

    async def generate_utilization_summary(self, assignments: List[Dict[str, Any]]) -> UtilizationSummary:
        return await self.reporting_tool.generate_utilization_summary(assignments)

    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def output_unassigned_tasks(self, unassigned_tasks: List[Dict[str, Any]]) -> UnassignedTaskReport:
        return await self.reporting_tool.generate_unassigned_tasks_report(unassigned_tasks)

    async def audit_assignment_decisions(self, assignment_decisions: List[Dict[str, Any]]) -> AuditLogResult:
        results = []
        for decision in assignment_decisions:
            res = await self.audit_logger.log_assignment_decision(decision)
            results.append(res)
        return AuditLogResult(success=all(r.success for r in results), details="Batch audit complete")

    @trace_agent(agent_name='Employee Work Assignment Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def call_llm(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> LLMResponse:
        return await self.llm_handler.call_llm(prompt, parameters)

# FastAPI app
app = FastAPI(
    title="Employee Work Assignment Agent",
    description="Assigns daily work tasks to employees based on real-time data and business rules.",
    version="1.0.0"
)

# CORS (allow all origins for demo; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = EmployeeWorkAssignmentAgent()

# Exception handler for malformed JSON
@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, ValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Malformed input data",
                "details": str(exc),
                "tips": "Check for missing fields, invalid types, or malformed JSON. Ensure all quotes and commas are correct."
            }
        )
    elif isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "tips": "Check your request and try again."
            }
        )
    else:
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "details": str(exc),
                "tips": "If you submitted a large payload, check for JSON formatting issues or reduce input size."
            }
        )

@app.post("/assign", response_model=AssignmentResult)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def assign_endpoint(request: Request):
    """
    Assign daily work tasks to employees.
    """
    try:
        data = await request.json()
    except Exception as e:
        logger.error(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Malformed JSON",
                "details": str(e),
                "tips": "Ensure your JSON is valid. Check for missing quotes, commas, or brackets."
            }
        )
    # Input validation
    validation = await agent.input_processor.validate_input(data)
    if not validation.valid:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Input validation failed",
                "details": validation.errors,
                "tips": "Check employee and task data for completeness and correctness."
            }
        )
    try:
        employees, tasks = await agent.input_processor.parse_input(data)
        result = await agent.assign_daily_work(employees, tasks)
        return {
            "success": True,
            "assignments": result.assignments,
            "utilization_summary": result.utilization_summary,
            "unassigned_tasks": result.unassigned_tasks
        }
    except Exception as e:
        logger.error(f"Assignment failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Assignment failed",
                "details": str(e),
                "tips": "Check input data and try again. If the problem persists, contact support."
            }
        )

@app.post("/llm-explain", response_model=LLMResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def llm_explain_endpoint(request: Request):
    """
    Get LLM explanation for assignment decisions.
    """
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "Prompt is required",
                    "tips": "Provide a non-empty prompt string."
                }
            )
        response = await agent.call_llm(prompt)
        return {
            "success": True,
            "content": response.content
        }
    except Exception as e:
        logger.error(f"LLM explain failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "LLM explanation failed",
                "details": str(e),
                "tips": "Try again later or contact support."
            }
        )

@app.get("/health")
async def health_check():
    return {"success": True, "status": "ok"}



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
        logger.info("Starting Employee Work Assignment Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=False)
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