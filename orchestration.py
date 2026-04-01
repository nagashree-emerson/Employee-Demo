
import logging
import asyncio
from typing import Dict, Any, List, Optional

import importlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Dynamically import agent classes from their respective folders
def _import_agent_class(module_path: str, class_name: str):
    try:
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        return agent_class
    except Exception as e:
        logger.error(f"Failed to import {class_name} from {module_path}: {e}")
        raise

# Import AttendanceClassificationAgent
AttendanceClassificationAgent = _import_agent_class(
    "code.employee_attendance_classification_agent_design.agent",
    "AttendanceClassificationAgent"
)

# Import EmployeeWorkAssignmentAgent
EmployeeWorkAssignmentAgent = _import_agent_class(
    "code.employee_work_assignment_agent_design.agent",
    "EmployeeWorkAssignmentAgent"
)

class OrchestrationEngine:
    """
    Orchestrates the workflow:
      1. Classifies attendance for each employee.
      2. Assigns daily work based on attendance and provided tasks.
    """

    def __init__(self):
        self.attendance_agent = AttendanceClassificationAgent()
        self.assignment_agent = EmployeeWorkAssignmentAgent()
        self._logger = logger

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestration entrypoint.

        Args:
            input_data: {
                "employees": [
                    {
                        "employee_id": ...,
                        "checkin_time": ...,
                        "shift_start": ...,
                        "grace_period": ...,
                        "half_day_cutoff": ...,
                        "leave_status": ...,
                        "holiday_status": ...,
                        "date": ...,
                        "additional_info": ...
                    },
                    ...
                ],
                "tasks": [
                    {
                        "id": ...,
                        "name": ...,
                        "required_skills": [...],
                        "estimated_effort": ...
                    },
                    ...
                ]
            }

        Returns:
            {
                "attendance_results": [...],
                "assignment_result": {...},
                "errors": [...]
            }
        """
        errors: List[Dict[str, Any]] = []
        attendance_results: List[Dict[str, Any]] = []
        assignment_input_employees: List[Dict[str, Any]] = []
        tasks: List[Dict[str, Any]] = input_data.get("tasks", [])

        employees_input: List[Dict[str, Any]] = input_data.get("employees", [])
        if not isinstance(employees_input, list):
            self._logger.error("Input 'employees' must be a list.")
            raise ValueError("Input 'employees' must be a list.")

        # Step 1: Attendance Classification for each employee
        for idx, emp in enumerate(employees_input):
            try:
                # Only pass fields expected by AttendanceClassificationAgent
                attendance_input = {
                    k: v for k, v in emp.items()
                    if k in [
                        "employee_id", "checkin_time", "shift_start", "grace_period",
                        "half_day_cutoff", "leave_status", "holiday_status", "date", "additional_info"
                    ]
                }
                result = await self.attendance_agent.classify(attendance_input)
                # result is AttendanceOutput (Pydantic model or dict)
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                else:
                    result_dict = dict(result)
                attendance_results.append({
                    "employee_id": emp.get("employee_id"),
                    **result_dict
                })
                if result_dict.get("success"):
                    # Prepare employee dict for assignment agent
                    assignment_input_employees.append({
                        "id": emp.get("employee_id"),
                        "name": emp.get("name") or emp.get("employee_name") or emp.get("employee_id"),
                        # Optionally, add more fields if EmployeeWorkAssignmentAgent expects them
                    })
                else:
                    errors.append({
                        "employee_id": emp.get("employee_id"),
                        "error_type": result_dict.get("error_type"),
                        "error_description": result_dict.get("error_description"),
                        "fixing_tips": result_dict.get("fixing_tips"),
                    })
            except Exception as e:
                self._logger.error(f"Attendance classification failed for employee {emp.get('employee_id')}: {e}")
                errors.append({
                    "employee_id": emp.get("employee_id"),
                    "error_type": type(e).__name__,
                    "error_description": str(e),
                })

        # Step 2: Assignment Agent
        assignment_result: Optional[Dict[str, Any]] = None
        try:
            # Prepare input for assignment agent
            assignment_input = {
                "employees": assignment_input_employees,
                "tasks": tasks
            }
            # The assign_daily_work expects two lists: employees, tasks
            # But the FastAPI endpoint expects a dict; the agent method expects lists.
            # We'll call the method directly with lists.
            # However, to be robust, check if the agent expects lists or dict.
            # We'll try both.
            assign_fn = getattr(self.assignment_agent, "assign_daily_work")
            # Try to import Employee and Task models for type conversion if needed
            # But for now, pass dicts as the agent's input processor handles dicts.
            assignment_result_obj = await assign_fn(assignment_input_employees, tasks)
            # assignment_result_obj is AssignmentResult (Pydantic model or dict)
            if hasattr(assignment_result_obj, "model_dump"):
                assignment_result = assignment_result_obj.model_dump()
            elif hasattr(assignment_result_obj, "dict"):
                assignment_result = assignment_result_obj.dict()
            else:
                assignment_result = dict(assignment_result_obj)
        except Exception as e:
            self._logger.error(f"Work assignment failed: {e}")
            errors.append({
                "step": "work_assignment",
                "error_type": type(e).__name__,
                "error_description": str(e),
            })
            assignment_result = None

        return {
            "attendance_results": attendance_results,
            "assignment_result": assignment_result,
            "errors": errors
        }

# Convenience function for direct use
async def run_orchestration(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async orchestration entrypoint.
    """
    engine = OrchestrationEngine()
    return await engine.execute(input_data)

# Synchronous wrapper for environments that require sync call
def run_orchestration_sync(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous orchestration entrypoint.
    """
    return asyncio.run(run_orchestration(input_data))
