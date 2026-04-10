# Human-in-the-Loop Workflow — `src/rl/hitl.py`

## Overview

The HITL system implements a formal 2-step approval workflow for pathway changes:

```
Student/Counselor  ──submit_request──►  Pending Queue
                                            │
Teacher  ──approve / reject──►        Approved / Rejected
```

This replaces simple direct overrides with an auditable, role-based process.

## Workflow

### 1. Submit Request

A student, counselor, or teacher submits a request to change a student's pathway:

```python
from src.rl.hitl import HITLManager

mgr = HITLManager()
req_id = mgr.submit_request(
    student_id=42,
    current_pathway='STEM',
    desired_pathway='SOCIAL_SCIENCES',
    reason='Student shows stronger language aptitude',
)
```

**Validation rules**:
- Desired pathway must be one of: STEM, SOCIAL_SCIENCES, ARTS_SPORTS
- Current and desired cannot be the same
- Only one pending request per student at a time

### 2. Teacher Reviews

```python
# Approve
mgr.approve_request(req_id, teacher_id='T001',
                     justification='Agreed after reviewing scores')

# Or reject
mgr.reject_request(req_id, teacher_id='T002',
                    reason='Scores still favour STEM')
```

### 3. Override Takes Effect

Approved requests override the RL recommendation:

```python
override = mgr.get_active_override(student_id=42)
if override:
    pathway = override['desired_pathway']  # Use this instead of RL
```

## Integration with DataManager

The `DataManager` in `pages/dashboard.py` checks for active HITL overrides before returning RL recommendations. The HITL manager is accessible via the Settings page.

## Request Lifecycle

```
PENDING ──approve──► APPROVED (active override)
   │
   └──reject──► REJECTED (archived, no override)
```

## Audit Trail

Every request records:
- Who submitted it and when
- Who resolved it and when
- The justification for approval/rejection

```python
all_requests = mgr.get_all_requests()  # Full audit log
```
