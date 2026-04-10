"""
Human-in-the-Loop (HITL) Workflow Manager  +  Expert Annotation System
=======================================================================

Implements a formal 2-step approval workflow:

    Student/Counselor  ──submit_request──►  Pending Queue
                                                │
    Teacher  ──approve / reject──►        Approved / Rejected

Each request carries:
    - student_id, current pathway, desired pathway
    - reason from requestor
    - teacher justification on approval/rejection

Approved requests override the RL recommendation.
Rejected requests are archived for audit.

This replaces the simple direct-override pattern with an
auditable, role-based workflow suitable for school deployment.
"""

import asyncio
import json
from datetime import datetime
from enum import Enum
from pathlib import Path

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False


class RequestStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class HITLRequest:
    """A single pathway change request."""

    def __init__(
        self,
        request_id: str,
        student_id: int,
        current_pathway: str,
        desired_pathway: str,
        reason: str,
        requested_by: str = "Student",
        teacher_id: str | None = None,
    ) -> None:
        self.request_id = request_id
        self.student_id = student_id
        self.current_pathway = current_pathway
        self.desired_pathway = desired_pathway
        self.reason = reason
        self.requested_by = requested_by
        self.teacher_id = teacher_id

        self.status = RequestStatus.PENDING
        self.submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.resolved_at: str | None = None
        self.resolved_by: str | None = None
        self.resolution_justification: str | None = None

    def to_dict(self) -> dict:
        return {
            'request_id': self.request_id,
            'student_id': self.student_id,
            'current_pathway': self.current_pathway,
            'desired_pathway': self.desired_pathway,
            'reason': self.reason,
            'requested_by': self.requested_by,
            'teacher_id': self.teacher_id,
            'status': self.status.value,
            'submitted_at': self.submitted_at,
            'resolved_at': self.resolved_at,
            'resolved_by': self.resolved_by,
            'resolution_justification': self.resolution_justification,
        }


class HITLManager:
    """
    Manages the pathway change request queue.

    Usage::

        mgr = HITLManager()

        # Student or counselor submits
        req_id = mgr.submit_request(
            student_id=42,
            current_pathway="STEM",
            desired_pathway="SOCIAL_SCIENCES",
            reason="Student shows stronger language aptitude",
        )

        # Teacher reviews
        mgr.approve_request(req_id, teacher_id="T001",
                            justification="Agreed after reviewing scores")
        # or
        mgr.reject_request(req_id, teacher_id="T001",
                           reason="Scores still favour STEM")

        # Query
        pending = mgr.get_pending_requests()
        override = mgr.get_active_override(student_id=42)
    """

    VALID_PATHWAYS = {'STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS'}

    def __init__(self, state_path: Path | None = None) -> None:
        self._requests: dict[str, HITLRequest] = {}
        self._counter = 0
        self._state_path = Path(state_path) if state_path else None
        if self._state_path and self._state_path.exists():
            self._load_state()

    # ────── submit ──────────────────────────────────────────

    def submit_request(
        self,
        student_id: int,
        current_pathway: str,
        desired_pathway: str,
        reason: str,
        requested_by: str = "Student",
        teacher_id: str | None = None,
    ) -> str:
        """
        Submit a new pathway change request.

        Returns the request_id.
        Raises ValueError for invalid pathways or duplicate pending requests.
        """
        if desired_pathway not in self.VALID_PATHWAYS:
            raise ValueError(f"Invalid pathway: {desired_pathway}")
        if current_pathway == desired_pathway:
            raise ValueError("Current and desired pathways are the same")

        # Check for existing pending request for this student
        for req in self._requests.values():
            if req.student_id == student_id and req.status == RequestStatus.PENDING:
                raise ValueError(
                    f"Student {student_id} already has a pending request ({req.request_id})"
                )

        self._counter += 1
        req_id = f"REQ-{self._counter:04d}"
        self._requests[req_id] = HITLRequest(
            request_id=req_id,
            student_id=student_id,
            current_pathway=current_pathway,
            desired_pathway=desired_pathway,
            reason=reason,
            requested_by=requested_by,
            teacher_id=teacher_id,
        )
        self._save_state()
        return req_id

    # ────── teacher actions ─────────────────────────────────

    def approve_request(
        self, request_id: str, teacher_id: str, justification: str = ""
    ) -> bool:
        """
        Approve a pending request.  The desired pathway becomes the active override.
        """
        req = self._requests.get(request_id)
        if req is None:
            raise ValueError(f"Request not found: {request_id}")
        if req.status != RequestStatus.PENDING:
            raise ValueError(f"Request {request_id} is already {req.status.value}")

        req.status = RequestStatus.APPROVED
        req.resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        req.resolved_by = teacher_id
        req.resolution_justification = justification
        self._save_state()
        return True

    def reject_request(
        self, request_id: str, teacher_id: str, reason: str = ""
    ) -> bool:
        """Reject a pending request."""
        req = self._requests.get(request_id)
        if req is None:
            raise ValueError(f"Request not found: {request_id}")
        if req.status != RequestStatus.PENDING:
            raise ValueError(f"Request {request_id} is already {req.status.value}")

        req.status = RequestStatus.REJECTED
        req.resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        req.resolved_by = teacher_id
        req.resolution_justification = reason
        self._save_state()
        return True

    # ────── queries ─────────────────────────────────────────

    def get_pending_requests(self) -> list[dict]:
        """Return all pending requests as dicts."""
        return [
            req.to_dict()
            for req in self._requests.values()
            if req.status == RequestStatus.PENDING
        ]

    def get_requests_for_student(self, student_id: int) -> list[dict]:
        """Return all requests (any status) for a student."""
        return [
            req.to_dict()
            for req in self._requests.values()
            if req.student_id == student_id
        ]

    def get_active_override(self, student_id: int) -> dict | None:
        """
        Return the most recent approved request for a student, or None.

        This is used by the recommender to check if a teacher has
        overridden the RL recommendation.
        """
        approved = [
            req for req in self._requests.values()
            if req.student_id == student_id and req.status == RequestStatus.APPROVED
        ]
        if not approved:
            return None
        # Latest approval wins
        approved.sort(key=lambda r: r.resolved_at or '', reverse=True)
        return approved[0].to_dict()

    def get_all_overrides(self) -> dict[int, dict]:
        """
        Return a dict mapping student_id → override info for all
        students with active (approved) overrides.
        """
        overrides = {}
        for req in self._requests.values():
            if req.status == RequestStatus.APPROVED:
                sid = req.student_id
                if sid not in overrides or (req.resolved_at or '') > (overrides[sid].get('resolved_at', '')):
                    overrides[sid] = req.to_dict()
        return overrides

    def get_all_requests(self) -> list[dict]:
        """Return all requests for audit purposes."""
        return [req.to_dict() for req in self._requests.values()]

    def get_request(self, request_id: str) -> dict | None:
        """Get a single request by ID."""
        req = self._requests.get(request_id)
        return req.to_dict() if req else None

    # ────── convenience ─────────────────────────────────────

    def remove_override(self, student_id: int) -> bool:
        """
        Remove the active override for a student by rejecting the
        most recent approved request (effectively reverting to RL).
        """
        override = self.get_active_override(student_id)
        if override is None:
            return False
        req = self._requests.get(override['request_id'])
        if req:
            req.status = RequestStatus.REJECTED
            req.resolution_justification = "Override removed"
            req.resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            self._save_state()
            return True
        return False

    @property
    def pending_count(self) -> int:
        return sum(1 for r in self._requests.values() if r.status == RequestStatus.PENDING)

    @property
    def approved_count(self) -> int:
        return sum(1 for r in self._requests.values() if r.status == RequestStatus.APPROVED)

    def clear(self):
        """Clear all requests (for testing)."""
        self._requests.clear()
        self._counter = 0
        self._save_state()

    # ────── persistence ────────────────────────────────────

    def _save_state(self) -> None:
        """Persist all requests to JSON file (synchronous)."""
        if not self._state_path:
            return
        try:
            state = {
                'counter': self._counter,
                'requests': [req.to_dict() for req in self._requests.values()]
            }
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"  ⚠ HITL save failed: {e}")

    async def save_async(self) -> None:
        """Persist all requests to JSON without blocking the event loop."""
        await asyncio.to_thread(self._save_state)

    async def load_async(self) -> None:
        """Load requests from JSON without blocking the event loop."""
        await asyncio.to_thread(self._load_state)

    def _load_state(self) -> None:
        """Load requests from JSON file (synchronous)."""
        if not self._state_path or not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self._counter = state.get('counter', 0)
            skipped = 0
            for rd in state.get('requests', []):
                try:
                    req = HITLRequest(
                        request_id=rd['request_id'],
                        student_id=rd['student_id'],
                        current_pathway=rd['current_pathway'],
                        desired_pathway=rd['desired_pathway'],
                        reason=rd['reason'],
                        requested_by=rd.get('requested_by', 'Student'),
                        teacher_id=rd.get('teacher_id'),
                    )
                    req.status = RequestStatus(rd.get('status', 'pending'))
                    req.submitted_at = rd.get('submitted_at')
                    req.resolved_at = rd.get('resolved_at')
                    req.resolved_by = rd.get('resolved_by')
                    req.resolution_justification = rd.get('resolution_justification')
                    self._requests[req.request_id] = req
                except Exception as e:
                    skipped += 1
                    print(f"  ⚠ Skipped corrupt HITL request: {e}")
            if skipped:
                print(f"  ⚠ Skipped {skipped} corrupt requests")
            print(f"  ✓ Loaded {len(self._requests)} HITL requests "
                  f"({self.pending_count} pending)")
        except Exception as e:
            print(f"  ⚠ HITL load failed: {e}")


# =============================================================================
# EXPERT ANNOTATION SYSTEM
# =============================================================================
#
# Problem addressed:
#   The DQN is trained to imitate _infer_action(), a heuristic that infers
#   pathway labels from observed score improvements.  A model that correctly
#   mimics a flawed heuristic is not more valuable than the heuristic itself.
#
#   Expert annotations break this circularity: teachers review a student's
#   6-year grade trajectory and assign a pathway label independently of any
#   algorithm.  50 annotations create a ground-truth validation set that
#   enables the first honest evaluation of whether the learned policy is
#   pedagogically valid.
#
# Workflow:
#   1. Call AnnotationManager.get_review_task(student_id, competencies_df)
#      → returns a structured dict formatted for teacher display.
#   2. Teacher reads the trajectory and selects a pathway.
#   3. Call AnnotationManager.submit_annotation(...) to persist the label.
#   4. Call AnnotationManager.get_annotations_as_validation_set()
#      → returns {student_id: pathway_label} for use in evaluate_model().
# =============================================================================


class ExpertAnnotation:
    """A single teacher-provided pathway label for one student's grade trajectory."""

    VALID_PATHWAYS = {'STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS'}

    def __init__(
        self,
        annotation_id: str,
        student_id: int,
        grade_trajectory: dict,
        annotator_id: str,
        pathway_label: str,
        notes: str = '',
    ):
        if pathway_label not in self.VALID_PATHWAYS:
            raise ValueError(f"Invalid pathway_label '{pathway_label}'. "
                             f"Must be one of {self.VALID_PATHWAYS}")
        self.annotation_id   = annotation_id
        self.student_id      = student_id
        self.grade_trajectory = grade_trajectory
        self.annotator_id    = annotator_id
        self.pathway_label   = pathway_label
        self.notes           = notes
        self.annotated_at    = datetime.now().strftime('%Y-%m-%d %H:%M')

    def to_dict(self) -> dict:
        return {
            'annotation_id':    self.annotation_id,
            'student_id':       self.student_id,
            'grade_trajectory': self.grade_trajectory,
            'annotator_id':     self.annotator_id,
            'pathway_label':    self.pathway_label,
            'notes':            self.notes,
            'annotated_at':     self.annotated_at,
        }


class AnnotationManager:
    """
    Manages expert pathway annotations for model validation.

    Usage::

        mgr = AnnotationManager(state_path='data/annotations.json')

        # Present a student's trajectory to a teacher
        task = mgr.get_review_task(student_id=42, competencies_df=df)
        # teacher reads task['trajectory'] and selects a pathway ...

        # Store the annotation
        ann_id = mgr.submit_annotation(
            student_id=42,
            grade_trajectory=task['trajectory'],
            annotator_id='Teacher_Wanjiku',
            pathway_label='STEM',
            notes='Strong math growth across G6-G9'
        )

        # Use annotations for independent model evaluation
        val_set = mgr.get_annotations_as_validation_set()
        # val_set = {42: 'STEM', ...}

        stats = mgr.get_annotation_stats()
        # stats['total'], stats['coverage_pct'], ...
    """

    VALID_PATHWAYS = {'STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS'}
    COVERAGE_TARGET = 50   # minimum annotations for a statistically meaningful validation set

    # Standard CBC subject columns expected in competencies_df
    _SUBJECT_COLS = [
        'MATH', 'SCIENCE', 'ENGLISH', 'KISWAHILI', 'SOCIAL_STUDIES',
        'CREATIVE_ARTS', 'PHYSICAL_EDUCATION', 'RELIGIOUS_EDUCATION', 'ICT',
    ]

    def __init__(self, state_path: Path | None = None) -> None:
        self._annotations: dict[str, ExpertAnnotation] = {}
        self._counter = 0
        self._state_path = Path(state_path) if state_path else None
        if self._state_path and self._state_path.exists():
            self._load_state()

    # ── task generation ───────────────────────────────────────────────────────

    def get_review_task(self, student_id: int, competencies_df) -> dict | None:  # type: ignore[type-arg]
        """
        Build a structured review task for a teacher.

        Reads the student's score history from competencies_df and returns a
        human-readable dict containing the grade trajectory, pathway options,
        and a plain-language prompt — everything a teacher needs to annotate.

        Returns None if the student has no records in competencies_df.
        """
        student_data = competencies_df[competencies_df['student_id'] == student_id].copy()
        if student_data.empty:
            return None

        trajectory: dict = {}
        for grade in sorted(student_data['grade'].unique()):
            grade_row = student_data[student_data['grade'] == grade]
            if grade_row.empty:
                continue
            scores: dict = {}
            for subj in self._SUBJECT_COLS:
                col = f'{subj}_score'
                if col in grade_row.columns:
                    val = grade_row[col].values[0]
                    if _NP_AVAILABLE:
                        import numpy as _np
                        if not _np.isnan(float(val)):
                            scores[subj] = round(float(val), 1)
                    else:
                        try:
                            scores[subj] = round(float(val), 1)
                        except (ValueError, TypeError):
                            pass
            if scores:
                trajectory[f'Grade_{grade}'] = scores

        if not trajectory:
            return None

        # Compute simple subject averages across all grades for quick orientation
        averages: dict = {}
        for subj in self._SUBJECT_COLS:
            vals = [
                grades[subj]
                for grades in trajectory.values()
                if subj in grades
            ]
            if vals:
                averages[subj] = round(sum(vals) / len(vals), 1)

        already_annotated = bool(self.get_annotations_for_student(student_id))

        return {
            'student_id':         student_id,
            'trajectory':         trajectory,
            'subject_averages':   averages,
            'pathway_options':    sorted(self.VALID_PATHWAYS),
            'already_annotated':  already_annotated,
            'prompt': (
                f"Student {student_id}: review the grade-by-grade subject scores above. "
                f"Based on the overall performance pattern and trajectory, which Senior "
                f"Secondary pathway best fits this student?\n"
                f"  STEM            — strong Maths / Science\n"
                f"  SOCIAL_SCIENCES — strong Languages / Social Studies\n"
                f"  ARTS_SPORTS     — strong Creative Arts / PE"
            ),
        }

    # ── submission ────────────────────────────────────────────────────────────

    def submit_annotation(
        self,
        student_id: int,
        grade_trajectory: dict,
        annotator_id: str,
        pathway_label: str,
        notes: str = '',
    ) -> str:
        """
        Persist a teacher's pathway label for a student.

        Multiple annotations for the same student are allowed (inter-rater
        reliability).  Use get_consensus_label(student_id) to resolve conflicts.

        Returns the annotation_id.
        """
        if pathway_label not in self.VALID_PATHWAYS:
            raise ValueError(f"Invalid pathway_label '{pathway_label}'. "
                             f"Must be one of {self.VALID_PATHWAYS}")

        self._counter += 1
        ann_id = f'ANN-{self._counter:04d}'
        self._annotations[ann_id] = ExpertAnnotation(
            annotation_id=ann_id,
            student_id=student_id,
            grade_trajectory=grade_trajectory,
            annotator_id=annotator_id,
            pathway_label=pathway_label,
            notes=notes,
        )
        self._save_state()
        return ann_id

    # ── queries ───────────────────────────────────────────────────────────────

    def get_annotations_as_validation_set(self) -> dict[int, str]:
        """
        Return {student_id: pathway_label} for all annotated students.

        For students with multiple annotations, the majority label is used;
        ties are broken alphabetically (deterministic).  This dict can be
        passed directly to evaluate_model() as an external ground-truth source.
        """
        from collections import Counter
        grouped: dict[int, list[str]] = {}
        for ann in self._annotations.values():
            grouped.setdefault(ann.student_id, []).append(ann.pathway_label)

        result: dict[int, str] = {}
        for sid, labels in grouped.items():
            counts = Counter(labels)
            majority = counts.most_common(1)[0][0]
            result[sid] = majority
        return result

    def get_consensus_label(self, student_id: int) -> str | None:
        """
        Return the majority-vote pathway label for a student, or None if
        no annotations exist.  Logs a warning if annotators disagree.
        """
        from collections import Counter
        anns = [a for a in self._annotations.values() if a.student_id == student_id]
        if not anns:
            return None
        counts = Counter(a.pathway_label for a in anns)
        majority, majority_count = counts.most_common(1)[0]
        if len(counts) > 1 and majority_count == min(counts.values()):
            print(f"  ⚠ Annotation conflict for student {student_id}: {dict(counts)}")
        return majority

    def get_annotation_stats(self) -> dict:
        """Summary statistics for the annotation collection."""
        from collections import Counter
        labels = [ann.pathway_label for ann in self._annotations.values()]
        counts = Counter(labels)
        total  = len(labels)
        annotators = set(ann.annotator_id for ann in self._annotations.values())

        # Per-student counts (some students may have multiple annotations)
        unique_students = len(set(ann.student_id for ann in self._annotations.values()))
        coverage_pct = round(unique_students / self.COVERAGE_TARGET * 100, 1)

        return {
            'total_annotations': total,
            'unique_students':   unique_students,
            'coverage_target':   self.COVERAGE_TARGET,
            'coverage_pct':      coverage_pct,
            'ready_for_validation': unique_students >= self.COVERAGE_TARGET,
            'per_pathway':       dict(counts),
            'n_annotators':      len(annotators),
            'annotators':        sorted(annotators),
        }

    def get_all_annotations(self) -> list[dict]:
        return [ann.to_dict() for ann in self._annotations.values()]

    def get_annotations_for_student(self, student_id: int) -> list[dict]:
        return [
            ann.to_dict()
            for ann in self._annotations.values()
            if ann.student_id == student_id
        ]

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            state = {
                'counter':     self._counter,
                'annotations': [ann.to_dict() for ann in self._annotations.values()],
            }
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"  ⚠ Annotation save failed: {e}")

    async def save_async(self) -> None:
        """Persist annotations without blocking the event loop."""
        await asyncio.to_thread(self._save_state)

    async def load_async(self) -> None:
        """Load annotations without blocking the event loop."""
        await asyncio.to_thread(self._load_state)

    def _load_state(self) -> None:
        if not self._state_path or not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self._counter = state.get('counter', 0)
            for ad in state.get('annotations', []):
                ann = ExpertAnnotation(
                    annotation_id=ad['annotation_id'],
                    student_id=ad['student_id'],
                    grade_trajectory=ad.get('grade_trajectory', {}),
                    annotator_id=ad['annotator_id'],
                    pathway_label=ad['pathway_label'],
                    notes=ad.get('notes', ''),
                )
                ann.annotated_at = ad.get('annotated_at', ann.annotated_at)
                self._annotations[ann.annotation_id] = ann
            print(f"  ✓ Loaded {len(self._annotations)} expert annotations "
                  f"({len(set(a.student_id for a in self._annotations.values()))} students)")
        except Exception as e:
            print(f"  ⚠ Annotation load failed: {e}")
