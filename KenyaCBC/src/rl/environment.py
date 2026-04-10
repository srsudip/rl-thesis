"""
Reinforcement Learning Environment for Pathway Improvement Recommendations.

State:  78-dim student trajectory vector (G4-G9 scores, growth rates,
        cluster weights, cosine similarities, core gaps, preference, feedback)
Action: 9 improvement recommendations (see dqn_coaching.ACTIONS)
Reward: Composite (Δsuitability + Δstrength + preference_alignment + feedback_agreement)

Built on pre-extracted grade transitions from real student data via
dqn_coaching.extract_transitions, so training uses actual student trajectories.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PATHWAYS, get_pathway_index


class PathwayEnvironment:
    """
    RL environment for pathway improvement recommendations.

    Each episode is one grade-to-grade transition for one student.
    reset(student_id) picks a random transition for that student.
    step(action) evaluates the recommended improvement action against the
    student's actual observed behaviour (inferred from score changes).
    """

    def __init__(self, assessments_df: pd.DataFrame, competencies_df: pd.DataFrame,
                 verbose: bool = True):
        self.assessments_df = assessments_df
        self.competencies_df = competencies_df

        # Import coaching components (avoids circular imports at module level)
        from src.rl.dqn_coaching import (
            extract_transitions, NUM_ACTIONS, get_state_dim, ACTIONS,
        )
        self._ACTIONS = ACTIONS
        self.num_actions = NUM_ACTIONS
        self.state_dim = get_state_dim()   # 78

        # Build pathway-preference dataframe if available in the input data
        df_pathways = self._build_pathways_df()

        if verbose:
            print("  PathwayEnvironment: extracting grade transitions...")
        try:
            self.transitions = extract_transitions(assessments_df, df_pathways)
        except Exception as exc:
            if verbose:
                print(f"  Warning: extract_transitions failed ({exc}). Using fallback.")
            self.transitions = []

        if not self.transitions:
            self._build_fallback_transitions()

        # Index transitions by student_id for fast episode sampling
        self.student_transitions: dict[int, list[dict]] = defaultdict(list)
        for t in self.transitions:
            self.student_transitions[t['student_id']].append(t)

        self.students: list[int] = list(self.student_transitions.keys())

        # Derive per-student ground-truth action and pathway
        self._student_best_action: dict[int, int] = {}
        self._student_preferred_pathway: dict[int, str] = {}
        self._build_ground_truth()

        self._current_transition: dict | None = None

        if verbose:
            print(f"  PathwayEnvironment ready: {len(self.students)} students, "
                  f"{len(self.transitions)} transitions, state_dim={self.state_dim}, "
                  f"action_dim={self.num_actions}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_pathways_df(self) -> pd.DataFrame | None:
        """Try to find a student→pathway preference mapping in the dataframes."""
        candidates = [
            (self.assessments_df, 'actual_pathway'),
            (self.assessments_df, 'pathway_affinity'),
            (self.assessments_df, 'recommended_pathway'),
            (self.competencies_df, 'actual_pathway'),
            (self.competencies_df, 'recommended_pathway'),
        ]
        for df, col in candidates:
            if col in df.columns:
                result = (df[['student_id', col]]
                          .drop_duplicates('student_id')
                          .rename(columns={col: 'actual_pathway'})
                          .copy())
                return result
        return None

    def _build_ground_truth(self):
        """
        For each student, use the HIGHEST-REWARD extracted transition to determine:
          - best_action: the action inferred from observed score changes
          - preferred_pathway: argmax of G9 pathway suitability (state[51:54])

        Suitability at the current grade is always used for preferred_pathway —
        not the action's target_pathway — because explore-type actions (ids 6/7)
        point to an alternative pathway rather than the student's best-fit one.
        Using the highest-reward transition avoids anomalous final-year grades.
        """
        pw_names = ['STEM', 'SOCIAL_SCIENCES', 'ARTS_SPORTS']
        for student_id, trans_list in self.student_transitions.items():
            # Use the terminal G9→G9 self-loop transition (highest grade_after)
            # which has the FULL trajectory state (G7+G8+G9) as current.
            # Falls back to highest-reward transition if no terminal is present.
            terminal_candidates = [t for t in trans_list
                                   if t['grade_before'] == t['grade_after']]
            if terminal_candidates:
                ref_t = terminal_candidates[-1]
            else:
                ref_t = max(trans_list, key=lambda t: t['grade_after'])

            best_action = ref_t['action']
            self._student_best_action[student_id] = best_action

            action_obj = self._ACTIONS[best_action]
            if action_obj.target_pathway:
                self._student_preferred_pathway[student_id] = action_obj.target_pathway
            else:
                # G9 suitability at state indices 51-53 (full G9 state from terminal transition)
                cw = ref_t['state'][51:54]
                self._student_preferred_pathway[student_id] = pw_names[int(np.argmax(cw))]

    def _build_fallback_transitions(self):
        """
        Fallback when extract_transitions fails: build minimal transitions from
        grade-9 competency scores, padding to 78 dims with zeros.
        """
        from config import COMPETENCIES
        comp_names = list(COMPETENCIES.keys())
        grade9 = self.competencies_df[self.competencies_df['grade'] == 9]
        self.transitions = []
        for _, row in grade9.iterrows():
            comp_scores = np.array(
                [float(row.get(f'{c}_score', 50.0)) / 100.0 for c in comp_names],
                dtype=np.float32,
            )
            state = np.zeros(78, dtype=np.float32)
            state[:len(comp_scores)] = comp_scores
            self.transitions.append({
                'student_id': int(row['student_id']),
                'grade_before': 8,
                'grade_after': 9,
                'state': state,
                'action': 5,   # "maintain_current" default
                'reward': 0.0,
                'next_state': state.copy(),
                'done': True,
            })

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, student_id: int | None = None) -> np.ndarray:
        """
        Reset to a new episode by picking a random grade transition for
        the given student (or a random student if none given).
        """
        if student_id is None:
            student_id = int(np.random.choice(self.students))

        trans_list = self.student_transitions.get(student_id)
        if trans_list:
            self._current_transition = trans_list[np.random.randint(len(trans_list))]
        else:
            self._current_transition = self.transitions[np.random.randint(len(self.transitions))]

        return self._current_transition['state']

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Evaluate the recommended action against actual student behaviour.

        Reward tiers:
          - Exact action match      → actual transition reward
          - Same target pathway     → 50% of transition reward
          - Wrong pathway direction → small negative signal
        """
        t = self._current_transition
        true_action = t['action']
        true_reward = float(t['reward'])

        true_pathway = self._ACTIONS[true_action].target_pathway
        pred_pathway = self._ACTIONS[action].target_pathway
        exact_match = (action == true_action)
        pathway_match = (true_pathway is not None and true_pathway == pred_pathway)

        if exact_match:
            reward = true_reward
        elif pathway_match:
            reward = true_reward * 0.5
        else:
            reward = min(true_reward * 0.1, 0.0) - 0.1

        info = {
            'student_id': t['student_id'],
            'recommended': action,
            'ground_truth': true_action,
            'match': exact_match,
            'pathway_match': pathway_match,
        }

        return t['next_state'], reward, bool(t['done']), info

    # ------------------------------------------------------------------
    # Accessors used by trainer / evaluation
    # ------------------------------------------------------------------

    def get_ground_truth_pathway(self, student_id: int) -> str:
        """Preferred pathway for the student (backward-compatible with trainer)."""
        return self._student_preferred_pathway.get(student_id, 'STEM')

    def get_ground_truth_action(self, student_id: int) -> int:
        """Ground-truth improvement action inferred from the student's last transition."""
        return self._student_best_action.get(student_id, 5)

    def get_latest_state(self, student_id: int) -> np.ndarray:
        """
        Return the state of the terminal G9→G9 self-loop transition.

        This is the FULL trajectory state (G7+G8+G9 in scope, G9 as current),
        built by extract_transitions for every student.  Using it for evaluation
        ensures the DQN sees the same G9-as-current distribution it was trained
        on (via the terminal transitions), eliminating the distribution shift
        that capped accuracy at ~72% when evaluating on earlier-grade states.
        """
        trans_list = self.student_transitions.get(student_id, [])
        if not trans_list:
            return np.zeros(self.state_dim, dtype=np.float32)
        # Terminal transition: grade_before == grade_after (self-loop)
        terminals = [t for t in trans_list if t['grade_before'] == t['grade_after']]
        if terminals:
            return terminals[-1]['state']
        # Fallback: next_state of the highest-grade transition
        latest = max(trans_list, key=lambda t: t['grade_after'])
        return latest['next_state']

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_action_dim(self) -> int:
        return self.num_actions
