"""Test rapide de compatibilité de student_agent.py avec le contrat public."""

from __future__ import annotations

import sys
import traceback

from rl_arena.arena_api import Action
from rl_arena.environment import TrainingEnvironment
from student_agent import MyAgent


def fail(message: str) -> None:
    print(f"[ECHEC] {message}")
    raise SystemExit(1)


def main() -> None:
    print("Vérification de l'agent...")
    try:
        agent = MyAgent()
        env = TrainingEnvironment()
        env.register_agent("student", agent)
        env.add_default_bots()
        observations = env.reset(seed=12345)
        if "student" not in observations:
            fail("aucune Observation initiale reçue")

        first_action = agent.act(observations["student"])
        if not isinstance(first_action, Action):
            fail(f"act() doit retourner Action, reçu {type(first_action).__name__}")

        for _ in range(40):
            result = env.step({})
            if result.terminated.get("student", False) or result.truncated:
                break

        print("[OK] import de l'agent")
        print("[OK] reset()")
        print("[OK] act(Observation) -> Action")
        print("[OK] exécution dans le moteur multi-agent")
        print("Votre agent respecte le contrat de base.")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("exception pendant l'exécution")


if __name__ == "__main__":
    main()
