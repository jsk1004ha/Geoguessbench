"""Evaluator state machine. Only observation() is sent to a model."""
import base64
import hashlib
import time
import uuid

from pydantic import ValidationError

from .geo import bearing, geo_score, haversine
from .render import render_view
from .schema import Action

COSTS = {"turn": 1.0, "look": 1.0, "zoom": 1.0, "move": 2.0}


class Episode:
    def __init__(self, dataset, protocol, round_id, *, state=None, clock=time.time):
        self.dataset, self.protocol, self.clock = dataset, protocol, clock
        self.rnd = dataset.rounds[round_id]
        node = dataset.nodes[self.rnd.start]
        self.state = state if state is not None else {
            "round_id": round_id, "node": node.id, "episode": uuid.uuid4().hex,
            "heading": self.rnd.start_heading if self.rnd.start_heading is not None else (node.heading or 0),
            "pitch": 0.0, "fov": protocol.initial_fov, "actions": 0, "cost": 0.0,
            "step": 0, "started_at": clock(), "status": "active", "events": [], "error": None,
            "result": None,
        }

    @property
    def final_only(self):
        return (self.protocol.track == "nmpz" or self.state["actions"] >= self.protocol.max_actions
                or self.protocol.max_cost - self.state["cost"] < 1)

    def expire(self):
        if self.state["status"] == "active" and self.clock() - self.state["started_at"] >= self.protocol.time_limit_s:
            self._finish("timeout")

    def exits(self):
        node = self.dataset.nodes[self.state["node"]]
        if self.protocol.track != "moving":
            return []
        return [{"exit": i, "bearing": round(bearing((node.lat, node.lon),
                (self.dataset.nodes[key].lat, self.dataset.nodes[key].lon)), 1)}
                for i, key in enumerate(node.neighbors)]

    def observation(self):
        self.expire()
        state = self.state
        if state["status"] != "active":
            # No scores, labels, GPS, dataset IDs or attribution URLs during a run.
            return {"done": True, "status": state["status"], "step": state["step"]}
        node = self.dataset.nodes[state["node"]]
        image = render_view(self.dataset, node, self.protocol, state["heading"], state["pitch"], state["fov"])
        image_hash = hashlib.sha256(image).hexdigest()
        state["observation_sha256"] = image_hash
        return {
            "done": False, "episode": state["episode"], "step": state["step"], "track": self.protocol.track,
            "image": "data:image/jpeg;base64," + base64.b64encode(image).decode(),
            "image_sha256": image_hash, "heading": state["heading"] if node.heading is not None else None,
            "pitch": state["pitch"] if node.panorama else None, "fov": state["fov"] if node.panorama else None, "actions_remaining": max(0, self.protocol.max_actions - state["actions"]),
            "cost_remaining": max(0, self.protocol.max_cost - state["cost"]),
            "time_remaining_s": max(0, self.protocol.time_limit_s - (self.clock() - state["started_at"])),
            "final_only": self.final_only, "exits": [] if self.final_only else self.exits(),
            "error": state["error"],
        }

    def step(self, payload, expected_step):
        self.expire()
        state = self.state
        if state["status"] != "active":
            raise ValueError("episode is already finished")
        if expected_step != state["step"]:
            raise ValueError("stale step; fetch the current observation instead of repeating an action")
        state["step"] += 1
        state["error"] = None
        event = {"step": expected_step, "observation_sha256": state.get("observation_sha256"),
                 "action": payload, "elapsed_s": self.clock() - state["started_at"]}
        state["events"].append(event)
        was_final = self.final_only
        try:
            action = Action.model_validate(payload)
            if action.type == "submit":
                event["accepted"] = True
                self._finish("ok", action)
                return
            if action.type == "abstain":
                event["accepted"] = True
                self._finish("abstain")
                return
            if was_final:
                raise ValueError("only submit or abstain is allowed")
            if action.type == "move" and self.protocol.track != "moving":
                raise ValueError("movement is forbidden in this track")
            cost = COSTS[action.type]
            if state["cost"] + cost > self.protocol.max_cost:
                raise ValueError("insufficient action budget")
            if action.type == "turn":
                state["heading"] = (state["heading"] + action.degrees) % 360
            elif action.type == "look":
                state["pitch"] = action.degrees
            elif action.type == "zoom":
                state["fov"] = action.fov
            elif action.type == "move":
                neighbors = self.dataset.nodes[state["node"]].neighbors
                if action.exit >= len(neighbors):
                    raise ValueError("exit does not exist")
                state["node"] = neighbors[action.exit]
                # Preserve compass heading when moving; never change the target GPS.
            state["actions"] += 1
            state["cost"] += cost
            event["accepted"] = True
        except (ValidationError, ValueError, TypeError):
            # Do not echo validator inputs/paths or give free unlimited invalid attempts.
            event["accepted"] = False
            state["error"] = "invalid_or_forbidden_action"
            if was_final:
                self._finish("invalid")
            else:
                state["actions"] += 1
                state["cost"] += min(1.0, self.protocol.max_cost - state["cost"])
        event["cost_after"] = state["cost"]

    def _finish(self, status, action=None):
        state = self.state
        state["status"] = status
        target = self.dataset.nodes[self.rnd.start]
        distance = haversine((target.lat, target.lon), (action.lat, action.lon)) if action else None
        state["result"] = {
            "round_id": self.rnd.id, "cluster": self.dataset.cluster[self.rnd.id],
            "country": self.rnd.country, "region": self.rnd.region, "status": status,
            "target": {"lat": target.lat, "lon": target.lon},
            "prediction": {"lat": action.lat, "lon": action.lon} if action else None,
            "confidence_25km": action.confidence_25km if action else None,
            "distance_km": distance, "score": geo_score(distance) if action else 0.0,
            "actions": state["actions"], "cost": state["cost"],
            "elapsed_s": min(self.protocol.time_limit_s, max(0, self.clock() - state["started_at"])),
        }
