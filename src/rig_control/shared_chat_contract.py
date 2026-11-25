"""Gateway-owned route names and reviewed serving gates for shared-weight chats."""

ROUTE_PREFIX_BY_MODEL = {
    "qwen3.6:35b-a3b-xl": "shared-35b-",
    "nemotron-3.5-bolt:74b-a9b": "shared-nemo-",
}

# A new serving implementation gets an offer only after benchmarking imports
# a validated recipe for that implementation. Older 35B service predates this gate.
SERVING_RECIPES = {
    "nemotron-3.5-bolt:74b-a9b": ("nemotron74_shared_kv_http_v1", "shared_kv_http_v1"),
}


def serving_approved(catalog: dict, model_id: str) -> bool:
    if model_id not in ROUTE_PREFIX_BY_MODEL:
        return False
    requirement = SERVING_RECIPES.get(model_id)
    if requirement is None:
        return True
    recipe_id, procedure = requirement
    recipe = (catalog.get("load_recipes") or {}).get("recipes", {}).get(recipe_id, {})
    model = (catalog.get("models") or {}).get(model_id, {})
    return (recipe.get("status") == "validated" and recipe.get("procedure") == procedure
            and recipe.get("model") == model_id and model.get("status") == "active"
            and not model.get("in_testing"))


def validate_admission(attached: dict, *, expected_instance: str, parent_gpu: int) -> dict:
    """Validate the authoritative orchestration receipt before publishing a route."""
    gpu, capacity = attached.get("gpu"), attached.get("context_cap")
    instance = attached.get("service_instance_id")
    reservation = attached.get("reservation")
    if not isinstance(expected_instance, str) or not expected_instance:
        raise ValueError("shared service has no instance identity")
    if instance != expected_instance:
        raise ValueError("shared service changed during admission")
    if type(gpu) is not int or gpu == parent_gpu:
        raise ValueError("shared child placement did not isolate the parent GPU")
    if type(capacity) is not int or capacity < 1:
        raise ValueError("shared child has no valid context capacity")
    if not isinstance(reservation, dict):
        raise ValueError("shared child has no reservation receipt")
    if not isinstance(reservation.get("reservation_id"), str) or not reservation["reservation_id"]:
        raise ValueError("shared child reservation has no identity")
    if reservation.get("service_instance_id") != instance:
        raise ValueError("shared child reservation belongs to another service instance")
    if reservation.get("context_capacity") != capacity:
        raise ValueError("shared child capacity differs from its reservation")
    placement = reservation.get("placement")
    if not isinstance(placement, dict) or placement.get("gpus") != [gpu]:
        raise ValueError("shared child placement differs from its reservation")
    if type(reservation.get("reserved_mib")) is not int or reservation["reserved_mib"] < 1:
        raise ValueError("shared child reservation has no memory allocation")
    return {"gpu": gpu, "context_capacity": capacity, "service_instance_id": instance,
            "reservation": reservation}
