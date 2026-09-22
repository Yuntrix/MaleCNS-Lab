import inspect

import brain_session
import lif_engine_realtime


print("=" * 100)
print("MALECNS BRAIN CORE API INSPECTOR")
print("=" * 100)


# ============================================================
# CONTINUOUS BRAIN SESSION INIT
# ============================================================

session_class = brain_session.ContinuousBrainSession


print()
print("=" * 100)
print("ContinuousBrainSession.__init__")
print("=" * 100)


print(
    "Signature:",
    inspect.signature(
        session_class.__init__
    )
)


print()

try:

    print(
        inspect.getsource(
            session_class.__init__
        )
    )

except Exception as exc:

    print(
        "SOURCE ERROR:",
        exc
    )


# ============================================================
# CREATE SCHEDULE
# ============================================================

print()
print("=" * 100)
print("ContinuousBrainSession.create_schedule")
print("=" * 100)


print(
    "Signature:",
    inspect.signature(
        session_class.create_schedule
    )
)


print()

try:

    print(
        inspect.getsource(
            session_class.create_schedule
        )
    )

except Exception as exc:

    print(
        "SOURCE ERROR:",
        exc
    )


# ============================================================
# REALTIME ENGINE PUBLIC OBJECTS
# ============================================================

print()
print("=" * 100)
print("lif_engine_realtime PUBLIC API")
print("=" * 100)


for name, obj in inspect.getmembers(
    lif_engine_realtime
):

    if name.startswith(
        "_"
    ):

        continue


    if inspect.isclass(
        obj
    ):

        print()
        print(
            f"CLASS: {name}"
        )

        try:

            print(
                "Signature:",
                inspect.signature(
                    obj
                )
            )

        except Exception:

            pass


    elif inspect.isfunction(
        obj
    ):

        print()
        print(
            f"FUNCTION: {name}"
        )

        try:

            print(
                "Signature:",
                inspect.signature(
                    obj
                )
            )

        except Exception:

            pass


# ============================================================
# LIKELY BRAIN / CONNECTOME CLASSES
# ============================================================

keywords = [
    "brain",
    "lif",
    "connect",
    "load",
    "network",
    "csr",
]


print()
print("=" * 100)
print("RELEVANT ENGINE SOURCES")
print("=" * 100)


for name, obj in inspect.getmembers(
    lif_engine_realtime
):

    if name.startswith(
        "_"
    ):

        continue


    lower = name.lower()


    if not any(
        keyword in lower
        for keyword in keywords
    ):

        continue


    if not (
        inspect.isclass(
            obj
        )
        or
        inspect.isfunction(
            obj
        )
    ):

        continue


    print()
    print("-" * 100)
    print(name)
    print("-" * 100)


    try:

        print(
            inspect.getsource(
                obj
            )
        )

    except Exception as exc:

        print(
            "Source unavailable:",
            exc
        )


# ============================================================
# BRAIN_SESSION MODULE GLOBALS
# ============================================================

print()
print("=" * 100)
print("brain_session RELEVANT GLOBAL NAMES")
print("=" * 100)


for name in dir(
    brain_session
):

    lower = name.lower()


    if (
        "simulate" in lower
        or
        "source" in lower
        or
        "brain" in lower
        or
        "load" in lower
    ):

        obj = getattr(
            brain_session,
            name
        )


        print()
        print(
            name,
            type(obj)
        )


        if (
            inspect.isfunction(
                obj
            )
            or
            inspect.isclass(
                obj
            )
        ):

            try:

                print(
                    "Signature:",
                    inspect.signature(
                        obj
                    )
                )

            except Exception:

                pass


print()
print("=" * 100)
print("DONE")
print("=" * 100)