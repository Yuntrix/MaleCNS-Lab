import inspect

import brain_session
import motor_decoder


print("=" * 100)
print("MALECNS BRAIN SESSION / MOTOR DECODER API INSPECTOR")
print("=" * 100)


# ============================================================
# MODULE FILES
# ============================================================

print()
print("brain_session:")
print(
    getattr(
        brain_session,
        "__file__",
        "UNKNOWN"
    )
)

print()
print("motor_decoder:")
print(
    getattr(
        motor_decoder,
        "__file__",
        "UNKNOWN"
    )
)


# ============================================================
# HELPERS
# ============================================================

def safe_signature(obj):

    try:

        return str(
            inspect.signature(
                obj
            )
        )

    except Exception as exc:

        return (
            f"<signature unavailable: {exc}>"
        )


def print_public_members(
    module,
    title
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


    for name, obj in inspect.getmembers(
        module
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

            print(
                f"  signature: "
                f"{safe_signature(obj)}"
            )


        elif inspect.isfunction(
            obj
        ):

            print()
            print(
                f"FUNCTION: {name}"
            )

            print(
                f"  signature: "
                f"{safe_signature(obj)}"
            )


# ============================================================
# MODULE CONTENTS
# ============================================================

print_public_members(
    brain_session,
    "BRAIN_SESSION PUBLIC CLASSES / FUNCTIONS"
)


print_public_members(
    motor_decoder,
    "MOTOR_DECODER PUBLIC CLASSES / FUNCTIONS"
)


# ============================================================
# CONTINUOUS BRAIN SESSION
# ============================================================

session_class = getattr(
    brain_session,
    "ContinuousBrainSession",
    None
)


print()
print("=" * 100)
print("ContinuousBrainSession DETAILS")
print("=" * 100)


if session_class is None:

    print(
        "ContinuousBrainSession bulunamadı."
    )

else:

    print(
        "Class signature:",
        safe_signature(
            session_class
        )
    )


    print()
    print(
        "Public methods:"
    )


    for name, obj in inspect.getmembers(
        session_class
    ):

        if name.startswith(
            "_"
        ):

            continue


        if callable(
            obj
        ):

            print(
                f"  {name}"
                f"{safe_signature(obj)}"
            )


    # ========================================================
    # IMPORTANT METHOD SOURCES
    # ========================================================

    keywords = [
        "step",
        "run",
        "advance",
        "stim",
        "spike",
        "inject",
        "chunk",
        "reset",
        "simulate",
    ]


    print()
    print("=" * 100)
    print("RELEVANT ContinuousBrainSession METHOD SOURCES")
    print("=" * 100)


    found_any = False


    for name, obj in inspect.getmembers(
        session_class
    ):

        if name.startswith(
            "_"
        ):

            continue


        if not callable(
            obj
        ):

            continue


        lower = name.lower()


        if not any(
            keyword in lower
            for keyword in keywords
        ):

            continue


        found_any = True


        print()
        print("-" * 100)
        print(
            f"METHOD: {name}"
            f"{safe_signature(obj)}"
        )
        print("-" * 100)


        try:

            source = inspect.getsource(
                obj
            )

            print(
                source
            )

        except Exception as exc:

            print(
                f"Source unavailable: {exc}"
            )


    if not found_any:

        print(
            "İlgili method adı bulunamadı."
        )


# ============================================================
# MOTOR DECODER DETAILS
# ============================================================

print()
print("=" * 100)
print("MOTOR DECODER DETAILS")
print("=" * 100)


for target_name in [
    "MotorDecoder",
    "decode",
    "decode_motor",
]:

    obj = getattr(
        motor_decoder,
        target_name,
        None
    )


    if obj is None:

        continue


    print()
    print("-" * 100)

    print(
        f"{target_name}: "
        f"{safe_signature(obj)}"
    )

    print("-" * 100)


    if inspect.isclass(
        obj
    ):

        for method_name, method in inspect.getmembers(
            obj
        ):

            if method_name.startswith(
                "_"
            ):

                continue


            if callable(
                method
            ):

                print(
                    f"  {method_name}"
                    f"{safe_signature(method)}"
                )


    try:

        print()
        print(
            inspect.getsource(
                obj
            )
        )

    except Exception as exc:

        print(
            f"Source unavailable: {exc}"
        )


print()
print("=" * 100)
print("DONE")
print("=" * 100)