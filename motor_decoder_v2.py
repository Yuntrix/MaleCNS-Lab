class MotorDecoder:

    def __init__(self):

        # High-confidence motor groups already used by the project.
        self.flight_types = {
            "DLMN C-F",
            "DLMN A, B",
            "DVMN 1A-C",
            "DVMN 2A, B",
            "DVMN 3A, B",
            "HDVM MN",
        }

        self.jump_types = {
            "TTMN",
        }


    def decode(
        self,
        neurons,
        spike_counts,
        motor_mask,
        steering_left_mask=None,
        steering_right_mask=None,
    ):

        motor_neurons = neurons[
            motor_mask
        ].copy()

        motor_neurons[
            "spike_count"
        ] = spike_counts[
            motor_mask
        ]

        motor_neurons[
            "type_clean"
        ] = (
            motor_neurons["type"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
        )


        # ====================================================
        # FLIGHT
        # ====================================================

        flight_mask = (
            motor_neurons[
                "type_clean"
            ].isin(
                self.flight_types
            )
        )

        flight_spikes = int(
            motor_neurons.loc[
                flight_mask,
                "spike_count"
            ].sum()
        )


        # ====================================================
        # JUMP
        # ====================================================

        jump_mask = (
            motor_neurons[
                "type_clean"
            ].isin(
                self.jump_types
            )
        )

        jump_spikes = int(
            motor_neurons.loc[
                jump_mask,
                "spike_count"
            ].sum()
        )


        # ====================================================
        # OTHER MOTOR
        # ====================================================

        total_motor_spikes = int(
            motor_neurons[
                "spike_count"
            ].sum()
        )

        known_spikes = (
            flight_spikes
            +
            jump_spikes
        )

        other_spikes = max(
            0,
            total_motor_spikes
            -
            known_spikes
        )


        # ====================================================
        # ACTION
        #
        # This translates population output into the cartoon
        # body's coarse actuator states. It does not choose an
        # action independently of neural activity.
        # ====================================================

        if (
            flight_spikes == 0
            and
            jump_spikes == 0
            and
            other_spikes == 0
        ):

            action = "IDLE"

        elif (
            jump_spikes
            >
            flight_spikes
        ):

            action = "JUMP"

        elif (
            flight_spikes
            >
            0
        ):

            action = "FLY"

        else:

            action = "MOVE"


        # ====================================================
        # DIRECTIONAL DESCENDING SIGNAL
        #
        # Optional masks are supplied by the server.
        # For V6 they correspond to anatomical LEFT / RIGHT
        # DNa04 neurons.
        #
        # The causal perturbation test showed that active DNa04
        # copies biased modeled motor output to the SAME
        # anatomical side on responsive trials. Therefore:
        #
        #   LEFT DNa04  > RIGHT DNa04 -> -1
        #   RIGHT DNa04 > LEFT DNa04  -> +1
        #
        # This is a modeled steering decoder, not a claim that
        # DNa04 alone is the complete biological steering system.
        # ====================================================

        steering_left_spikes = 0
        steering_right_spikes = 0
        steering_signal = 0.0
        steering_direction = 0

        if (
            steering_left_mask is not None
            and
            steering_right_mask is not None
        ):

            steering_left_spikes = int(
                spike_counts[
                    steering_left_mask
                ].sum()
            )

            steering_right_spikes = int(
                spike_counts[
                    steering_right_mask
                ].sum()
            )

            steering_total = (
                steering_left_spikes
                +
                steering_right_spikes
            )

            if steering_total > 0:

                steering_signal = (
                    steering_right_spikes
                    -
                    steering_left_spikes
                ) / steering_total

                if steering_signal > 0:
                    steering_direction = 1

                elif steering_signal < 0:
                    steering_direction = -1


        return {

            "action":
                action,

            "flight_spikes":
                flight_spikes,

            "jump_spikes":
                jump_spikes,

            "other_spikes":
                other_spikes,

            "total_motor_spikes":
                total_motor_spikes,

            "steering_left_spikes":
                steering_left_spikes,

            "steering_right_spikes":
                steering_right_spikes,

            "steering_signal":
                float(
                    steering_signal
                ),

            "steering_direction":
                int(
                    steering_direction
                ),
        }
