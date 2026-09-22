class MotorDecoder:

    def __init__(self):

        # Bunlar yüksek güvenle bildiğimiz motor grupları.
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
        motor_mask
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
        # OTHER
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
        # Buradaki seçim beynin yerine karar vermiyor.
        # Motor population output'unu animasyon komutuna çeviriyor.
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
        }