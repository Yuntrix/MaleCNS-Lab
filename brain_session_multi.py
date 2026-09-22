import numpy as np

from brain_session import simulate_chunk


class MultiInputBrainSession:

    def __init__(
        self,
        brain,
        input_groups,
        seed=101
    ):

        self.brain = brain
        self.seed = seed

        # ====================================================
        # INPUT GROUPS
        #
        # Örnek:
        #
        # {
        #     "background": np.array([...]),
        #     "looming": np.array([...])
        # }
        # ====================================================

        self.group_indices = {}

        used_indices = set()

        all_indices = []

        start = 0

        self.group_slices = {}


        for name, indices in input_groups.items():

            indices = np.asarray(
                indices,
                dtype=np.int32
            )


            # Duplicate güvenliği
            for neuron in indices:

                neuron_int = int(
                    neuron
                )

                if neuron_int in used_indices:

                    raise ValueError(
                        f"Nöron birden fazla input grubunda: "
                        f"{neuron_int}"
                    )

                used_indices.add(
                    neuron_int
                )


            end = (
                start
                +
                len(indices)
            )


            self.group_indices[
                name
            ] = indices


            self.group_slices[
                name
            ] = slice(
                start,
                end
            )


            all_indices.extend(
                indices.tolist()
            )


            start = end


        self.source_indices = np.asarray(

            all_indices,

            dtype=np.int32

        )


        self.source_mask = np.zeros(

            brain.N,

            dtype=np.bool_

        )


        self.source_mask[
            self.source_indices
        ] = True


        print(
            "Multi-input source neuron:",
            len(
                self.source_indices
            )
        )


        for name in self.group_indices:

            print(
                f"  {name}:",
                len(
                    self.group_indices[
                        name
                    ]
                )
            )


        self.reset(
            seed
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seed=None
    ):

        if seed is not None:

            self.seed = seed


        self.rng = np.random.default_rng(
            self.seed
        )


        n = self.brain.N


        self.v = np.full(

            n,

            self.brain.v_rest,

            dtype=np.float32

        )


        self.g = np.zeros(

            n,

            dtype=np.float32

        )


        self.last_update = np.zeros(

            n,

            dtype=np.int32

        )


        self.refractory_until = np.zeros(

            n,

            dtype=np.int32

        )


        self.hot_indices = np.empty(

            n,

            dtype=np.int32

        )


        self.is_hot = np.zeros(

            n,

            dtype=np.uint8

        )


        self.hot_count = 0


        self.delay_buffer = np.empty(

            (
                self.brain.delay_steps,
                50000
            ),

            dtype=np.int32

        )


        self.delay_counts = np.zeros(

            self.brain.delay_steps,

            dtype=np.int32

        )


        self.last_spike_step = np.full(

            n,

            -1,

            dtype=np.int32

        )


        self.ever_touched = np.zeros(

            n,

            dtype=np.uint8

        )


        self.global_step = 0


    # ========================================================
    # MULTI INPUT SCHEDULE
    # ========================================================

    def create_schedule(
        self,
        group_rates,
        chunk_steps
    ):

        schedule = np.zeros(

            (
                chunk_steps,
                len(
                    self.source_indices
                )
            ),

            dtype=np.uint8

        )


        stimulus_by_group = {}


        for name in self.group_indices:

            indices = self.group_indices[
                name
            ]


            count = len(
                indices
            )


            rate_hz = float(

                group_rates.get(
                    name,
                    0.0
                )

            )


            group_slice = self.group_slices[
                name
            ]


            if (
                rate_hz <= 0
                or
                count == 0
            ):

                stimulus_by_group[
                    name
                ] = 0

                continue


            probability = (

                rate_hz

                *

                self.brain.dt

                /

                1000.0

            )


            group_events = 0


            for step in range(
                chunk_steps
            ):

                values = self.rng.random(
                    count
                )


                spikes = (

                    values

                    <
                    probability

                )


                schedule[
                    step,
                    group_slice
                ] = spikes


                group_events += int(
                    spikes.sum()
                )


            stimulus_by_group[
                name
            ] = group_events


        return (

            np.ascontiguousarray(
                schedule
            ),

            stimulus_by_group

        )


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        group_rates,
        chunk_ms
    ):

        chunk_steps = int(

            round(

                chunk_ms

                /

                self.brain.dt

            )

        )


        (

            stimulus_schedule,
            stimulus_by_group

        ) = self.create_schedule(

            group_rates,

            chunk_steps

        )


        (

            chunk_spikes,
            self.hot_count,
            stimulus_events,
            peak_hot,
            overflow

        ) = simulate_chunk(

            self.brain.N,

            self.brain.indptr,
            self.brain.indices,
            self.brain.weights,

            self.source_indices,
            self.source_mask,

            stimulus_schedule,

            self.v,
            self.g,

            self.last_update,
            self.refractory_until,

            self.hot_indices,
            self.is_hot,
            self.hot_count,

            self.delay_buffer,
            self.delay_counts,

            self.last_spike_step,

            self.ever_touched,

            self.global_step,

            self.brain.v_rest,
            self.brain.v_reset,
            self.brain.v_threshold,

            self.brain.mem_decay,
            self.brain.syn_decay,
            self.brain.coupling,
            self.brain.alpha,

            self.brain.delay_steps,
            self.brain.refractory_steps,

            self.brain.w_syn

        )


        self.global_step += (
            chunk_steps
        )


        return {

            "spike_counts":
                chunk_spikes,

            "stimulus_events":
                int(
                    stimulus_events
                ),

            "stimulus_by_group":
                stimulus_by_group,

            "total_spikes":
                int(
                    chunk_spikes.sum()
                ),

            "descending_spikes":
                int(

                    chunk_spikes[
                        self.brain.descending_mask
                    ].sum()

                ),

            "motor_spikes":
                int(

                    chunk_spikes[
                        self.brain.motor_mask
                    ].sum()

                ),

            "hot_neurons":
                int(
                    self.hot_count
                ),

            "peak_hot":
                int(
                    peak_hot
                ),

            "overflow":
                bool(
                    overflow
                )

        }