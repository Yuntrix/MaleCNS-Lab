import numpy as np

from numba import njit


# ============================================================
# STATEFUL CHUNK SIMULATOR
# ============================================================

@njit(
    cache=True,
    fastmath=True,
    inline="always"
)
def advance_lazy_session(
    v,
    g,
    steps,
    v_rest,
    mem_decay,
    syn_decay,
    alpha
):

    if steps <= 0:
        return v, g


    em = mem_decay ** steps
    eg = syn_decay ** steps


    new_v = (

        v_rest

        +

        (
            v
            -
            v_rest
        )
        *
        em

        +

        g
        *
        alpha
        *
        (
            eg
            -
            em
        )

    )


    new_g = (
        g
        *
        eg
    )


    return (
        np.float32(new_v),
        np.float32(new_g)
    )


# ============================================================
# BİR CHUNK ÇALIŞTIR
# ============================================================

@njit(
    cache=True,
    fastmath=True
)
def simulate_chunk(

    n,

    indptr,
    indices,
    weights,

    source_indices,
    source_mask,

    stimulus_schedule,

    v,
    g,

    last_update,
    refractory_until,

    hot_indices,
    is_hot,
    hot_count,

    delay_buffer,
    delay_counts,

    last_spike_step,

    ever_touched,

    global_step_start,

    v_rest,
    v_reset,
    v_threshold,

    mem_decay,
    syn_decay,
    coupling,
    alpha,

    delay_steps,
    refractory_steps,

    w_syn

):

    chunk_steps = stimulus_schedule.shape[0]


    chunk_spike_counts = np.zeros(
        n,
        dtype=np.int32
    )


    MAX_SPIKES_PER_STEP = 50000


    current_spikes = np.empty(
        MAX_SPIKES_PER_STEP,
        dtype=np.int32
    )


    threshold_delta = (
        v_threshold
        -
        v_rest
    )


    stimulus_events = 0

    peak_hot = hot_count

    overflow = False


    # ========================================================
    # CHUNK LOOP
    # ========================================================

    for local_step in range(
        chunk_steps
    ):

        step = (
            global_step_start
            +
            local_step
        )


        current_count = 0


        # ====================================================
        # 1) HOT NÖRONLAR
        # ====================================================

        new_hot_count = 0


        for h in range(
            hot_count
        ):

            neuron = hot_indices[
                h
            ]


            if is_hot[
                neuron
            ] == 0:

                continue


            if (
                step
                <
                refractory_until[
                    neuron
                ]
            ):

                is_hot[
                    neuron
                ] = 0

                continue


            old_v = v[
                neuron
            ]

            old_g = g[
                neuron
            ]


            v[
                neuron
            ] = (

                v_rest

                +

                (
                    old_v
                    -
                    v_rest
                )

                *

                mem_decay

                +

                old_g

                *

                coupling

            )


            g[
                neuron
            ] = (

                old_g

                *

                syn_decay

            )


            last_update[
                neuron
            ] = step


            # ------------------------------------------------
            # SPIKE
            # ------------------------------------------------

            if (
                v[
                    neuron
                ]
                >
                v_threshold
            ):

                if (
                    current_count
                    >=
                    MAX_SPIKES_PER_STEP
                ):

                    overflow = True

                    return (
                        chunk_spike_counts,
                        hot_count,
                        stimulus_events,
                        peak_hot,
                        overflow
                    )


                current_spikes[
                    current_count
                ] = neuron

                current_count += 1


                last_spike_step[
                    neuron
                ] = step


                is_hot[
                    neuron
                ] = 0

                continue


            # ------------------------------------------------
            # ARTIK KENDİ BAŞINA SPIKE ATAMAZ
            # ------------------------------------------------

            if (
                g[
                    neuron
                ]
                <=
                threshold_delta
            ):

                is_hot[
                    neuron
                ] = 0

                continue


            hot_indices[
                new_hot_count
            ] = neuron

            new_hot_count += 1


        hot_count = (
            new_hot_count
        )


        # ====================================================
        # 2) EXTERNAL STIMULUS
        # ====================================================

        for s in range(
            len(
                source_indices
            )
        ):

            if (
                stimulus_schedule[
                    local_step,
                    s
                ]
                ==
                0
            ):

                continue


            neuron = source_indices[
                s
            ]


            stimulus_events += 1


            if (
                last_spike_step[
                    neuron
                ]
                ==
                step
            ):

                continue


            if (
                current_count
                >=
                MAX_SPIKES_PER_STEP
            ):

                overflow = True

                return (
                    chunk_spike_counts,
                    hot_count,
                    stimulus_events,
                    peak_hot,
                    overflow
                )


            current_spikes[
                current_count
            ] = neuron

            current_count += 1


            last_spike_step[
                neuron
            ] = step


        # ====================================================
        # 3) DELAYED SPIKES
        # ====================================================

        delay_slot = (
            step
            %
            delay_steps
        )


        delayed_count = (
            delay_counts[
                delay_slot
            ]
        )


        # ====================================================
        # 4) SYNAPTIC PROPAGATION
        # ====================================================

        for d in range(
            delayed_count
        ):

            pre = delay_buffer[
                delay_slot,
                d
            ]


            edge_start = indptr[
                pre
            ]

            edge_end = indptr[
                pre + 1
            ]


            for edge in range(
                edge_start,
                edge_end
            ):

                post = indices[
                    edge
                ]


                if (
                    step
                    <
                    refractory_until[
                        post
                    ]
                ):

                    continue


                # -------------------------------------------
                # COLD NÖRONU ŞU ANA GETİR
                # -------------------------------------------

                if (
                    is_hot[
                        post
                    ]
                    ==
                    0
                ):

                    delta_steps = (

                        step

                        -

                        last_update[
                            post
                        ]

                    )


                    if delta_steps > 0:

                        new_v, new_g = advance_lazy_session(

                            v[
                                post
                            ],

                            g[
                                post
                            ],

                            delta_steps,

                            v_rest,

                            mem_decay,
                            syn_decay,
                            alpha

                        )


                        v[
                            post
                        ] = new_v

                        g[
                            post
                        ] = new_g


                        last_update[
                            post
                        ] = step


                # -------------------------------------------
                # SYNAPTIC INPUT
                # -------------------------------------------

                g[
                    post
                ] += (

                    weights[
                        edge
                    ]

                    *

                    w_syn

                )


                ever_touched[
                    post
                ] = 1


                # -------------------------------------------
                # HOT YAP
                # -------------------------------------------

                if (

                    is_hot[
                        post
                    ]
                    ==
                    0

                    and

                    g[
                        post
                    ]
                    >
                    threshold_delta

                ):

                    is_hot[
                        post
                    ] = 1


                    hot_indices[
                        hot_count
                    ] = post

                    hot_count += 1


        # ====================================================
        # 5) SPIKE RESET
        # ====================================================

        for s in range(
            current_count
        ):

            neuron = current_spikes[
                s
            ]


            chunk_spike_counts[
                neuron
            ] += 1


            v[
                neuron
            ] = v_reset


            g[
                neuron
            ] = 0.0


            last_update[
                neuron
            ] = step


            is_hot[
                neuron
            ] = 0


            if source_mask[
                neuron
            ]:

                refractory_until[
                    neuron
                ] = step

            else:

                refractory_until[
                    neuron
                ] = (

                    step

                    +

                    refractory_steps

                    +

                    1

                )


        # ====================================================
        # 6) DELAY BUFFER
        # ====================================================

        delay_counts[
            delay_slot
        ] = current_count


        for s in range(
            current_count
        ):

            delay_buffer[
                delay_slot,
                s
            ] = current_spikes[
                s
            ]


        if (
            hot_count
            >
            peak_hot
        ):

            peak_hot = hot_count


    return (
        chunk_spike_counts,
        hot_count,
        stimulus_events,
        peak_hot,
        overflow
    )


# ============================================================
# CONTINUOUS SESSION
# ============================================================

class ContinuousBrainSession:

    def __init__(
        self,
        brain,
        source_types,
        seed=101
    ):

        self.brain = brain


        self.source_mask = (

            brain.types.isin(
                source_types
            )

        ).to_numpy()


        self.source_indices = np.flatnonzero(
            self.source_mask
        ).astype(
            np.int32
        )


        self.seed = seed


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
    # STIMULUS CHUNK
    # ========================================================

    def create_schedule(
        self,
        rate_hz,
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


        if rate_hz <= 0:

            return schedule


        probability = (

            rate_hz

            *

            self.brain.dt

            /

            1000.0

        )


        # Bilerek step-step yapıyoruz.
        # Monolithic engine ile aynı RNG sırası.
        for step in range(
            chunk_steps
        ):

            values = self.rng.random(

                len(
                    self.source_indices
                )

            )


            schedule[
                step
            ] = (

                values

                <
                probability

            )


        return np.ascontiguousarray(
            schedule
        )


    # ========================================================
    # BİR CHUNK ÇALIŞTIR
    # ========================================================

    def step(
        self,
        rate_hz,
        chunk_ms
    ):

        chunk_steps = int(

            round(

                chunk_ms

                /

                self.brain.dt

            )

        )


        stimulus_schedule = (
            self.create_schedule(

                rate_hz,

                chunk_steps

            )
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