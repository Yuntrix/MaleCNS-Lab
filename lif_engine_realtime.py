import numpy as np
import pandas as pd

from numba import njit
from scipy.sparse import load_npz


# ============================================================
# LAZY EXACT UPDATE
#
# Nöron uzun süre sessiz kaldıysa aradaki her 0.1 ms'yi
# tek tek hesaplamak yerine aynı denklemin kapalı formuyla
# direkt bugünkü state'e getiriyoruz.
# ============================================================

@njit(
    cache=True,
    fastmath=True,
    inline="always"
)
def advance_lazy(
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
# REALTIME SIMULATOR
# ============================================================

@njit(
    cache=True,
    fastmath=True
)
def simulate_realtime(

    n,

    indptr,
    indices,
    weights,

    source_indices,
    source_mask,

    stimulus_schedule,

    dt,
    simulation_steps,

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

    # ========================================================
    # STATE
    # ========================================================

    V = np.full(
        n,
        v_rest,
        dtype=np.float32
    )

    g = np.zeros(
        n,
        dtype=np.float32
    )


    last_update = np.zeros(
        n,
        dtype=np.int32
    )


    refractory_until = np.zeros(
        n,
        dtype=np.int32
    )


    spike_counts = np.zeros(
        n,
        dtype=np.int32
    )


    # ========================================================
    # HOT NEURONS
    #
    # Sadece kendi kendine threshold'a ulaşabilecek durumda
    # olan nöronlar timestep timestep takip edilir.
    # ========================================================

    hot_indices = np.empty(
        n,
        dtype=np.int32
    )

    is_hot = np.zeros(
        n,
        dtype=np.uint8
    )

    hot_count = 0

    peak_hot_count = 0


    # ========================================================
    # SPIKE BUFFER
    # ========================================================

    MAX_SPIKES_PER_STEP = 50000


    current_spikes = np.empty(
        MAX_SPIKES_PER_STEP,
        dtype=np.int32
    )

    last_spike_step = np.full(
        n,
        -1,
        dtype=np.int32
    )


    # ========================================================
    # DELAY RING
    # ========================================================

    delay_buffer = np.empty(
        (
            delay_steps,
            MAX_SPIKES_PER_STEP
        ),
        dtype=np.int32
    )

    delay_counts = np.zeros(
        delay_steps,
        dtype=np.int32
    )


    # ========================================================
    # STATS
    # ========================================================

    stimulus_events = 0

    overflow = False

    ever_touched = np.zeros(
        n,
        dtype=np.uint8
    )


    threshold_delta = (
        v_threshold
        -
        v_rest
    )


    # ========================================================
    # MAIN LOOP
    # ========================================================

    for step in range(
        simulation_steps
    ):

        current_count = 0


        # ====================================================
        # 1) HOT NÖRONLARI UPDATE ET
        # ====================================================

        new_hot_count = 0


        for h in range(
            hot_count
        ):

            neuron = hot_indices[
                h
            ]


            # Eski/stale kayıt
            if is_hot[
                neuron
            ] == 0:

                continue


            # Refractory ise takipten çıkar.
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


            # ------------------------------------------------
            # Bir timestep exact update
            # ------------------------------------------------

            old_v = V[
                neuron
            ]

            old_g = g[
                neuron
            ]


            V[
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
            # Threshold
            # ------------------------------------------------

            if (
                V[
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
                        spike_counts,
                        stimulus_events,
                        int(
                            ever_touched.sum()
                        ),
                        peak_hot_count,
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
            # ÇOK ÖNEMLİ:
            #
            # Eğer g threshold farkından küçükse ve V zaten
            # threshold altında ise, yeni input gelmeden bu
            # nöron artık kendi başına spike atamaz.
            #
            # Her timestep hesaplamayı bırakıyoruz.
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


            # Hâlâ gerçekten HOT
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

        source_count = len(
            source_indices
        )


        for s in range(
            source_count
        ):

            if stimulus_schedule[
                step,
                s
            ] == 0:

                continue


            neuron = source_indices[
                s
            ]


            stimulus_events += 1


            # Aynı timestep'te threshold spike varsa
            # ikinci kez ekleme.
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
                    spike_counts,
                    stimulus_events,
                    int(
                        ever_touched.sum()
                    ),
                    peak_hot_count,
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


                # -------------------------------------------
                # refractory input almaz
                # -------------------------------------------

                if (
                    step
                    <
                    refractory_until[
                        post
                    ]
                ):

                    continue


                # -------------------------------------------
                # Eğer bu nöron HOT değilse state'i geçmişten
                # şu ana exact olarak getir.
                # -------------------------------------------

                if is_hot[
                    post
                ] == 0:

                    delta_steps = (

                        step

                        -

                        last_update[
                            post
                        ]

                    )


                    if delta_steps > 0:

                        new_v, new_g = advance_lazy(

                            V[
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


                        V[
                            post
                        ] = new_v

                        g[
                            post
                        ] = new_g

                        last_update[
                            post
                        ] = step


                # -------------------------------------------
                # Synaptic input
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
                # Bu nöron yeni input olmadan threshold'a
                # ulaşabilecek kadar güçlü depolarize olduysa
                # HOT listesine al.
                # -------------------------------------------

                if (

                    is_hot[
                        post
                    ] == 0

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
        # 5) CURRENT SPIKE RESET
        # ====================================================

        for s in range(
            current_count
        ):

            neuron = current_spikes[
                s
            ]


            spike_counts[
                neuron
            ] += 1


            V[
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


            # -----------------------------------------------
            # External source population refractory değil
            # -----------------------------------------------

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
        # 6) CURRENT SPIKES -> DELAY RING
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


        if hot_count > peak_hot_count:

            peak_hot_count = (
                hot_count
            )


    return (
        spike_counts,
        stimulus_events,
        int(
            ever_touched.sum()
        ),
        peak_hot_count,
        overflow
    )


# ============================================================
# ENGINE
# ============================================================

class RealtimeMaleCNSLIF:

    def __init__(
        self,
        matrix_file="data/processed/lif-connectome-csr.npz",
        neurons_file="data/processed/simulation-neurons.parquet"
    ):

        print("=" * 70)
        print("REALTIME MALECNS ENGINE YÜKLENİYOR")
        print("=" * 70)


        self.neurons = pd.read_parquet(
            neurons_file
        )


        self.N = len(
            self.neurons
        )


        print(
            "Nöron:",
            f"{self.N:,}"
        )


        print(
            "Connectome yükleniyor..."
        )


        self.W = load_npz(
            matrix_file
        ).tocsc()


        self.indptr = np.ascontiguousarray(
            self.W.indptr
        )


        self.indices = np.ascontiguousarray(

            self.W.indices.astype(
                np.int32,
                copy=False
            )

        )


        self.weights = np.ascontiguousarray(

            self.W.data.astype(
                np.float32,
                copy=False
            )

        )


        print(
            "Bağlantı:",
            f"{self.W.nnz:,}"
        )


        # ====================================================
        # METADATA
        # ====================================================

        self.types = (

            self.neurons[
                "type"
            ]

            .fillna("")

            .astype(str)

        )


        self.superclasses = (

            self.neurons[
                "superclass"
            ]

            .fillna("")

            .astype(str)

        )


        self.descending_mask = (

            self.superclasses
            ==
            "descending_neuron"

        ).to_numpy()


        self.motor_mask = (

            self.superclasses.isin(

                [
                    "vnc_motor",
                    "cb_motor"
                ]

            )

        ).to_numpy()


        # ====================================================
        # MODEL
        # ====================================================

        self.dt = 0.1

        self.v_rest = -52.0
        self.v_reset = -52.0
        self.v_threshold = -45.0

        self.tau_membrane = 20.0
        self.tau_synapse = 5.0

        self.synaptic_delay = 1.8
        self.refractory_ms = 2.2

        self.w_syn = 0.275


        self.mem_decay = np.float32(

            np.exp(

                -self.dt

                /

                self.tau_membrane

            )

        )


        self.syn_decay = np.float32(

            np.exp(

                -self.dt

                /

                self.tau_synapse

            )

        )


        self.alpha = np.float32(

            self.tau_synapse

            /

            (
                self.tau_synapse
                -
                self.tau_membrane
            )

        )


        self.coupling = np.float32(

            self.alpha

            *

            (
                self.syn_decay
                -
                self.mem_decay
            )

        )


        self.delay_steps = int(

            round(

                self.synaptic_delay

                /

                self.dt

            )

        )


        self.refractory_steps = int(

            round(

                self.refractory_ms

                /

                self.dt

            )

        )


        print(
            "Realtime engine hazır."
        )


    # ========================================================
    # STIMULUS SCHEDULE
    #
    # NumPy Generator kullanıyoruz.
    # Böylece eski engine ile AYNI random stimulus oluşuyor.
    # ========================================================

    def create_stimulus_schedule(

        self,

        source_count,

        rate_hz,

        seed,

        simulation_steps,

        stim_start_ms,

        stim_end_ms

    ):

        schedule = np.zeros(

            (
                simulation_steps,
                source_count
            ),

            dtype=np.uint8

        )


        if rate_hz <= 0:

            return schedule


        rng = np.random.default_rng(
            seed
        )


        probability = (

            rate_hz

            *

            self.dt

            /

            1000.0

        )


        for step in range(
            simulation_steps
        ):

            time_ms = (

                step

                *

                self.dt

            )


            if (

                stim_start_ms
                <=
                time_ms
                <
                stim_end_ms

            ):

                values = rng.random(
                    source_count
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
    # RUN
    # ========================================================

    def run(

        self,

        source_types,

        rate_hz,

        seed,

        simulation_ms=140.0,

        stim_start_ms=20.0,

        stim_end_ms=80.0

    ):

        source_mask = (

            self.types.isin(
                source_types
            )

        ).to_numpy()


        source_indices = np.flatnonzero(
            source_mask
        ).astype(
            np.int32
        )


        simulation_steps = int(

            round(

                simulation_ms

                /

                self.dt

            )

        )


        stimulus_schedule = (
            self.create_stimulus_schedule(

                len(
                    source_indices
                ),

                rate_hz,

                seed,

                simulation_steps,

                stim_start_ms,

                stim_end_ms

            )
        )


        (

            spike_counts,
            stimulus_events,
            touched_neurons,
            peak_hot_neurons,
            overflow

        ) = simulate_realtime(

            self.N,

            self.indptr,
            self.indices,
            self.weights,

            source_indices,
            source_mask,

            stimulus_schedule,

            self.dt,
            simulation_steps,

            self.v_rest,
            self.v_reset,
            self.v_threshold,

            self.mem_decay,
            self.syn_decay,
            self.coupling,
            self.alpha,

            self.delay_steps,
            self.refractory_steps,

            self.w_syn

        )


        return {

            "spike_counts":
                spike_counts,

            "stimulus_events":
                int(
                    stimulus_events
                ),

            "touched_neurons":
                int(
                    touched_neurons
                ),

            "peak_hot_neurons":
                int(
                    peak_hot_neurons
                ),

            "overflow":
                bool(
                    overflow
                ),

            "total_spikes":
                int(
                    spike_counts.sum()
                ),

            "unique_neurons":
                int(
                    np.count_nonzero(
                        spike_counts
                    )
                ),

            "descending_spikes":
                int(
                    spike_counts[
                        self.descending_mask
                    ].sum()
                ),

            "motor_spikes":
                int(
                    spike_counts[
                        self.motor_mask
                    ].sum()
                )
        }