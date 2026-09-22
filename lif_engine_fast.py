import numpy as np
import pandas as pd

from numba import njit
from scipy.sparse import load_npz


# ============================================================
# NUMBA SİMÜLASYONU
# ============================================================

@njit(cache=True, fastmath=True)
def simulate_fast(

    n,

    indptr,
    indices,
    weights,

    source_indices,
    source_mask,

    dt,
    simulation_ms,

    stim_start_ms,
    stim_end_ms,
    rate_hz,

    v_rest,
    v_reset,
    v_threshold,

    mem_decay,
    syn_decay,
    coupling,

    delay_steps,
    refractory_steps,

    w_syn,

    seed

):

    np.random.seed(seed)


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

    refractory = np.zeros(
        n,
        dtype=np.int16
    )

    spike_counts = np.zeros(
        n,
        dtype=np.int32
    )


    # ========================================================
    # ACTIVE NEURON LIST
    #
    # Bütün 165k nöronu her timestep taramayacağız.
    # Bir nöron ilk kez synaptic input alınca buraya girer.
    # ========================================================

    active_indices = np.empty(
        n,
        dtype=np.int32
    )

    is_active = np.zeros(
        n,
        dtype=np.uint8
    )

    active_count = 0


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
    # SYNAPTIC DELAY RING BUFFER
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
    # COUNTERS
    # ========================================================

    stimulus_events = 0

    overflow = False


    simulation_steps = int(
        round(
            simulation_ms
            /
            dt
        )
    )


    poisson_probability = (

        rate_hz

        *

        dt

        /

        1000.0

    )


    # ========================================================
    # ANA LOOP
    # ========================================================

    for step in range(
        simulation_steps
    ):

        time_ms = (
            step
            *
            dt
        )


        current_count = 0


        # ====================================================
        # 1) SADECE AKTİF NÖRONLARI UPDATE ET
        # ====================================================

        for a in range(
            active_count
        ):

            neuron = active_indices[a]


            # -----------------------------------------------
            # REFRACTORY DEĞİLSE LIF UPDATE
            # -----------------------------------------------

            if refractory[neuron] == 0:

                old_v = V[neuron]
                old_g = g[neuron]


                V[neuron] = (

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


                g[neuron] = (

                    old_g

                    *

                    syn_decay

                )


                # -------------------------------------------
                # THRESHOLD
                # -------------------------------------------

                if V[neuron] > v_threshold:

                    if current_count >= MAX_SPIKES_PER_STEP:

                        overflow = True

                        return (
                            spike_counts,
                            stimulus_events,
                            active_count,
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
        # 2) EXTERNAL POISSON STIMULUS
        # ====================================================

        if (
            rate_hz > 0.0
            and
            time_ms >= stim_start_ms
            and
            time_ms < stim_end_ms
        ):

            for s in range(
                len(
                    source_indices
                )
            ):

                neuron = source_indices[s]


                if (
                    np.random.random()
                    <
                    poisson_probability
                ):

                    stimulus_events += 1


                    # Aynı timestep'te threshold ile
                    # zaten spike attıysa tekrar ekleme.
                    if (
                        last_spike_step[
                            neuron
                        ]
                        !=
                        step
                    ):

                        if current_count >= MAX_SPIKES_PER_STEP:

                            overflow = True

                            return (
                                spike_counts,
                                stimulus_events,
                                active_count,
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
        # 3) DELAY SÜRESİ DOLAN ESKİ SPIKE'LARI AL
        # ====================================================

        delay_slot = (
            step
            %
            delay_steps
        )

        delayed_count = delay_counts[
            delay_slot
        ]


        # ====================================================
        # 4) EVENT-DRIVEN SYNAPTIC PROPAGATION
        #
        # SciPy:
        # W[:, spikes].sum()
        #
        # yerine sadece spike atan presynaptic nöronların
        # gerçekten var olan outgoing synapse'larını geziyoruz.
        # ====================================================

        for d in range(
            delayed_count
        ):

            pre = delay_buffer[
                delay_slot,
                d
            ]


            start = indptr[
                pre
            ]

            end = indptr[
                pre + 1
            ]


            for edge in range(
                start,
                end
            ):

                post = indices[
                    edge
                ]


                # Refractory postsynaptic nöron
                # input kabul etmiyor.
                if refractory[
                    post
                ] != 0:

                    continue


                synaptic_value = (

                    weights[
                        edge
                    ]

                    *

                    w_syn

                )


                g[
                    post
                ] += synaptic_value


                # İlk kez aktif olduysa listeye ekle.
                if is_active[
                    post
                ] == 0:

                    is_active[
                        post
                    ] = 1

                    active_indices[
                        active_count
                    ] = post

                    active_count += 1


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


            # Stimulus population refractory değil.
            if source_mask[
                neuron
            ]:

                refractory[
                    neuron
                ] = 0

            else:

                refractory[
                    neuron
                ] = refractory_steps


        # ====================================================
        # 6) ESKİ REFRACTORY TIMER'LARINI AZALT
        # ====================================================

        for a in range(
            active_count
        ):

            neuron = active_indices[
                a
            ]


            if (
                refractory[
                    neuron
                ] > 0
                and
                last_spike_step[
                    neuron
                ] != step
            ):

                refractory[
                    neuron
                ] -= 1


        # ====================================================
        # 7) CURRENT SPIKE'LARI DELAY BUFFER'A YAZ
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


    return (
        spike_counts,
        stimulus_events,
        active_count,
        overflow
    )


# ============================================================
# ENGINE
# ============================================================

class FastMaleCNSLIF:

    def __init__(

        self,

        matrix_file=
        "data/processed/lif-connectome-csr.npz",

        neurons_file=
        "data/processed/simulation-neurons.parquet"

    ):

        print("=" * 70)
        print("FAST MALECNS ENGINE YÜKLENİYOR")
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


        # ====================================================
        # CSC GEREKİYOR:
        #
        # column = presynaptic neuron
        # rows   = postsynaptic neurons
        # ====================================================

        self.W = load_npz(
            matrix_file
        ).tocsc()


        # Numba için contiguous arrays
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
        # LIF PARAMETRELERİ
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


        self.coupling = np.float32(

            (

                self.tau_synapse

                /

                (
                    self.tau_synapse
                    -
                    self.tau_membrane
                )

            )

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
            "Fast engine hazır."
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


        spike_counts, stimulus_events, active_count, overflow = (

            simulate_fast(

                self.N,

                self.indptr,
                self.indices,
                self.weights,

                source_indices,
                source_mask,

                self.dt,
                simulation_ms,

                stim_start_ms,
                stim_end_ms,
                float(
                    rate_hz
                ),

                self.v_rest,
                self.v_reset,
                self.v_threshold,

                self.mem_decay,
                self.syn_decay,
                self.coupling,

                self.delay_steps,
                self.refractory_steps,

                self.w_syn,

                seed

            )

        )


        return {

            "spike_counts":
                spike_counts,

            "stimulus_events":
                int(
                    stimulus_events
                ),

            "active_neurons":
                int(
                    active_count
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