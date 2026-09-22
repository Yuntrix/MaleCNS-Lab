from collections import deque

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


class MaleCNSLIF:

    def __init__(
        self,
        matrix_file="data/processed/lif-connectome-csr.npz",
        neurons_file="data/processed/simulation-neurons.parquet"
    ):

        print("=" * 70)
        print("MALECNS BRAIN ENGINE YÜKLENİYOR")
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

        print(
            "Bağlantı:",
            f"{self.W.nnz:,}"
        )


        # ====================================================
        # NÖRON BİLGİLERİ
        # ====================================================

        self.types = (
            self.neurons["type"]
            .fillna("")
            .astype(str)
        )

        self.superclasses = (
            self.neurons["superclass"]
            .fillna("")
            .astype(str)
        )


        # ====================================================
        # OUTPUT GRUPLARI
        # ====================================================

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


        # ====================================================
        # EXACT INTEGRATION
        # ====================================================

        self.mem_decay = np.exp(
            -self.dt
            /
            self.tau_membrane
        )

        self.syn_decay = np.exp(
            -self.dt
            /
            self.tau_synapse
        )

        self.coupling = (

            self.tau_synapse

            /

            (
                self.tau_synapse
                -
                self.tau_membrane
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
            "Brain engine hazır."
        )


    # ========================================================
    # SİMÜLASYON
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


        # ====================================================
        # STATE
        # ====================================================

        V = np.full(
            self.N,
            self.v_rest,
            dtype=np.float32
        )

        g = np.zeros(
            self.N,
            dtype=np.float32
        )

        refractory = np.zeros(
            self.N,
            dtype=np.int16
        )

        spike_counts = np.zeros(
            self.N,
            dtype=np.int32
        )


        delay_queue = deque(

            [

                np.empty(
                    0,
                    dtype=np.int32
                )

                for _ in range(
                    self.delay_steps
                )

            ]

        )


        rng = np.random.default_rng(
            seed
        )


        poisson_probability = (

            rate_hz

            *

            self.dt

            /

            1000.0

        )


        simulation_steps = int(
            round(
                simulation_ms
                /
                self.dt
            )
        )


        stimulus_events = 0


        # ====================================================
        # LOOP
        # ====================================================

        for step in range(
            simulation_steps
        ):

            time_ms = (
                step
                *
                self.dt
            )


            # ------------------------------------------------
            # REFRACTORY STATE
            # ------------------------------------------------

            was_refractory = (
                refractory
                >
                0
            )

            free_mask = (
                ~was_refractory
            )


            # ------------------------------------------------
            # EXACT LIF UPDATE
            # ------------------------------------------------

            if np.any(
                free_mask
            ):

                old_v = V[
                    free_mask
                ].copy()

                old_g = g[
                    free_mask
                ].copy()


                V[
                    free_mask
                ] = (

                    self.v_rest

                    +

                    (
                        old_v
                        -
                        self.v_rest
                    )
                    *
                    self.mem_decay

                    +

                    old_g
                    *
                    self.coupling
                )


                g[
                    free_mask
                ] = (

                    old_g
                    *
                    self.syn_decay
                )


            # ------------------------------------------------
            # NORMAL SPIKES
            # ------------------------------------------------

            threshold_spikes = (

                (V > self.v_threshold)

                &

                free_mask

            )


            # ------------------------------------------------
            # EXTERNAL STIMULUS
            # ------------------------------------------------

            forced_spikes = np.zeros(
                self.N,
                dtype=bool
            )


            if (
                rate_hz > 0
                and
                stim_start_ms
                <=
                time_ms
                <
                stim_end_ms
            ):

                random_values = rng.random(
                    len(
                        source_indices
                    )
                )

                selected = source_indices[
                    random_values
                    <
                    poisson_probability
                ]

                forced_spikes[
                    selected
                ] = True

                stimulus_events += len(
                    selected
                )


            # ------------------------------------------------
            # SPIKES
            # ------------------------------------------------

            spikes = (

                threshold_spikes

                |

                forced_spikes

            )


            spike_indices = np.flatnonzero(
                spikes
            ).astype(
                np.int32
            )


            # ------------------------------------------------
            # DELAYED SYNAPTIC INPUT
            # ------------------------------------------------

            delayed_spikes = (
                delay_queue.popleft()
            )


            if delayed_spikes.size > 0:

                incoming = np.asarray(

                    self.W[
                        :,
                        delayed_spikes
                    ].sum(
                        axis=1
                    )

                ).ravel().astype(
                    np.float32
                )


                # refractory nöron input kabul etmez
                incoming[
                    was_refractory
                ] = 0.0


                g += (

                    incoming

                    *

                    self.w_syn

                )


            # ------------------------------------------------
            # SPIKE RESET
            # ------------------------------------------------

            if spike_indices.size > 0:

                spike_counts[
                    spike_indices
                ] += 1


                V[
                    spike_indices
                ] = self.v_reset

                g[
                    spike_indices
                ] = 0.0


                normal_indices = spike_indices[
                    ~source_mask[
                        spike_indices
                    ]
                ]


                refractory[
                    normal_indices
                ] = self.refractory_steps


                # external stimulus population refractory yok
                refractory[
                    source_indices
                ] = 0


            # ------------------------------------------------
            # REFRACTORY TIMER
            # ------------------------------------------------

            decrement_mask = (

                was_refractory

                &

                (
                    refractory
                    >
                    0
                )

            )


            refractory[
                decrement_mask
            ] -= 1


            # ------------------------------------------------
            # DELAY QUEUE
            # ------------------------------------------------

            delay_queue.append(
                spike_indices
            )


        # ====================================================
        # RETURN
        # ====================================================

        return {

            "spike_counts":
                spike_counts,

            "stimulus_events":
                stimulus_events,

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