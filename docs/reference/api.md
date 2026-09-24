# API reference

The signatures below come from Python annotations. Descriptions come from
Google-style docstrings. Griffe analyzes the source without importing the
runtime modules.

## Experiment configuration

::: run_experiment.ExperimentPlan
    options:
      members: false

::: run_experiment.GHZ_Klocal

::: run_experiment.parse_experiment_config

::: run_experiment.run_from_config

## Experiment execution

::: evaluate_shadow.ShadowScalingExperiment
    options:
      members:
        - run
        - run_batched
        - estimate_full
        - save_results
        - load_results

::: experiment.Experiments
    options:
      members:
        - plot_avg_var
        - plot_avg_var_zero_segments
        - plot_sample_complexity
        - get_plot_dataframe

## Shadow protocols

::: shadow.Shadow
    options:
      members:
        - init
        - sample
        - create_snapshots
        - estimate_weak_property
        - estimate_weak_properties
        - estimate_properties
        - ground_truth

::: shadow.SGMShadow
    options:
      members:
        - init

::: shadow.PauliShadow
    options:
      members: false

::: shadow.CliffordShadow
    options:
      members: false

::: tools.majorana.FermionicGaussianShadow
    options:
      members: false

## States and observables

::: tools.state.State
    options:
      members:
        - init_random
        - init
        - prepare_state
        - n

::: tools.state.HRState
    options:
      members: false

::: tools.state.GHZType
    options:
      members: false

::: tools.majorana.MajoranaState
    options:
      members: false

::: tools.observable.Observable
    options:
      members:
        - init
        - op
        - circuit
        - trace
        - n

::: tools.observable.PauliObservable
    options:
      members:
        - init
        - init_random
        - op
        - circuit
        - trace
        - obs_string
        - n
        - is_ZType

## Estimation and distributions

::: tools.estimator.Estimator
    options:
      members:
        - __call__
        - validate_Ns
        - init_state
        - update
        - finalize

::: tools.estimator.MedianOfMeans
    options:
      members: false

::: tools.distributions.uniform

::: tools.distributions.binomial

::: tools.distributions.fixed_weight

::: tools.distributions.build_sgm_distribution

## Majorana utilities

::: tools.majorana.majorana_tuples

::: tools.majorana.jordan_wigner_majorana

::: tools.majorana.majorana_pauli_data

::: tools.majorana.majorana_observables_list

::: tools.majorana.sample_even_majorana_permutations

::: tools.majorana.pauli_to_majorana_data

::: tools.majorana.majorana_degree_from_pauli

## Low-level mathematical utilities

::: tools.binary_solver.solve_binary

::: tools.binary_solver.stabilizer_phase

::: tools.clifford.Tableau
    options:
      members: false

::: tools.noise.depolarizing_noise

## Plotting

::: plotting.plot_avg_var

::: plotting.plot_avg_var_zero_segments

::: plotting.plot_sample_complexity
