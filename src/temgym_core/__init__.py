from typing_extensions import TypeAlias
from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray


PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float
Radians: TypeAlias = float
Degrees: TypeAlias = float


class ShapeYX(NamedTuple):
    """Describe image shape as (height, width) in pixels.

    Parameters
    ----------
    y : int
        Image height in pixels.
    x : int
        Image width in pixels.

    Notes
    -----
    Conforms to numpy/jax indexing order (y first, then x).
    """
    y: int
    x: int


class ScaleYX(NamedTuple):
    """Describe pixel scale in metres per pixel along y and x.

    Parameters
    ----------
    y : float
        Pixel size along y, metres per pixel.
    x : float
        Pixel size along x, metres per pixel.

    Notes
    -----
    Values can be different if pixels are anisotropic.
    """
    y: float
    x: float


class CoordsXY(NamedTuple):
    """Continuous coordinates in the optical frame.

    Parameters
    ----------
    x : numpy.ndarray
        X position(s), metres. Shape broadcastable to context.
    y : numpy.ndarray
        Y position(s), metres. Shape broadcastable to context.
    """
    x: NDArray[np.floating]
    y: NDArray[np.floating]


class PixelsYX(NamedTuple):
    """Discrete pixel coordinates for images.

    Parameters
    ----------
    y : numpy.ndarray
        Pixel row indices. Integer dtype.
    x : numpy.ndarray
        Pixel column indices. Integer dtype.

    Notes
    -----
    Pixel indices are 0-based.
    """
    y: NDArray[np.integer]
    x: NDArray[np.integer]


# Convenience re-exports
try:
    from .plotting import (  # noqa: F401
        plot_model,
        plot_model_plotly,
        plot_abcd_profile,
        PlotParams,
        legacy_beam_plot_params,
    )
except Exception:
    # Plotting has optional dependencies (matplotlib); ignore import errors at package import time.
    pass

try:
    from .diffraction_metrology import (  # noqa: F401
        DiffractionSpotRegistrationError,
        RegisteredBraggSpotMeasurement,
        extract_registered_bragg_spot_offsets,
    )
except Exception:
    pass

try:
    from .phase_branch_continuation import (  # noqa: F401
        CONTINUATION_MANIFEST_SCHEMA,
        HARD_EDGE_SIGN_EXPLANATION,
        MAX_CONTINUATION_PHASE_STEP_RAD,
        ContinuationEvidenceOrigin,
        ContinuationFrame,
        ContinuationFrameRole,
        ContinuationManifestAudit,
        ContinuationObservableKind,
        ContinuationRoute,
        DACContinuationGrid,
        ExperimentalContinuationManifest,
        HardEdgeAliasEnumeration,
        HardEdgePhaseAlias,
        PhaseContinuationError,
        PhaseContinuationResult,
        QReferenceTransform,
        RelativeQSlopeFit,
        audit_continuation_manifest,
        build_experimental_continuation_manifest,
        continuation_manifest_from_payload,
        continuation_manifest_to_payload,
        design_raw_dac_continuation_grid,
        enumerate_hard_edge_phase_aliases,
        fit_relative_q_slope,
        transform_q_reference_covariance,
        unwrap_oriented_ellipse_angles,
        unwrap_signed_response_vector_rotations,
    )
except Exception:
    pass

try:
    from .microscope_model import (  # noqa: F401
        DeflectorConfig,
        LensConfig,
        OperatingMode,
        MicroscopeModel,
    )
except Exception:
    # Microscope model imports optional dependencies through components; keep package import robust.
    pass

try:
    from .multiscale_sample import (  # noqa: F401
        CubicActionField2D,
        CubicInterpolatedSample2D,
        MAGICALSampleAtlas,
        SampleLevel2D,
        SamplePyramid2D,
        SampleView,
        select_sample_view,
        validate_sample_level,
    )
    from .layer_metrology import (  # noqa: F401
        BoundaryUncertaintyModel,
        CertifiedLayerGeometry2D,
        LayerBoundaryExtraction,
        LayerBoundaryRegistration2D,
        LayerObservableCovarianceAugmentation,
        RegisteredLayerMetrology,
        RegisteredLayerPeriodMetrology,
        UnresolvedLayerBoundaryError,
        clip_certified_layer_geometry_to_detector,
        derive_certified_layer_geometry,
        extract_registered_layer_boundaries,
        freeze_layer_boundary_registration,
        layer_observable_covariance_augmentation,
        predict_registered_layer_observables,
        predict_registered_layer_period_observables,
        registered_layer_reference_residuals_px,
        registered_layer_metrology_from_sample_view,
        registered_layer_period_metrology_from_boundary_metrology,
        registered_layer_period_metrology_from_sample_view,
    )
except Exception:
    pass

try:
    from .microscope_inversion import (  # noqa: F401
        FrameObservationBlock,
        HeterogeneousInversionDataset,
        MultistartBasin,
        MultistartBasinDiagnostics,
        compact_30_state_plan,
        diagnose_multistart_basins,
        fit_heterogeneous_dataset,
        hard_edge_solenoid_matrix,
        jacobian_svd,
    )
    from .identifiability_reporting import (  # noqa: F401
        LocalNullFamilyReport,
        MatchedModelRecovery,
        assess_matched_model_recovery,
        verify_local_null_families,
    )
    from .nuisance_profiled_subspace import (  # noqa: F401
        ActiveSubspaceStability,
        NuisanceProfiledSubspace,
        compare_scaled_active_subspaces,
        fit_nuisance_profiled_polytope,
        fit_nuisance_profiled_subspace,
        nuisance_profiled_active_bounds,
        nuisance_profiled_identifiable_subspace,
        profile_nuisance_profiled_polytope_direction,
    )
    from .neoarm_calibration import (  # noqa: F401
        CheckpointObservableDataset,
        DetectorGeometry,
        DetectorNoiseModel,
        GaussianSliceSafetyReport,
        IncompleteCheckpointError,
        KnownFourierFeatureSet,
        MagnificationDACInfeasibleError,
        SignedObjectDefocusDesign,
        SignedObjectDefocusInfeasibleError,
        assess_gaussian_slice_safety,
        aggregate_checkpoint_observables_for_fit,
        build_acquisition_manifest,
        build_heterogeneous_dataset_from_checkpoint,
        conjugate_gaussian_grid_reference,
        design_nominal_magnification_dacs,
        design_signed_object_defocus_dacs,
        frozen_topology_fingerprint,
        known_fourier_features_from_sample_view,
        load_completed_frame_sequence_checkpoint,
        predict_frame,
        propagate_gaussians_through_column,
        shared_fourier_calibration_priors_from_dataset,
        strip_shared_fourier_calibration_covariance,
    )
    from .physical_observables import (  # noqa: F401
        AngularDistributionReciprocalVectors,
        CentroidDifference,
        ImagingLayerBoundaries,
        ImagingLayerPeriods,
        ImagingReciprocalVectors,
        IndexedDiffractionBraggSpots,
        LayerCertificateCalibration,
        LayerCertificateObservableCovarianceAugmentation,
        NativeDiffractionReciprocalVectors,
        PhysicalObservableBlockSpecification,
        SingularObservableMapError,
        UnsupportedPhysicalObservableError,
        calibration_covariance_augmentations_for_physical_observables,
        calibration_priors_for_physical_observables,
        fourier_observable_covariance_augmentation,
        imaging_reciprocal_vectors_from_feature_set,
        indexed_diffraction_bragg_spots_from_feature_set,
        layer_certificate_calibrations_for_physical_observables,
        layer_certificate_covariance_augmentations_for_physical_observables,
        make_calibration_aware_physical_predictor,
        predict_physical_observable_blocks,
        specifications_from_aggregated_dataset,
        strip_shared_layer_certificate_covariance,
        strip_shared_multilayer_certificate_covariance,
        validate_physical_observable_schema,
    )
    from .calibration_covariance import (  # noqa: F401
        FourierObservableCovarianceAugmentation,
        deduplicate_calibration_priors,
        gaussian_prior_from_metadata,
        gaussian_prior_to_metadata,
        make_fourier_observable_covariance_augmentation,
    )
    from .critical_line_calibration import (  # noqa: F401
        ApertureEdgeScanDesign,
        ApertureEdgeScanPrediction,
        AxialPlaneCovarianceBlock,
        CONDENSER_PROBE_OBSERVABLE_NAMES,
        DEFLECTOR_SAMPLE_CENTRAL_DIFFERENCE_OBSERVABLE_NAMES,
        DEFLECTOR_SAMPLE_POSITION_DIFFERENCE_OBSERVABLE_NAMES,
        ENTRANCE_COVARIANCE_COORDINATE_NAMES,
        PHASE_SPACE_COVARIANCE_OBSERVABLE_NAMES,
        POSITION_COVARIANCE_OBSERVABLE_NAMES,
        CondenserIdentifiabilityAudit,
        CondenserProbeDesignSelection,
        DeflectorCentralDifference,
        DeflectorCommandKickMap,
        EntranceCovarianceChart,
        EntranceCovarianceNuisanceAudit,
        HardEdgeInternalPlaneBounds,
        HardEdgeInternalPlaneConstraint,
        audit_axial_plane_covariance_nuisance_identifiability,
        audit_condenser_identifiability,
        audit_entrance_covariance_nuisance_identifiability,
        aperture_edge_scan_from_moments,
        axial_plane_covariance_observables,
        axial_plane_covariance_observables_with_entrance_covariance,
        circular_gaussian_aperture_transmission,
        circular_perforation_signal,
        condenser_aperture_edge_scan,
        condenser_aperture_sweep,
        condenser_independent_reconnaissance_response,
        condenser_parameter_indices,
        condenser_reconnaissance_frame_observables,
        condenser_reconnaissance_response,
        central_deflector_command_difference,
        critical_line_scan,
        deflector_command_phase_space_moments,
        greedy_condenser_probe_design,
        hard_edge_matrix_between_planes,
        ideal_angular_kick_response_matrices,
        phase_space_covariance_observables,
        phase_space_probe_observables,
        restrict_hard_edge_internal_plane_bounds,
        restrict_hard_edge_internal_planes_bounds,
        sample_perforation_edge_scan,
        sample_plane_probe_observables,
        sample_plane_probe_observables_with_entrance_covariance,
        signed_wobble_response,
        three_normal_aperture_edge_design,
        validate_hard_edge_internal_plane_bounds,
        validate_hard_edge_internal_planes_bounds,
    )
    from .neoarm_diffraction import (  # noqa: F401
        BoundNativeDiffractionRequest,
        NativeDiffractionDesign,
        NativeDiffractionInfeasibleError,
        NativeDiffractionMode,
        bind_native_diffraction_requests,
        propagate_native_diffraction_gaussians,
        render_native_diffraction_detector,
        solve_native_diffraction_mode,
        solve_native_diffraction_modes,
        verify_native_diffraction_mode,
    )
    from .topology_validation import (  # noqa: F401
        AnalyticGaussianDetectorReference,
        FisherBiasBlockInfluence,
        FisherParameterStabilityReport,
        GaussianPerforationCertificate,
        NuisanceProfiledOpticalStabilityReport,
        PhysicalRenderSignature,
        PhysicalValidationCoverage,
        PhysicalValidationDecision,
        PhysicalValidationEvidence,
        PhysicalValidationLedger,
        PhysicalValidationRequirements,
        PhysicalValidationThresholds,
        analytic_single_gaussian_detector_reference,
        assess_fisher_parameter_stability,
        assess_nuisance_profiled_optical_stability,
        certify_single_gaussian_perforation,
        evaluate_physical_validation,
        stable_validation_signature,
    )
    from .grid_wave_reference import (  # noqa: F401
        GridWaveDiagnostics,
        GridWaveOptions,
        GridWaveReferenceResult,
        GridWaveSamplingError,
        RadialSymplecticCertificate,
        certify_radial_symplectic_map,
        round_solenoid_grid_reference,
        scaled_collins_czt_1d,
        scaled_collins_czt_2d,
        simpson_uniform_weights,
    )
    from .physical_nuisance import (  # noqa: F401
        BlockRankDiagnostics,
        BlockwiseIdentifiabilityDiagnostics,
        GaussianPriorFactor,
        GlobalObservationLayout,
        HeldOutBlockEvaluation,
        HeldOutLikelihoodEvaluation,
        NuisanceIdentifiabilityDiagnostics,
        SharedLatentNuisanceLikelihood,
        SignedDiscrepancyModes,
        diagnose_blockwise_identifiability,
        diagnose_blockwise_nuisance_identifiability,
        diagnose_nuisance_identifiability,
        evaluate_held_out_shared_likelihood,
    )
except Exception:
    pass
