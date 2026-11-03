# GNSS Orientation Optimizer

## Final System Output Report and Implementation Specification

**Project:** GNSS Orientation & Temporal Optimization Pipeline\
**Primary deliverable:** Offline-capable Android application integrating
the existing temporal C/N₀ and orientation-recommendation models\

------------------------------------------------------------------------

## 1. Purpose and Scope

The completed system brings together the existing GNSS data pipeline and
two trained machine-learning models in a single Android application. The
application presents live GNSS measurements, the phone's current
orientation, a model-derived orientation recommendation, and an
explanation of the features that distinguish the recommended orientation
from the current one.

The system has two connected parts:

1.  **Data and model preparation:** a DuckDB-based pipeline that
    produces the combined GNSS corpus used by the training and
    validation workflows.
2.  **Android application:** a native application that reads live phone
    sensors and GNSS measurements, prepares model inputs, runs both
    models, evaluates candidate orientations, and displays the results.

The application is intended to operate locally on the Android device
after installation. A network connection is not part of the live
inference path.

This report describes the expected final behaviour and output. It is
also an implementation contract for Hermes Agent. Where a fact depends
on the actual model artifacts, device, Android version, or runtime
measurements, the report deliberately refers to the value or behaviour
discovered during implementation instead of inventing one.

## 2. Final Deliverables

The completed project contains:

-   The corrected GNSS Status and OrientationDeg merge pipeline.
-   The resulting `combined_corpus.parquet` dataset.
-   Validation checks and tests for the merge pipeline.
-   The existing temporal C/N₀ model and orientation-recommendation
    model, in their original training formats and/or faithful mobile
    deployment formats.
-   Model export and numerical-equivalence validation utilities.
-   A native Android application written in Kotlin with Jetpack Compose.
-   GNSS and orientation sensor acquisition components.
-   A feature-engineering and model-inference layer.
-   A candidate-orientation evaluation and ranking component.
-   A top-three feature explanation component.
-   A physical-device APK build and installation procedure.
-   A `README_ANDROID.md` describing architecture, setup, permissions,
    model integration, coordinate conventions, and known limitations.
-   Automated tests and recorded validation results.

The report does not assign unverified performance scores, prediction
accuracy, battery consumption, latency, or real-world signal gains.
Those values belong in the implementation's measured test results.

------------------------------------------------------------------------

## 3. Data Pipeline Output

### 3.1 Combined corpus

The data pipeline produces:

`results/merged_data/combined_corpus.parquet`

The corpus combines satellite-level GNSS Status measurements with the
closest valid preceding OrientationDeg measurement, subject to the
project's session and timestamp-matching rules.

The output retains the raw fields needed by downstream model training
and future analysis. Derived features are created in the relevant model
preprocessing code unless the existing pipeline explicitly defines them
as part of the persisted corpus.

The corpus supports the two model workflows:

-   Temporal signal prediction, which uses current and historical signal
    measurements and satellite state.
-   Orientation evaluation, which uses satellite geometry, candidate
    antenna orientation, and the orientation-related physics features
    required by the trained model.

### 3.2 Session-safe association

Status and orientation measurements are associated within the same
recording session and, where the source data supports it, the same
device.

A timestamp match is not treated as sufficient evidence that two records
belong together. Session identity is preserved or reconstructed only
using information supported by the source data. Any ambiguity in session
reconstruction is documented rather than concealed.

The join selects a preceding orientation measurement for a Status
timestamp. The configured maximum time difference is applied
consistently. The implementation records how many Status records receive
a valid orientation match, how many remain unmatched, and how many are
outside the permitted time tolerance.

Whether unmatched Status records remain in the final corpus or are
excluded is an explicit pipeline decision documented in the
configuration and validation output. It is not an accidental side effect
of an inner join or a `WHERE` clause.

### 3.3 CSV and Parquet input handling

Both supported input paths use explicit, verified column mappings:

-   Parquet input is read using its inspected schema.
-   CSV input is read with the correct header handling and explicit
    column names/types where appropriate.

The CSV path does not depend on incorrect assumptions about
DuckDB-generated column names. Header rows are not accidentally included
as measurements, and valid data rows are not skipped.

### 3.4 Repeatable output generation

The merge process can be rerun without relying on manual deletion of the
previous output. It writes to a temporary output, validates that output,
and replaces the previous corpus only after successful completion.

A failed run does not intentionally destroy the last valid corpus.
Temporary files and DuckDB resources are cleaned up appropriately.

### 3.5 Resource behaviour

Large transformations remain in DuckDB rather than loading the entire
corpus into pandas or Python memory. DuckDB memory limits and a
temporary spill directory are configured for the available machine.

The implementation is described as memory-constrained or out-of-core
processing where applicable. It does not claim zero-RAM execution.

### 3.6 Data validation report

After the merge, the pipeline reports the information it actually
measured, including:

-   Input files discovered and formats detected.
-   Input row counts.
-   Detected device and session counts.
-   Output row count and file size.
-   Duplicate orientation timestamp counts.
-   Matched and unmatched Status records.
-   Records rejected by timestamp tolerance, if filtering is enabled.
-   Null and invalid values in important fields.
-   Timestamp ordering and range checks.
-   Schema compatibility with downstream training scripts.

These are runtime outputs. They are not hardcoded into this report.

------------------------------------------------------------------------

## 4. Model Responsibilities

The two models have distinct responsibilities. They are integrated into
one application, but they are not treated as interchangeable predictors.

### 4.1 Model 1 --- Temporal C/N₀ Prediction

The temporal model evaluates the recent history of satellite signal
measurements and predicts future C/N₀ behaviour at the horizons
supported by its trained artifact.

Its inputs are built from the live GNSS stream and a rolling history
maintained by the application. Depending on the model's actual feature
schema, these can include:

-   Current C/N₀.
-   Historical C/N₀ lags.
-   C/N₀ changes and slopes.
-   Satellite elevation and its changes.
-   Satellite azimuth and its changes.
-   Carrier-frequency-derived features.
-   Constellation indicators.
-   Receiver tracking information such as `UsedInFix`, when available
    and expected by the model.
-   Other features explicitly required by the trained artifact.

The app uses the model's actual feature order, preprocessing, target
definition, and supported prediction horizons. It does not invent
additional horizons or assume that all horizons documented in an earlier
experiment are present in the final artifact.

The temporal model provides context about the expected signal trend. It
can help the application decide whether the current reception is
predicted to remain stable or whether a fresh orientation evaluation is
useful.

A temporal forecast is not itself an orientation recommendation.

### 4.2 Model 2 --- Orientation Recommendation

The orientation model evaluates hypothetical phone/antenna orientations
using the currently available satellite geometry and the physics-derived
features expected by the trained model.

The EDA reference describes the core geometric representation as the dot
product between a satellite line-of-sight unit vector in ENU coordinates
and an antenna-normal boresight vector:

\[
`\cos`{=tex}(`\theta`{=tex}*{`\mathrm{inc}`{=tex}})=`\mathbf{s}`{=tex}*{`\mathrm{enu}`{=tex}}`\cdot`{=tex}`\mathbf{n}`{=tex}\_{`\mathrm{enu}`{=tex}}
\]

The reference line-of-sight convention is:

\[ `\mathbf{s}`{=tex}\_{`\mathrm{enu}`{=tex}} =
```{=tex}
\begin{bmatrix}
\cos(El)\sin(Az)\\
\cos(El)\cos(Az)\\
\sin(El)
\end{bmatrix}
```
\]

The application uses the same convention, angle units, rotation order,
and antenna-normal definition as the validated model implementation.
Android sensor axes are transformed into this convention rather than
passed through without conversion.

The orientation model does not receive the current `Cn0DbHz` as an input
when evaluating hypothetical candidate orientations. Current C/N₀
describes the phone's present physical orientation and would leak the
current outcome into a hypothetical candidate evaluation. It remains
available to the temporal model and to the app's measured-signal
display.

The EDA reference lists a 28-feature orientation feature set and an
example candidate-yaw grid. The final Android implementation must use
the exact feature schema and candidate dimensions supported by the
actual trained model. It must not assume that a yaw-only model can
optimize pitch and roll, or that a model trained with one orientation
convention can safely accept another.

### 4.3 Model integration contract

The integration layer:

1.  Collects current measurements and the required recent history.
2.  Builds the temporal model input using its own schema.
3.  Builds candidate-specific orientation model inputs using the
    orientation model's own schema.
4.  Applies the same preprocessing used during training.
5.  Executes the models using faithful on-device implementations.
6.  Validates output shape, finiteness, and expected ranges.
7.  passes valid predictions to the recommendation logic.

The models are not chained together unless the training and inference
design explicitly supports that dependency. The temporal model may
inform when an orientation search is run, while the orientation model
evaluates the candidate orientations. The integration report explains
the actual chosen orchestration.

------------------------------------------------------------------------

## 5. Android Application Overview

### 5.1 Application behaviour

The app opens to a single primary dashboard. It shows the latest
available measurements and model results, with clear distinctions
between:

-   Measured values from Android GNSS APIs.
-   Measured phone orientation from Android sensors.
-   Forecasts from the temporal model.
-   Predicted candidate performance from the orientation model.
-   The recommendation calculated from candidate predictions.
-   Explanations derived from model comparisons or feature attribution.

The dashboard does not display sample values, placeholder measurements,
or stale readings as if they were live. When data is unavailable, the
relevant field displays an explicit state such as "Waiting for GNSS",
"Permission required", or "Insufficient measurements".

### 5.2 Main dashboard sections

The primary screen contains the following sections.

#### A. GNSS status

Shows the current data-acquisition state and the availability of the
GNSS stream.

The section presents, when available:

-   GNSS status.
-   Number of visible satellites.
-   Number of satellites with usable/active signal measurements.
-   Constellation distribution.
-   Time of the most recent measurement update.

Satellite counts are calculated from the current Android GNSS callback
data. The app does not reuse dataset counts from offline EDA as live
values.

#### B. Current phone orientation

Shows the phone's measured orientation:

-   Yaw / azimuth.
-   Pitch.
-   Roll.

The displayed values follow the coordinate convention documented by the
app. The app also indicates when the relevant orientation sensor is
unavailable or its accuracy is insufficient for the intended
calculation.

Angle wrapping and device screen rotation are handled consistently. The
displayed orientation is the phone's current measured orientation, not
the candidate orientation selected by the optimizer.

#### C. Recommended orientation

Shows the candidate orientation selected by the orientation evaluation
process.

The section displays only the dimensions the model and optimizer
genuinely evaluate. If the deployed model supports candidate yaw only,
the app must not present pitch and roll as optimized outputs. It may
show the current measured pitch and roll separately.

The recommendation includes:

-   Recommended yaw, where supported.
-   Recommended pitch and roll only where the model and search genuinely
    support them.
-   The angular difference between the current and recommended
    orientation.
-   Predicted signal score or predicted C/N₀ summary used by the
    optimizer.
-   Estimated difference relative to the current-orientation prediction.
-   Recommendation freshness or last evaluation time.

The app describes this as a model recommendation. It does not claim that
the phone physically rotates itself.

#### D. Signal strength and quality

Shows signal measurements derived from the current GNSS stream.

The dashboard can present:

-   Mean C/N₀.
-   Median C/N₀.
-   Minimum and maximum C/N₀.
-   Number of satellites contributing to the summary.
-   Per-constellation summaries, where useful.
-   Signal quality category.

The quality category is based on a documented, configurable rule. If it
is a heuristic classification rather than a model output, the UI and
documentation identify it as a heuristic.

Only valid measurements are included in the relevant aggregates. The
aggregation policy for zero, null, missing, and invalid C/N₀ values is
documented.

The offline EDA report describes the historical active-signal
distribution, including a median of 32.0 dB-Hz and mean of 31.62 dB-Hz
for the analysed corpus. These are historical dataset statistics, not
live phone readings, model guarantees, or thresholds automatically
adopted by the application.

#### E. Top three recommendation features

Shows the three most influential feature contributions to the difference
between the current-orientation evaluation and the
recommended-orientation evaluation.

Each item includes:

-   Feature name in readable language.
-   Direction of contribution.
-   Contribution magnitude, when the attribution method provides a
    meaningful comparable value.
-   A short explanation of what the feature represents.

The list is calculated from the actual model evaluation. It is not a
static list of commonly important GNSS features and is not simply the
three largest raw input values.

Where exact SHAP-style attribution is unsupported or impractical, the
implementation may use controlled feature ablation or another documented
local comparison method. The UI labels the method accurately. The app
avoids presenting correlation as causation.

If the attribution method cannot produce a valid explanation for the
current evaluation, the section reports that an explanation is
unavailable rather than inventing one.

#### F. Model and data status

Shows whether the required components are ready:

-   GNSS data acquisition.
-   Orientation sensor acquisition.
-   Temporal model loading/inference.
-   Orientation model loading/inference.
-   Recommendation engine.
-   Last successful update.

Errors are visible and understandable. A failed model load does not
result in a fabricated recommendation.

------------------------------------------------------------------------

## 6. Recommendation and Evaluation Behaviour

### 6.1 Candidate evaluation

The app evaluates a candidate set supported by the orientation model.

The EDA reference describes an example yaw search using discrete
candidate angles. The final search resolution and dimensions are taken
from the actual inference implementation and device-use requirements.

A coarse-to-fine search may be used if it remains consistent with the
model's supported input space and does not create unsupported claims of
precision.

### 6.2 Objective function

Candidate orientations are ranked using a documented score derived from
the orientation model's predictions over the currently visible satellite
set.

The objective may account for the aggregate predicted reception across
satellites and any weighting policy already validated by the project. It
should not select an orientation solely because one satellite has an
unusually high prediction.

The implementation documents:

-   Which satellites are included.
-   How missing or invalid measurements are handled.
-   Whether satellites are weighted.
-   How predictions are aggregated.
-   How ties are resolved.
-   What happens when too few usable satellites are available.

The score is a model-derived comparison, not a direct measurement of
total received RF power unless the model explicitly defines it that way.

### 6.3 Current-versus-candidate comparison

The recommendation engine evaluates the current orientation using the
same model and feature-generation path used for candidate orientations.
This makes the comparison internally consistent.

The dashboard reports the difference between the current-orientation
prediction and the selected candidate prediction using the same units
and aggregation method.

If the model outputs per-satellite C/N₀ predictions, the aggregate is
described as an aggregate prediction. If it outputs another score, the
app preserves that score's meaning and units.

### 6.4 Recommendation stability

The recommendation is stabilized to prevent small measurement
fluctuations from causing continuous changes.

The implementation uses documented controls such as:

-   Smoothing of noisy inputs where appropriate.
-   A minimum improvement threshold.
-   Hysteresis before replacing an existing recommendation.
-   A minimum interval between full candidate searches.
-   A minimum-data requirement.

These controls are configurable and their selected values are recorded
in the implementation configuration. They are not assigned invented
values in this report.

### 6.5 Temporal model role

The temporal model evaluates signal trends using the history available
to the app. Its output can be used as a trigger or context signal for
refreshing orientation evaluation, consistent with the actual trained
model and project inference design.

A temporal forecast that predicts degradation does not prove that
changing orientation will fix the degradation. The app distinguishes
predicted signal trend from predicted orientation benefit.

------------------------------------------------------------------------

## 7. GNSS and Sensor Data Acquisition

### 7.1 GNSS measurements

The app uses Android's GNSS APIs to obtain the fields supported by the
device and operating-system version.

The acquisition layer collects, where available:

-   SVID.
-   Constellation type.
-   C/N₀.
-   Elevation.
-   Azimuth.
-   Carrier frequency.
-   Used-in-fix status.
-   Other fields only when exposed by the relevant Android API and
    required by the model.

The app does not fabricate unavailable fields. Missing carrier frequency
or other unavailable information is handled according to the model's
documented preprocessing policy.

`GnssStatus` and `GnssMeasurementsEvent` provide different levels of
information. The implementation uses each API only for fields it
actually exposes and checks device support before relying on it.

### 7.2 Orientation sensors

The app uses the available Android rotation/orientation sensors to
estimate device orientation. The selected sensor and fallback behaviour
are documented.

The coordinate conversion is validated against known device
orientations. It accounts for the Android device frame, screen rotation,
angle wrapping, and the ENU frame expected by the antenna-geometry
calculations.

Magnetic heading and true-north-referenced satellite azimuth are not
treated as interchangeable without a documented conversion. Any
limitations caused by unavailable declination or heading accuracy are
surfaced.

### 7.3 Rolling temporal history

A bounded in-memory history stores the recent measurements required by
the temporal model. It is keyed by the signal identity required by the
model, such as constellation, SVID, and carrier frequency, and handles
gaps or satellite reacquisition according to the training-time feature
rules.

The app does not use measurements from one satellite to fill the history
of another. It does not silently treat missing epochs as valid
observations.

------------------------------------------------------------------------

## 8. Model Deployment and Numerical Validation

### 8.1 On-device inference

The intended deployment runs both models locally on the Android device.
The exact runtime is selected after inspecting the trained artifacts.

Possible deployment formats include ONNX, TensorFlow Lite, or a faithful
native implementation, depending on model compatibility.

The mobile implementation preserves:

-   Feature names and order.
-   Data types.
-   Scaling and normalization.
-   Missing-value behaviour.
-   Categorical encoding.
-   Angle units.
-   Derived feature calculations.
-   Model output interpretation.

The app does not silently replace a trained model with a hand-written
approximation.

### 8.2 Reference inference

A Python reference inference path is retained for validation.
Representative input vectors are evaluated through the original model
and the Android-compatible implementation.

The comparison records:

-   Input feature parity.
-   Output shape.
-   Prediction differences.
-   Numerical tolerance used.
-   Any known conversion limitations.

Numerical equivalence is reported from actual test results. This report
does not claim that equivalence has already been achieved.

### 8.3 Model artifact integrity

The Android project includes the exact model artifacts used by the app
and records their source/version or checksum where practical. The export
process is reproducible and does not include the full training corpus in
the APK.

------------------------------------------------------------------------

## 9. Android Application Structure

The implementation is organised into separable responsibilities.

  -----------------------------------------------------------------------
  Component                           Responsibility
  ----------------------------------- -----------------------------------
  GNSS manager                        Registers callbacks and converts
                                      Android GNSS measurements into app
                                      data objects

  Orientation manager                 Reads sensors and provides
                                      orientation in the documented
                                      coordinate convention

  Feature engineering                 Builds model-specific inputs from
                                      current data and rolling history

  Temporal inference                  Runs the temporal model and
                                      interprets its outputs

  Orientation inference               Evaluates candidate-specific
                                      features with the orientation model

  Recommendation engine               Scores and ranks candidates and
                                      stabilizes the selected
                                      recommendation

  Explanation engine                  Calculates and formats the top
                                      three local feature contributions

  ViewModel                           Exposes application state through
                                      observable state streams

  Compose UI                          Displays measurements, predictions,
                                      recommendation, explanations, and
                                      status

  Model validation tools              Compare Android inference against
                                      the Python reference
  -----------------------------------------------------------------------

The UI is not responsible for model calculations or GNSS callback
processing. Sensor callbacks remain lightweight, and inference/search
work is performed away from the main UI thread.

------------------------------------------------------------------------

## 10. User Experience and Runtime States

The app has a single main dashboard with a restrained technical visual
style. The interface prioritises legibility and clear separation between
measured and predicted values.

The app provides explicit states for:

-   Initial startup.
-   Permission request.
-   Waiting for GNSS measurements.
-   GNSS available.
-   Insufficient satellite data.
-   Orientation sensor unavailable.
-   Model loading.
-   Model ready.
-   Inference in progress.
-   Recommendation available.
-   Recommendation unavailable.
-   Model or data-processing error.

The last-update time is shown for live values. When data stops arriving,
the app does not continue to present the last value as if it were
current.

The app requests only the Android permissions required for its
foreground functionality. It does not require a user account, cloud
backend, or network connection for normal inference.

------------------------------------------------------------------------

## 11. Physical Device Build and Installation

The final Android deliverable is built as a debug APK and installed on
the USB-connected Android device using ADB.

The deployment workflow:

1.  Detects the connected device with `adb devices -l`.
2.  Confirms that the device is authorised for USB debugging.
3.  Reads the device model and Android API level.
4.  Builds the app using the project's Gradle configuration.
5.  Installs or updates the APK on the selected device.
6.  Launches the application.
7.  Inspects runtime logs.
8.  Verifies permissions, sensor registration, GNSS callbacks, model
    loading, inference, and UI updates.

If more than one device is connected, the intended device is selected
explicitly by serial. A successful APK installation is not considered
proof that the application works; runtime verification is part of
completion.

The final report records the actual device details and installation
outcome obtained during deployment. No device model, Android version, or
installation status is assumed in advance.

------------------------------------------------------------------------

## 12. Testing and Verification

### 12.1 Data pipeline tests

Tests cover:

-   Parquet input.
-   CSV fallback.
-   Correct CSV column mapping and header handling.
-   Session and device isolation.
-   Timestamp matching and tolerance.
-   Missing orientation measurements.
-   Duplicate orientation timestamps.
-   Empty or missing input directories.
-   Existing output handling.
-   Paths containing spaces or apostrophes.
-   Output schema compatibility.

Synthetic data is used for deterministic edge-case tests. Real-corpus
validation is recorded separately and is only reported if it is actually
run.

### 12.2 Model tests

Tests cover:

-   Model artifact loading.
-   Feature ordering.
-   Preprocessing parity.
-   Missing-value handling.
-   Finite outputs.
-   Candidate evaluation.
-   Current-versus-candidate comparison.
-   Explanation generation.
-   Python-versus-mobile numerical comparison.

### 12.3 Android tests

Tests cover:

-   Permission granted and denied.
-   GNSS callback registration and cleanup.
-   Sensor availability and fallback.
-   Coordinate conversion.
-   Screen rotation and app lifecycle.
-   Missing GNSS data.
-   Model-loading errors.
-   Inference failures.
-   Recommendation stability.
-   UI state updates.

### 12.4 Physical-device checks

On the connected phone, verification checks that:

-   The displayed orientation changes in response to physical phone
    movement.
-   GNSS satellite and signal values come from actual device callbacks.
-   Both model artifacts load.
-   The inference pipeline executes using live or explicitly marked test
    input.
-   Candidate evaluation returns valid results when sufficient data is
    available.
-   The top-three explanation reflects the current comparison.
-   Missing data and errors are visible rather than replaced with
    fabricated values.

A controlled before-and-after orientation experiment may be recorded
separately. Model-predicted improvement and measured post-rotation
changes are kept distinct.

------------------------------------------------------------------------

## 13. Known Boundaries and Interpretation

The following distinctions are part of the intended final product:

1.  **Prediction is not measurement.** A model's predicted C/N₀ or score
    is not displayed as an actual GNSS measurement.
2.  **Recommendation is not actuation.** The phone does not physically
    rotate. The user acts on the displayed guidance.
3.  **Candidate dimensions follow model capability.** A yaw-only model
    is not represented as a full three-axis optimizer.
4.  **Feature attribution is method-dependent.** Explanations are
    labelled according to the method actually implemented.
5.  **Android hardware support varies.** The app only uses fields and
    sensors exposed by the connected device and OS.
6.  **GNSS measurements vary with environment.** Satellite visibility,
    multipath, antenna placement, motion, and receiver behaviour can
    affect live measurements.
7.  **Historical EDA statistics are not live thresholds.** Dataset
    summaries are context, not hardcoded runtime readings.
8.  **Session reconstruction has limits.** If the source data lacks
    reliable session metadata, the pipeline documents the assumptions
    and ambiguity.
9.  **The model's training assumptions govern deployment.** The app does
    not extend the model's supported feature space without validation.

------------------------------------------------------------------------

## 14. Implementation Decisions Hermes Agent Must Resolve from the Repository

The following items must be confirmed from the real code and artifacts
before implementation is considered complete:

-   Exact model filenames, serialization formats, and versions.
-   Exact temporal prediction horizons and target definitions.
-   Exact input feature names, order, data types, and preprocessing for
    both models.
-   Whether the orientation model searches yaw only or supports
    additional candidate dimensions.
-   Whether pitch and roll are inputs, fixed assumptions, or used only
    to derive geometric features.
-   Exact antenna-normal rotation matrix and Euler rotation order.
-   Whether azimuth is referenced to magnetic north or true north in
    each data source.
-   Which Android GNSS API provides every required runtime field.
-   How missing carrier frequency, `UsedInFix`, and other unavailable
    fields are handled.
-   The candidate scoring and satellite weighting policy.
-   The method used to generate the top-three feature explanations.
-   Whether the temporal model gates orientation evaluation or simply
    provides additional context.
-   The policy for unmatched Status/orientation rows in the merged
    corpus.
-   How session and device identity are represented in each input
    format.

These are implementation decisions, not opportunities to guess. Hermes
Agent must inspect the source code, schemas, and artifacts, document the
chosen behaviour, and add tests for the critical assumptions.

------------------------------------------------------------------------

## 15. Final Acceptance Criteria

The project is considered complete when the repository contains a
working data pipeline and a native Android application that satisfies
all of the following:

### Data and model readiness

-   The merged corpus is generated with correct CSV and Parquet
    handling.
-   Session/device association is protected against cross-session
    matches wherever source metadata permits.
-   Timestamp matching and unmatched-row behaviour are explicit.
-   Output replacement is safe.
-   The final schema is compatible with the actual training scripts.
-   Both trained models can be loaded through the chosen deployment
    path.
-   Android preprocessing is validated against the Python reference.

### Application behaviour

-   The dashboard shows actual current phone orientation.
-   The dashboard shows actual available GNSS measurements and signal
    summaries.
-   The temporal model runs with the feature history it requires.
-   The orientation model evaluates candidates within its supported
    input space.
-   The app displays a recommendation and a clearly labelled predicted
    comparison.
-   The app displays up to three valid explanation features, derived
    from the implemented attribution method.
-   Missing data, unavailable sensors, and inference failures are
    clearly represented.
-   The app remains responsive during data acquisition and inference.

### Physical-device deployment

-   The APK builds successfully.
-   The APK is installed on the connected physical Android device using
    ADB.
-   The application launches.
-   Runtime logs are inspected.
-   Sensor, GNSS, model, and UI behaviour are verified on the device.

### Documentation

-   Build and installation instructions are present.
-   Model integration and feature definitions are documented.
-   Coordinate conventions are documented.
-   Optimization and explanation methods are documented.
-   Tests and measured validation results are recorded honestly.
-   Remaining limitations are stated clearly.

------------------------------------------------------------------------

## 16. Final Output Summary

The final user-facing output is a compact Android GNSS dashboard that
combines live satellite measurements, measured phone orientation,
temporal signal forecasts, and candidate-orientation evaluation.

The user can see the phone's current orientation, the model's
recommended orientation within the model's supported search space, the
predicted difference between the current and recommended evaluations,
live signal-strength summaries, and the three most influential features
explaining the recommendation.

Behind that screen, the data pipeline preserves the relationship between
Status and OrientationDeg measurements, the model integration maintains
the training-time feature definitions, and the Android application
handles sensor acquisition, inference, optimization, and display as
separate components.

The final implementation is judged by verified behaviour on the
connected physical Android device and by reproducible tests---not by the
presence of source files or a successful APK build alone.
