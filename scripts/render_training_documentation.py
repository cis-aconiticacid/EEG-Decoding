from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "Training"
CUSTOM_ARCHITECTURE_IDS = {"D-095", "D-096", "D-097", "D-098", "D-099"}


@dataclass(frozen=True)
class Experiment:
    title: str
    purpose: str
    objective: str
    subjects: str
    temporal: str
    status: str
    nodes: tuple[str, ...]
    source: tuple[str, ...]
    results: tuple[str, ...]
    interpretation: str


EXPERIMENTS: dict[str, Experiment] = {
    "D-061": Experiment(
        "Multi-Subject Paper-Latent Alignment",
        "Learn an EEG representation aligned to frozen image latents and evaluate nearest-image retrieval.",
        "Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.",
        "Sixteen subjects, fitted independently for each subject/task combination.",
        "Not excluded: training uses the first 30 source-order images per class and evaluation uses the last 20.",
        "Completed historical runs are documented.",
        ("EEG 62 x 400\n40--440 ms", "Per-subject normalization\nfirst 30 images/class", "Waveform encoder\n4 Transformer blocks", "256 image-latent queries\n1024-D targets", "Latent MSE + contrastive\nnearest-gallery readout"),
        ("scripts/run_d061_paper_latent.py", "config/d061_paper_latent_s17.json"),
        ("Recorded epoch-70 class accuracies include 39.562% for subject 0, 30.125% for subject 1, and 33.734% for subject 2.", "The public repository does not distribute the underlying checkpoints or raw result JSON files."),
        "These are image-assisted retrieval results from independent subject fits. The source-order 30/20 split can retain temporal-order effects.",
    ),
    "D-063": Experiment(
        "Raw and Local-Frequency Classification Probe",
        "Compare raw-waveform, frequency-only, and fused inputs for 80-class EEG classification.",
        "Ordinary 80-way cross-entropy classification.",
        "Subject 0 only.",
        "Not excluded: the first 30 source-order images per class train the probe and the last 20 form the test set.",
        "Three-arm, three-seed probe completed.",
        ("EEG 62 x 400\n40--440 ms", "Raw branch and Hann PSD\n2.5--80 Hz", "Electrode + coordinate tokens\n128-D", "2-layer Transformer\n4 heads", "80-way linear head\ncross-entropy"),
        ("scripts/run_d063_local_frequency_probe.py", "config/d063_local_frequency_probe.json"),
        ("Indexed run rows include 16.563% and 38.063% test accuracy; the complete arm-by-seed table remains in the local result archive.",),
        "This experiment directly classifies labels. Its fixed source-order split does not rule out temporal-order confounding.",
    ),
    "D-064": Experiment(
        "Masked Frequency Pretraining and Linear Probe",
        "Measure how frequency-patch masking affects a self-supervised encoder and frozen linear-probe performance.",
        "Masked reconstruction pretraining followed by an 80-way linear probe; probe training uses cross-entropy.",
        "Subject 0 only.",
        "Partially controlled inside the first 30: positions 0--23 fit and 24--29 validate; the official last 20 are excluded from evaluation.",
        "Mask-ratio and seed sweep completed.",
        ("EEG 62 x 400", "Hann log-power\nfrequency patches", "Random patch masking\nmasked reconstruction", "Frozen encoder\nmean representation", "Linear 80-class probe\n24/6 development split"),
        ("scripts/run_d064_mask_ratio.py", "config/d064_local_mask_ratio.json"),
        ("For mask ratio 0.15, recorded validation accuracies across seeds 17, 23, and 41 were 4.792%, 5.625%, and 5.000%.", "The official last-20-per-class test split received zero evaluations in this experiment."),
        "The reported values are development linear-probe metrics, not official test scores.",
    ),
    "D-066": Experiment(
        "Latent Fine-Tuning from a Masked EEG Encoder",
        "Fine-tune the masked EEG encoder to predict image latents and evaluate image-gallery retrieval.",
        "Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.",
        "Subject 0 only, with three random seeds.",
        "Not excluded: the first 30 source-order images per class train the model and the last 20 are evaluated.",
        "Three seed runs completed at epoch 70.",
        ("D064 masked encoder", "Full unmasked EEG\n62 channels", "Fine-tuned Transformer\n128-D tokens", "256-query image decoder\n1024-D latent grid", "Latent objective\nnearest-gallery retrieval"),
        ("scripts/run_d066_latent_finetune.py", "config/d066_latent_finetune.json"),
        ("Recorded test class accuracies were 37.125%, 38.312%, and 36.125% for seeds 17, 23, and 41.", "Same-gallery training accuracy was about 94%, indicating a large train/test gap."),
        "This is embedding alignment with retrieval readout. The 30/20 source-order split leaves temporal-order effects possible.",
    ),
    "A-077": Experiment(
        "Pairwise Identity and Temporal-Position Diagnostics",
        "Test whether PSD differences predict subject identity (T1) or class-block position proximity (T3).",
        "RBF SVC and random-forest binary classification; no neural embedding objective.",
        "All sixteen subjects.",
        "Mixed: T1 uses image-disjoint 48/16/16 splits; T3 deliberately measures position-order effects. T2 is omitted because no session label exists.",
        "T1 and T3 completed; T2 intentionally unavailable.",
        ("EEG 40--440 ms", "Hann log10 PSD\n2.5--80 Hz", "Matched trial pairs\nabsolute feature difference", "RBF SVC or random forest", "T1 identity / T3 position\ngrouped evaluation"),
        ("scripts/a077_prepare_pair_tasks.py", "scripts/a077_run_models.py", "config/a077_pair_tasks.json"),
        ("T1 test balanced accuracy was 89.550% for the SVC and 86.150% for the random forest.", "T3 random-forest test balanced accuracy was 54.900%, close to but above the 50% reference.", "T2 has no score because the archive contains no verified session identifier."),
        "T1 supports subject-identifying signal in the constructed PSD pairs. T3 explicitly probes order proximity and must not be presented as session decoding.",
    ),
    "D-078": Experiment(
        "A0--A3 Continuation and Augmentation Study",
        "Continue the D066 latent model while comparing four waveform-augmentation arms.",
        "Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.",
        "Subject 0 only.",
        "Not excluded: the historical first-30/last-20 source-order split is retained.",
        "Four warm-start continuation arms completed through epoch 120.",
        ("D066 epoch-70 parent", "A0--A3 waveform\naugmentation arms", "Shared latent encoder\nseparate optimizers", "+50 continuation epochs", "Image-latent retrieval\nfixed diagnostics"),
        ("scripts/d078_run_continuation.py", "src/d078_augmentation.py", "config/d078_a100_continuation.json"),
        ("Epoch-120 test class accuracy was 37.188% for A0, 37.312% for A1, and 37.062% for A2 in the indexed records.",),
        "The differences are small, single-seed, warm-start comparisons and should not be interpreted as robust augmentation effects.",
    ),
    "D-079": Experiment(
        "Coordinate and Early-Fusion Ablation",
        "Ablate coordinate bias and add a learned patch-by-channel table during continuation from D066.",
        "Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.",
        "Subject 0 only.",
        "Not excluded: the historical first-30/last-20 source-order split is retained.",
        "B0/B1/B2 warm-start arms completed through epoch 120.",
        ("D066 epoch-70 parent", "B0 baseline / B1 no coordinates", "B2 learned table\n16 patches x 62 channels", "Separate continuation\noptimizers", "Latent retrieval\nfixed checkpoints"),
        ("scripts/d079_run_no_coord_early_fusion.py", "src/d079_architecture.py", "config/d079_no_coord_early_fusion.json"),
        ("The indexed parent checkpoint metrics were 37.125% for B0, 19.875% for B1, and 2.938% for B2 before their continuation updates.", "The local result archive, not this public repository, contains the complete epoch-120 comparison."),
        "This is a warm-start architecture ablation. Parent-state and source-order effects must be considered when comparing arms.",
    ),
    "D-080": Experiment(
        "Coordinate Ablations from Random Initialization",
        "Repeat B0/B1/B2 without pretrained EEG-model weights to separate architecture effects from warm-start effects.",
        "Image-latent MSE plus 0.1 symmetric contrastive loss; not cross-entropy classification.",
        "Subject 0 only.",
        "Not excluded: training uses the first 30 source-order images per class and diagnostics use the last 20.",
        "Three fixed 70-epoch arms completed.",
        ("Shared seed-17 random state", "B0 / B1 / B2\narchitecture variants", "Frequency-token Transformer", "256-query image decoder", "70-epoch latent training\nfixed diagnostics"),
        ("scripts/d080_run_from_scratch.py", "src/d080_from_scratch.py", "config/d080_from_scratch_no_coord.json"),
        ("At epoch 0, recorded test accuracies were 0.312% for B0, 1.375% for B1, and 1.125% for B2; these are initialization diagnostics, not trained endpoints.", "The complete epoch-70 results remain in the local result archive."),
        "This design removes trained-model inheritance but still uses the source-order 30/20 evaluation protocol.",
    ),
    "D-083": Experiment(
        "Synthetic CUDA Feasibility Check",
        "Verify that the D081 joint waveform/frequency pipeline can execute its two training stages on CUDA.",
        "Stage-A masked reconstruction and Stage-B latent alignment on synthetic tensors.",
        "No real subject data; synthetic feasibility only.",
        "Not applicable: no EEG-ImageNet train/test split is used.",
        "Feasibility check passed; this is not an accuracy experiment.",
        ("Synthetic EEG tensors", "Joint waveform + frequency tokens", "Stage A masked reconstruction", "Stage B latent alignment", "CUDA forward/backward\nfinite-loss checks"),
        ("scripts/d083_local_gpu_feasibility.py", "src/d081_joint_waveform_frequency.py", "src/d081_training.py"),
        ("The stored feasibility record reported Stage-B loss 1.46881.", "No real-data accuracy or retrieval metric was produced."),
        "A finite synthetic loss verifies executable plumbing only; it is not evidence of EEG decoding performance.",
    ),
    "D-085": Experiment(
        "Exploratory Riemannian SVM Record",
        "Preserve an exploratory covariance/tangent-space subject-classification result from a non-EEG-ImageNet dataset.",
        "Covariance features, tangent-space projection, and SVM classification.",
        "Per-subject evaluation on the exploratory external dataset.",
        "Not applicable to the EEG-ImageNet first-30/last-20 protocol.",
        "Historical record only; no public runnable source is retained in this folder.",
        ("External exploratory EEG", "Epoch covariance matrices", "Riemannian reference mean", "Tangent-space projection", "Per-subject SVM"),
        (),
        ("The historical record reports mean subject test accuracy 41.690%, median 39.656%, and range 26.562%--65.938%.",),
        "This result is not an EEG-ImageNet result and must not be compared directly with the other numbered experiments.",
    ),
    "D-088": Experiment(
        "Convolution-Routing Comparison",
        "Compare four local/global convolution routes under masked pretraining and supervised fine-tuning.",
        "Masked JEPA-style pretraining, then 80-way cross-entropy classification.",
        "Subject 0 only.",
        "Not excluded: 27/3 development trials come from the first 30 source-order images; the last 20 form a one-time test.",
        "Four-route benchmark completed.",
        ("EEG 62 x 400", "16 waveform patches\nfrequency side input", "Local / global / separate / shared convolution", "Masked pretraining\nthen classifier fine-tuning", "80-way development selection\none-time test"),
        ("scripts/run_d088_subject0_conv_routing.py", "src/d088_dual_attention.py"),
        ("The indexed summary records best development loss 2.87269 for local convolution and 3.14535 for global convolution.", "Corresponding stored test losses were 5.71287 and 5.50517; route accuracies are not reproduced in the public tree."),
        "Development selection and final testing are separated, but the split still follows source order.",
    ),
    "D-089": Experiment(
        "Masked Waveform Pretraining and Retrieval",
        "Pretrain convolution-routing encoders with masked waveform reconstruction, then align them to image latents for retrieval.",
        "Masked waveform value/slope loss followed by latent MSE, cosine, and contrastive losses; no classification head.",
        "Subject 0 only.",
        "Not excluded: 27/3 development trials are drawn from the first 30; the official last 20 are sealed until final evaluation.",
        "Checkpointed implementation; no public scalar result record.",
        ("EEG 40--440 ms", "Waveform + PSD patch tokens", "Four convolution routes", "Masked value + slope pretraining", "Image-latent retrieval\nno classifier head"),
        ("scripts/run_d089_local_curve_pretrain.py", "scripts/run_d089_corrected_masked_retrieval.py", "src/d089_masked_retrieval.py"),
        ("No accuracy or loss value is claimed here because the public bundle contains no result summary for this experiment.",),
        "The code defines a leak-aware development/test policy, but a checkpoint alone is not a reported result.",
    ),
    "D-092": Experiment(
        "Sample-Efficient Latent Post-Training",
        "Adapt the D089 local-convolution encoder to image latents with a small negative-target set and staged unfreezing.",
        "Latent MSE, cosine, contrastive, and consistency losses; no classification head.",
        "Subject 0 only.",
        "Development only: 27/3 trials from the first 30; official last-20 EEG is never forwarded.",
        "Checkpointed implementation; no public scalar result record.",
        ("D089 local-conv checkpoint", "27/3 development EEG", "Phase A latent head", "Phase B last-block unfreezing", "Sample-efficient latent retrieval\nno test forward"),
        ("scripts/run_d092_sample_efficient_posttrain.py", "src/d092_sample_efficient_latent.py", "config/d092_sample_efficient_posttrain.json"),
        ("No scalar result is claimed because the public bundle contains no result summary.",),
        "The configuration explicitly prevents test EEG forwarding during development.",
    ),
    "D-093": Experiment(
        "Stage-1 Classification Baseline",
        "Train an 80-class classifier on the D089 encoder and compare classification with latent-retrieval experiments.",
        "80-way cross-entropy with label smoothing 0.05.",
        "Subject 0 only.",
        "Not excluded: 27/3 development trials are drawn from the first 30, followed by refit on all 30 and one test on the last 20.",
        "Completed with one official test evaluation.",
        ("D089 local-conv encoder", "27/3 development split", "Learned pooling + MLP head", "Cross-entropy development selection", "Refit on first 30\none test on last 20"),
        ("scripts/run_d093_stage1_classification_baseline.py", "src/d093_stage1_classifier.py", "config/d093_stage1_classification_baseline.json"),
        ("Best development accuracy was 7.917% at epoch 14.", "After refit, official test accuracy was 3.187% with test loss 4.38329."),
        "The held-out evaluation is explicit, but it remains a single-subject, source-order split result.",
    ),
    "D-094": Experiment(
        "Multi-Scale Temporal/Spatial CNN",
        "Train the requested multi-scale temporal and spatial CNN for 80-class EEG decoding.",
        "80-way cross-entropy with label smoothing 0.05.",
        "Subject 0 only.",
        "Development only: 27/3 trials from the first 30; the official last 20 receive zero classifier forwards.",
        "Development run completed; official test remains sealed.",
        ("EEG 62 x 400\n40--440 ms", "Anti-aliased 250-Hz resampling\n62 x 100", "Per-electrode kernels\n36/68/132/260 ms", "Spatial 62-to-64 convolution\n2 depthwise temporal blocks", "10 segments: mean + variance\n128-hidden MLP to 80 classes"),
        ("scripts/run_d094_multiscale_cnn_stage1.py", "src/d094_multiscale_cnn.py", "config/d094_multiscale_cnn_stage1.json"),
        ("The selected epoch-50 development checkpoint achieved 9.167% validation accuracy (22/240), macro accuracy 9.167%, and loss 4.8633.", "The official last-20-per-class test set was not evaluated."),
        "The development score exceeds the 1.25% chance reference but is not a held-out test result.",
    ),
    "D-095": Experiment(
        "Hybrid Temporal/Spatial Classifier",
        "Combine aligned waveform and local-frequency tokens in a hybrid temporal/spatial classifier.",
        "Ordinary 80-way cross-entropy without label smoothing.",
        "Subject 0 only.",
        "Not excluded: the first 30 source-order images per class train the model and the last 20 form the official test.",
        "Fixed 100-epoch baseline and ablations completed.",
        ("EEG 62 x 400\n40--440 ms", "Waveform multiscale conv\n+ aligned 12--80-Hz FFT", "Learned gated fusion\n192-D tokens", "4 temporal/spatial attention blocks", "Electrode pooling\n3072-to-80 classifier"),
        ("model.py", "run.py", "config/config.json"),
        ("Recorded test accuracy: 42.188% for unified192, 45.875% with the 18--28 Hz exclusion, 14.750% without the frequency branch, and 48.688% with the optional post-fusion projection.", "All reported runs used one seed and reached 100% training accuracy."),
        "The projection-enabled variant has the highest recorded score, but single-seed ablations do not establish a robust causal effect.",
    ),
    "D-096": Experiment(
        "Position-Aware Waveform/Frequency Cross-Attention",
        "Replace D-095's scalar frequency gate with directional, position-aware cross-attention.",
        "Ordinary 80-way cross-entropy without label smoothing.",
        "Subject 0 only.",
        "Not excluded: the first 30 source-order images per class train the model and the last 20 form the official test.",
        "One fixed 100-epoch A100 run completed; test accuracy was 46.8125% (749/1600).",
        ("EEG 62 x 400\n40--440 ms", "Waveform and aligned\n12--80-Hz frequency tokens", "Position-aware cross-attention\nwaveform queries frequency", "4 temporal/spatial\nattention blocks", "Electrode pooling\n3072-to-80 classifier"),
        ("model.py", "run.py", "config/config.json"),
        ("The seed-17 run reached 100% training accuracy and 46.8125% official test accuracy (749/1600) in 304.704 seconds.", "Mean attention entropy was 96.40% of the maximum log(16), and mean diagonal mass was 0.06277, close to the uniform 1/16 reference.", "D-096 was 4.625 percentage points above D-095 unified192 but 1.875 points below D-095 postfusion192; all are single-seed observations."),
        "The architecture preserves separate signal streams until directional cross-attention, but the nearly uniform averaged routing statistics show little time-selective alignment. Attention weights remain routing diagnostics rather than causal attribution.",
    ),
    "D-097": Experiment(
        "Frequency Mask x Post-Cross-Position Factorial Ablation",
        "Measure the separate and combined effects of masking retained frequency bins below 25 Hz and removing the post-cross-attention position injection.",
        "Ordinary 80-way cross-entropy without label smoothing.",
        "Subject 0 only.",
        "Not excluded: the first 30 source-order images per class train the model and the last 20 form the official test.",
        "Four fixed 100-epoch A100 groups use the same seed and training protocol; results are recorded after endpoint completion.",
        ("EEG 62 x 400\n40--440 ms", "Waveform + aligned\n12--80-Hz tokens", "Factor A: mask bins\nbelow 25 Hz", "Factor B: post-cross\nposition on/off", "Electrode pooling\n3072-to-80 classifier"),
        ("model.py", "run.py", "config/full_frequency_with_position.json", "config/masked_below25_with_position.json", "config/full_frequency_without_post_cross_position.json", "config/masked_below25_without_post_cross_position.json"),
        ("All groups retain position in Q/K; only the post-cross z_s+P injection is toggled.", "The frequency factor zeros 15.625, 19.53125, and 23.4375 Hz bins before frequency projection.", "The four one-seed endpoint metrics are reported together with descriptive main effects and interaction."),
        "This is a controlled implementation ablation within D-096; it does not establish a general causal effect beyond the fixed subject, split, seed, and budget.",
    ),
    "D-098": Experiment(
        "One Independent Cross-Attention Model per Participant",
        "Train the D-097 winning architecture independently for all sixteen EEG-ImageNet participants.",
        "Ordinary 80-way cross-entropy without label smoothing.",
        "Participants 0--15, with one independently initialized model per participant.",
        "The first 30 source-order images per available class train each participant model; the remaining 20 form its one-time endpoint test.",
        "All sixteen fixed 100-epoch A100 runs completed; mean participant accuracy is 37.0685% and pooled sample accuracy is 37.1026%.",
        ("Per-participant EEG\n62 x 400", "Waveform + masked\nfrequency tokens", "Position-aware cross-attention", "4 temporal/spatial\nattention blocks", "Independent checkpoint\nper participant"),
        ("model.py", "run.py", "config/config.json"),
        ("All sixteen participants use the same seed, optimizer, architecture, and fixed epoch budget.", "Participants 2 and 12 are missing one and two complete classes in the verified official archives and are evaluated on 79 and 78 available classes.", "Per-participant and aggregate metrics are recorded after all endpoint runs complete."),
        "These are participant-specific single-seed models. Across-participant dispersion does not replace repeated-seed uncertainty.",
    ),
    "D-099": Experiment(
        "Participant 12 Full-Frequency Control",
        "Repeat the D-098 participant-12 model while restoring all retained 12--80 Hz frequency bins.",
        "Ordinary 80-way cross-entropy without label smoothing.",
        "Participant 12 only; the official archive contains 78 available classes.",
        "The first 30 source-order images per available class train the model; the remaining 20 form the official test.",
        "The fixed epoch-100 endpoint reached 17.8846%; retrospective checkpoint comparison peaked at 17.9487% at epoch 50.",
        ("Participant-12 EEG\n62 x 400", "Waveform + all 17\nfrequency bins", "Position-aware cross-attention", "4 temporal/spatial\nattention blocks", "80-class endpoint"),
        ("../D-098/model.py", "run.py", "config/config.json"),
        ("Only the below-25-Hz frequency mask changes relative to D-098 participant 12.", "Full frequency improves accuracy at all four saved checkpoints.", "Epoch 50 is selected retrospectively on the test set and is not an untouched held-out estimate."),
        "This is a single-seed participant-specific control. The repeated test-set checkpoint comparison is exploratory.",
    ),
}


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def diagram_tex(experiment_id: str, experiment: Experiment) -> str:
    nodes = []
    arrows = []
    for index, label in enumerate(experiment.nodes):
        lines = r"\\".join(tex_escape(part) for part in label.split("\n"))
        nodes.append(f"\\node[block] (n{index}) at ({index * 5.3},0) {{{lines}}};")
        if index:
            arrows.append(f"\\draw[arrow] (n{index - 1}) -- (n{index});")
    return "\n".join(
        [
            r"\documentclass[tikz,border=10pt]{standalone}",
            r"\usepackage{fontspec}",
            r"\setmainfont{Latin Modern Roman}",
            r"\usetikzlibrary{arrows.meta}",
            r"\begin{document}",
            r"\begin{tikzpicture}[block/.style={draw=blue!55!black,fill=blue!4,rounded corners=3pt,align=center,text width=4.35cm,minimum height=2.7cm,inner sep=6pt,font=\small},arrow/.style={-{Stealth[length=2.5mm]},line width=1pt,draw=blue!65!black}]",
            f"\\node[font=\\Large\\bfseries,anchor=south west] at (-2.1,2.0) {{{tex_escape(experiment_id + ': ' + experiment.title)}}};",
            *nodes,
            *arrows,
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )


def render_tex(tex_path: Path, png_path: Path) -> None:
    xelatex = shutil.which("xelatex")
    pdftoppm = shutil.which("pdftoppm")
    if not xelatex or not pdftoppm:
        raise RuntimeError("xelatex and pdftoppm are required to render architecture diagrams")
    with tempfile.TemporaryDirectory(prefix="eeg-architecture-") as temp_dir:
        temp = Path(temp_dir)
        subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={temp}", str(tex_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            [pdftoppm, "-singlefile", "-png", "-r", "160", str(temp / "ARCHITECTURE.pdf"), str(png_path.with_suffix(""))],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def links(paths: tuple[str, ...]) -> str:
    if not paths:
        return "- No runnable public source is retained for this historical record."
    return "\n".join(f"- [`{path}`]({path})" for path in paths)


def write_experiment(experiment_id: str, experiment: Experiment) -> None:
    directory = TRAINING_ROOT / experiment_id
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    for path in experiment.source:
        if not (directory / path).is_file():
            raise FileNotFoundError(directory / path)

    if experiment_id not in CUSTOM_ARCHITECTURE_IDS:
        (directory / "ARCHITECTURE.tex").write_text(
            diagram_tex(experiment_id, experiment), encoding="utf-8"
        )
    elif not (directory / "ARCHITECTURE.tex").is_file():
        raise FileNotFoundError(directory / "ARCHITECTURE.tex")
    render_tex(directory / "ARCHITECTURE.tex", directory / "architecture.png")

    if experiment_id not in CUSTOM_ARCHITECTURE_IDS:
        kind_note = (
            "`A` stands for **Analysis**: this numbered record evaluates diagnostic structure "
            "in EEG features rather than training a neural decoding model.\n\n"
            if experiment_id.startswith("A-")
            else ""
        )
        (directory / "README.md").write_text(
            f"""# {experiment_id}: {experiment.title}

{kind_note}{experiment.purpose}

## Experiment contract

- Training purpose: {experiment.purpose}
- Objective: {experiment.objective}
- Subject scope: {experiment.subjects}
- Temporal-effect control: {experiment.temporal}
- Status: {experiment.status}

## Documentation

- [Rendered architecture and explanation](ARCHITECTURE.md)
- [Results analysis](RESULTS_ANALYSIS.md)
- [TeX architecture source](ARCHITECTURE.tex)

## Source and configuration

{links(experiment.source)}

Data, generated runs, and checkpoints are intentionally excluded from the public repository. Training entry points resolve shared inputs from the repository-level `data/` directory and download the official EEG archives when they are absent.
""",
            encoding="utf-8",
        )
    elif not (directory / "README.md").is_file():
        raise FileNotFoundError(directory / "README.md")

    if experiment_id not in CUSTOM_ARCHITECTURE_IDS:
        pipeline = "\n".join(
            f"{index}. {node.replace(chr(10), ' — ')}"
            for index, node in enumerate(experiment.nodes, 1)
        )
        (directory / "ARCHITECTURE.md").write_text(
            f"""# {experiment_id} Architecture

![{experiment_id} architecture](architecture.png)

The image above is rendered from [`ARCHITECTURE.tex`](ARCHITECTURE.tex).

## Data flow

{pipeline}

## Training interface

- Objective: {experiment.objective}
- Subject scope: {experiment.subjects}
- Temporal-effect control: {experiment.temporal}

## Authoritative implementation

{links(experiment.source)}
""",
            encoding="utf-8",
        )
    elif not (directory / "ARCHITECTURE.md").is_file():
        raise FileNotFoundError(directory / "ARCHITECTURE.md")

    if experiment_id not in CUSTOM_ARCHITECTURE_IDS:
        result_lines = "\n".join(f"- {item}" for item in experiment.results)
        (directory / "RESULTS_ANALYSIS.md").write_text(
            f"""# {experiment_id} Results Analysis

## Recorded outcome

{result_lines}

## Interpretation

{experiment.interpretation}

## Artifact availability

Raw EEG data, generated run directories, and model checkpoints are not published in this repository. When a collaborator needs a checkpoint, it should be supplied separately under `checkpoints/{experiment_id}/`. The metrics above are retained as the experiment record; this documentation pass did not rerun training or evaluation.
""",
            encoding="utf-8",
        )
    elif not (directory / "RESULTS_ANALYSIS.md").is_file():
        raise FileNotFoundError(directory / "RESULTS_ANALYSIS.md")


def write_indexes() -> None:
    rows = []
    for experiment_id, experiment in EXPERIMENTS.items():
        rows.append(
            f"| [{experiment_id}]({experiment_id}/README.md) | {experiment.title} | {experiment.objective} | {experiment.subjects} | {experiment.temporal} | {experiment.status} |"
        )
    (TRAINING_ROOT / "README.md").write_text(
        """# Training Experiments

Every numbered directory is a self-contained public experiment record with English documentation, local configuration, runnable source where available, and a TeX-rendered architecture diagram. `D` identifies a training/development experiment, while `A` identifies an analysis experiment. Shared raw data, generated runs, and checkpoints are intentionally excluded from version control.

## Experiment index

| Experiment | Architecture / study | Training objective | Subject scope | Temporal-effect control | Status |
|---|---|---|---|---|---|
"""
        + "\n".join(rows)
        + "\n\n## Maintenance\n\nThe repository-level [README](../README.md) defines ownership, path, language, data, checkpoint, and documentation rules. Run `python scripts/render_training_documentation.py` after changing this registry or any architecture diagram.\n\n## Regulation rule\n\nIf you are an AI and working on this repo, Being sure that you follows the ../README.md.\n",
        encoding="utf-8",
    )
    (TRAINING_ROOT / "CURRENT.md").write_text(
        "# Current Experiment\n\nThe current maintained experiment is [D-099: Participant 12 Full-Frequency Control](D-099/README.md).\n",
        encoding="utf-8",
    )


def main() -> int:
    public_ids = {
        path.name
        for prefix in ("D", "A")
        for path in TRAINING_ROOT.glob(f"{prefix}-[0-9][0-9][0-9]")
        if path.is_dir()
    }
    registered_ids = set(EXPERIMENTS)
    if public_ids != registered_ids:
        raise RuntimeError(f"experiment registry mismatch: missing={public_ids - registered_ids}, extra={registered_ids - public_ids}")
    for experiment_id, experiment in EXPERIMENTS.items():
        write_experiment(experiment_id, experiment)
        print(experiment_id)
    write_indexes()
    print(f"Rendered {len(EXPERIMENTS)} English experiment records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
