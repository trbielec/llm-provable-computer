# Taxonomy of Hybrid Architectures Involving Rule-Based Reasoning and Machine Learning in Clinical Decision Systems

## Citation

- Title: *Taxonomy of hybrid architectures involving rule-based reasoning and machine learning in clinical decision systems: A scoping review*
- Authors: Slawomir Kierner, Jacek Kucharski, Zofia Kierner
- Journal: *Journal of Biomedical Informatics*, Volume 144, August 2023, 104428
- DOI: <https://doi.org/10.1016/j.jbi.2023.104428>
- Source page: <https://www.sciencedirect.com/science/article/pii/S1532046423001491>
- License: CC BY 4.0
- Retrieval date: 2026-03-23

## Extraction Note

This is a cleaned Markdown working copy produced from the rendered ScienceDirect article text using `agent-browser`. It focuses on the article sections needed to understand the taxonomy and the five hybrid approaches. Site navigation, figures, and the full reference list were omitted from this working copy.

## Review

This paper is a scoping review of 71 papers on clinical decision systems that combine rule-based reasoning with machine learning. Its main contribution is a taxonomy based on two questions:

1. Why do authors combine rules and ML?
2. Where do rules and ML sit relative to each other in the inference pipeline?

The paper identifies five architecture archetypes:

| Archetype | Count | Share | Core idea | Best fit | Main tradeoff |
| --- | ---: | ---: | --- | --- | --- |
| REML | 35 | 49.3% | Rules are embedded inside the ML architecture | When prior knowledge should shape the model itself | Highest integration, but weakest runtime transparency |
| MLRB | 17 | 23.9% | ML pre-processes data, rules make the final inference | When ML is useful for representation/feature extraction but outputs must stay clinically bounded | Final rule layer is explainable, but upstream ML can still be opaque |
| RBML | 12 | 16.9% | Rules pre-process data before ML prediction | When rules can denoise, segment, or engineer clinically meaningful features | Explainability is concentrated in preprocessing, not final prediction |
| RMLT | 5 | 7.0% | Rules shape the training process or feature selection | When data is small, high-dimensional, or uncertain | Rules help training, but may disappear from runtime reasoning |
| PERML | 2 | 2.8% | Rules and ML run in parallel and an aggregator combines outputs | When trust, disagreement handling, and auditability matter | Underexplored in the literature despite strong practical appeal |

### Practical Read of the Five Approaches

- `REML` is the dominant pattern because it often improves predictive performance while still allowing some prior knowledge to be injected into the model. In practice, this is the most "fused" approach.
- `MLRB` uses ML as a front-end and rules as a bounded decision layer. This is useful when you want learned representations but still want a rule engine to gate, validate, or constrain outputs.
- `RBML` uses rules as a structured preprocessing stage. The paper finds this especially common in imaging workflows, where rules help isolate or describe the relevant signal before ML classification.
- `RMLT` is less about runtime explainability and more about making training workable: feature selection, partitioning training data, or preserving clinically important variables.
- `PERML` is the least studied but arguably the most interesting for trustworthy systems. It lets rule-based and ML-based signals remain separate, which makes disagreement visible and audit trails clearer.

### Motivators Reported by the Review

- Prediction accuracy is the main driver. It is the primary motivator in about 75% of the reviewed papers and appears somewhere in about 95%.
- Data efficiency is next. It appears in almost 40% of papers, often because of small datasets, fuzzy clinical inputs, or computationally expensive data such as imaging.
- Conformance with clinical knowledge is rarely the primary driver, but it appears as a secondary reason when authors need alignment with guidelines or evolving clinical standards.
- Explainability and trustworthiness are surprisingly underemphasized relative to current regulatory and clinical concerns. Only three papers treat them as the main benefit.
- Health outcomes are barely measured. Only one paper treats actual post-deployment health outcomes as the primary motivation.

### Design Takeaways

- If the goal is raw predictive performance and tight knowledge-model fusion, `REML` is the default literature choice.
- If the goal is clinically bounded outputs, `MLRB` is attractive because the rule layer remains the final arbiter.
- If the goal is strong feature engineering with explicit clinical structure, `RBML` is a good fit.
- If the bottleneck is training with small, noisy, or high-dimensional data, `RMLT` is the right pattern family.
- If the goal is trust, human oversight, and explicit disagreement handling between symbolic and learned signals, `PERML` looks like the best research direction, even though it is currently rare.

## Cleaned Working Copy

### Highlights

- This paper reviews clinical decision system architectures utilizing a combination of rule-based reasoning and machine learning to identify recurrent architecture archetypes and motivations of their authors.
- Five distinct architecture archetypes were identified: Rules are Embedded in ML architecture (REML), ML pre-processes input data for Rule-Based inference (MLRB), Rule-Based method pre-processes input data for ML prediction (RBML), Rules influence ML training (RMLT), Parallel Ensemble of Rules and ML (PERML).
- Parallel ensembles that leverage the benefits of machine learning and rule-based reasoning are rare in the present literature and offer a promising avenue for further research and exploration.

### Abstract

**Background**  
As the application of Artificial Intelligence (AI) technologies increases in the healthcare sector, the industry faces a need to combine medical knowledge, often expressed as clinical rules, with advances in machine learning (ML), which offer high prediction accuracy at the expense of transparency of decision making.

**Purpose**  
This paper seeks to review the present literature, identify hybrid architecture patterns that incorporate rules and machine learning, and evaluate the rationale behind their selection to inform future development and research on the design of transparent and precise clinical decision systems.

**Methods**  
PubMed, IEEE Explore, and Google Scholar were queried in search for papers from 1992 to 2022, with the keywords: "clinical decision system", "hybrid clinical architecture", "machine learning and clinical rules". Excluded articles did not use both ML and rules or did not provide any explanation of employed architecture. A proposed taxonomy was used to organize the results, analyze them, and depict them in graphical and tabular form. Two researchers, one with expertise in rule-based systems and another in ML, reviewed identified papers and discussed the work to minimize bias, and the third one re-reviewed the work to ensure consistency of reporting.

**Results**  
The authors screened 957 papers and reviewed 71 that met their criteria. Five distinct architecture archetypes were determined: Rules are Embedded in ML architecture (REML) (most used), ML pre-processes input data for Rule-Based inference (MLRB), Rule-Based method pre-processes input data for ML prediction (RBML), Rules influence ML training (RMLT), Parallel Ensemble of Rules and ML (PERML), which was rarely observed in clinical contexts.

**Conclusions**  
Most architectures in the reviewed literature prioritize prediction accuracy over explainability and trustworthiness, which has led to more complex embedded approaches. Alternatively, parallel (PERML) architectures may be employed, allowing for a more transparent system that is easier to explain to patients and clinicians. The potential of this approach warrants further research.

**Other**  
A limitation of the study may be that it reviews scientific literature, while algorithms implemented in clinical practice may present different distributions of motivations and implementations of hybrid architectures.

### 1. Introduction

The paper frames hybrid clinical AI as a response to two simultaneous facts:

- healthcare decisions need human oversight, accountability, and adherence to accepted clinical rules;
- ML methods are increasingly powerful, but often hard to explain.

The authors position hybrid systems as a practical bridge between symbolic clinical knowledge and data-driven prediction. They also connect the topic to "Augmented Intelligence" rather than fully autonomous AI, with clinicians remaining the final decision-makers.

A key point in the introduction is that there was no concise taxonomy for describing how rules and ML are actually combined in clinical systems. The review is intended to supply that taxonomy and make tradeoffs easier to discuss.

### Table 1. Statement of Significance

- **Problem:** Clinical Decision Systems must be precise and trustworthy to be adopted in healthcare.
- **What is Already Known:** Machine learning approaches are increasingly used in Clinical Decision Systems while much of medical knowledge is represented through clinical rules.
- **What This Paper Adds:** This paper investigates the potential of hybrid architectures that combine the accepted clinical rules with the predictive power of machine learning. It identifies and evaluates the rationales behind selecting hybrid architectural patterns to inform future development and research of precise and trustworthy Clinical Decision Systems.

### 2. Rule-Based Inference

The paper defines rule-based systems broadly enough to include expert-authored decision trees, rule engines, fuzzy logic systems, and linear models when they remain understandable to humans. The important constraint is that the rules must be meaningfully interpretable and grounded in expert knowledge.

The authors emphasize a complementarity:

- rules provide interpretability and explicit incorporation of clinical knowledge;
- ML can extract patterns from data, especially where manual rule acquisition would be slow or incomplete.

That complementarity is the basis for the full taxonomy.

### 3. Methodology

The review follows PRISMA-ScR principles and asks two explicit questions:

1. Why would the author combine rules and ML?
2. What is the relative position of rules and ML in the inference pipeline?

Method summary:

- Search sources: PubMed, IEEE Xplore, Google Scholar
- Time coverage: 1992-2022, plus a March 2023 supplementary search
- Initial corpus after de-duplication: 957 papers
- Included after screening: 71 papers

The exclusion criteria removed papers that:

- used only rules or only ML rather than both;
- did not explain the hybrid architecture in enough detail;
- were judged to have poor scientific quality.

### 4. Taxonomy

The paper organizes the literature along two dimensions.

#### Motivators

1. Prediction Accuracy
2. Data Efficiency
3. Conformance with Clinical Knowledge
4. Explainability and Trustworthiness
5. Health Outcomes

#### Relative Position of Rules and ML

1. **Rules are Embedded in ML architecture (REML)**  
Rules are embedded in a neural network architecture forming a combined system.

2. **ML pre-processes input data for Rule-Based inference (MLRB)**  
An ML algorithm pre-processes data that is then used by a rule-based inference engine.

3. **Rule-Based method pre-processes input data for ML prediction (RBML)**  
Rules first pre-process input data for an ML model, which then produces the final output.

4. **Rules influence ML Training (RMLT)**  
Rules set parameters for the ML algorithm or influence the training process, for example via feature or data selection.

5. **Parallel Ensemble of Rules and ML (PERML)**  
Rule-based inference and an ML model execute in parallel, and an aggregator combines outputs into the final prediction.

### Table 2. Architecture Counts Derived From the Review

| Architecture | Papers |
| --- | ---: |
| REML | 35 |
| MLRB | 17 |
| RBML | 12 |
| RMLT | 5 |
| PERML | 2 |

### 5. Discussion of Prominent Architectural Patterns

#### 5.1. Rules Embedded in ML Architecture (REML)

This is the dominant pattern in the review. It includes approaches such as:

- ANFIS (Adaptive Neuro-Fuzzy Inference System)
- KBANN (Knowledge-Based Artificial Neural Networks)
- logic-neuron approaches
- generic fuzzy perceptron variants

Core idea:

- encode rules or prior knowledge into the model structure itself;
- then tune parameters using learning algorithms on historical data.

Why it is used:

- it preserves some explicit domain structure;
- it often improves predictive performance;
- it can be applied to both tabular clinical data and medical imaging.

Paper examples include breast cancer, brain tumor, lung cancer, CKD progression, COVID chest X-ray detection, dialysis control, and anesthesia monitoring/control.

The paper's judgment is that REML is powerful and common, but it is also the most entangled hybrid form. Rules are no longer a clearly separate runtime component; instead, they become part of the learned system.

#### 5.2. ML Pre-Processes Input Data for Rule-Based Inference (MLRB)

In this serial design:

1. ML transforms or extracts structure from raw inputs.
2. A rule-based system consumes that processed representation and makes the final decision.

Typical uses in the paper:

- forecasting COVID time series using fractal-dimension preprocessing plus fuzzy logic;
- preprocessing epilepsy symptoms before fuzzy inference;
- ICU infection prediction;
- MRI pain classification;
- seizure prediction from EEG.

Why it is useful:

- rules can validate, constrain, or reject implausible ML outputs;
- the final decision layer stays relatively interpretable;
- it is attractive when learned preprocessing is needed but final clinical judgment should remain rule-governed.

#### 5.3. Rules Pre-Process Input Data for ML (RBML)

This is the opposite serial arrangement:

1. rules or fuzzy methods preprocess the data;
2. ML then consumes the transformed features and produces the final output.

The paper highlights this pattern in imaging tasks, especially tumor detection and segmentation. Rules help with:

- contrast enhancement,
- boundary or edge detection,
- narrowing the candidate region,
- reducing feature complexity before ML classification.

The paper argues that this can improve prediction quality while also surfacing clinically meaningful intermediate structure, especially when preprocessing exposes the suspicious region in an image.

#### 5.4. Rules Influence ML Training (RMLT)

Here rules are not primarily part of runtime inference. Instead, they help shape training by:

- selecting features,
- partitioning training data,
- preserving clinically relevant variables,
- handling uncertainty in small or fuzzy datasets.

Examples discussed in the paper include:

- fuzzy partitions for elderly readmission prediction,
- fuzzy gain ratio for tumor classification with very high-dimensional gene-expression data,
- information diffusion and multicriteria methods for diabetes severity estimation.

The paper treats this as a useful but comparatively small category. It helps when the core problem is not runtime reasoning but training feasibility, data sparsity, or dimensionality.

#### 5.5. Parallel Ensemble of Rules and ML (PERML)

This is the most structurally transparent option in the paper's taxonomy:

1. a rule-based subsystem runs independently;
2. an ML subsystem runs independently;
3. an aggregator combines or compares their outputs.

The paper points out several advantages:

- disagreement can be surfaced explicitly;
- outputs can be averaged, gated, or rejected;
- uncertainty quantification can be layered in more naturally;
- rules remain visible as rules rather than disappearing into the learned model.

Only two reviewed papers used this design, in anesthesia control and acute abdomen diagnosis. The paper treats this rarity as a research gap rather than a sign that the approach is weak.

### 6. Discussion About Key Motivators

#### 6.1. Prediction Accuracy

Prediction accuracy is the dominant motivator in the literature. The paper notes that this is partly because scientific articles usually need a quantifiable performance claim, and many reviewed systems were not yet deployed deeply enough to report stronger outcome-based evidence.

#### 6.2. Data Efficiency

Data efficiency matters because clinical data is often:

- scarce,
- noisy,
- fuzzy,
- high-dimensional,
- or computationally expensive to process.

The paper repeatedly links hybrid methods to handling these constraints more gracefully than purely learned systems.

#### 6.3. Conformance With Clinical Knowledge

This appears less often as a primary motivator than the authors expected. The paper suggests this may be because relatively few systems in the review were already deployed in tightly regulated clinical workflows where formal conformance matters most.

#### 6.4. Explainability and Trustworthiness

The paper views this as underrepresented. Despite frequent discussion of trustworthy AI by regulators and researchers, very few reviewed papers made explainability their primary reason for the chosen architecture.

The authors explicitly suggest that architecture matters here:

- embedded approaches are least transparent,
- serial approaches are more explainable,
- parallel approaches may be the most confidence-building.

#### 6.5. Health Outcomes

The review finds almost no direct evidence linking architecture choice to actual measured health outcomes after deployment. That gap is important, because it means most architecture arguments in the literature still rest on proxy metrics such as prediction accuracy rather than clinical impact.

### 7. Conclusions and Directions

The paper's main conclusion is straightforward:

- hybrid rule/ML systems in clinical decision support fall into five recurring architecture types;
- the literature is dominated by embedded approaches focused on prediction accuracy;
- explainability, trustworthiness, and real health-outcome evidence are comparatively underdeveloped;
- parallel architectures deserve more attention because they may better support transparency, auditability, and clinician trust.

### 8. Limitations

The paper notes several review limitations:

- it only covers scientific literature, not necessarily what is deployed in production clinical settings;
- search terms and registries may have missed relevant work;
- only the top 100 results for broad queries were screened in some cases;
- the review was limited to English-language literature.

### Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Bottom Line

If the goal is to understand the "different approaches," the paper's contribution is not a new algorithm but a clean architecture map:

- `REML`: fuse rules into the model;
- `MLRB`: use ML first, then rules;
- `RBML`: use rules first, then ML;
- `RMLT`: use rules to shape training;
- `PERML`: keep rules and ML separate and combine them at the end.

For system design, the most useful insight is that these are not just implementation details. They reflect different priorities: predictive power, data efficiency, knowledge conformance, interpretability, and trust.
